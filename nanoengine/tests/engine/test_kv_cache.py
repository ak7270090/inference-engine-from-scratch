"""The cached path must give the same logits and tokens as recomputing everything (phase 1)."""

import pytest
import torch

from nanoengine.engine.generation.batch_builder import build_step_inputs
from nanoengine.engine.generation.cached_generator import CachedGenerator
from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.generation.naive_generator import NaiveGenerator
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.memory.contiguous_allocator import ContiguousSlotAllocator
from nanoengine.engine.memory.kv_cache_pool import (
    KVCachePool,
    count_kv_bytes_per_token,
    count_slots_for_budget,
)
from nanoengine.engine.memory.paged_allocator import PagedSlotAllocator
from nanoengine.engine.sequence_state import Sequence
from nanoengine.models.qwen3 import Qwen3Config
from tests.conftest import build_tiny_config

CPU = torch.device("cpu")
LOGITS_TOLERANCE = 1e-5
REGION_LENGTH = 32
BLOCK_SIZE = 4


def build_request(request_id: str, prompt: list[int], **sampling_options) -> GenerationRequest:
    return GenerationRequest(request_id, prompt, SamplingParams(**sampling_options))


def build_runner(model, allocator, num_slots: int) -> ModelRunner:
    pool = KVCachePool(model.config, num_slots, CPU, torch.float32)
    return ModelRunner(model, pool, allocator, CPU)


def build_contiguous_runner(model, num_regions: int = 4) -> ModelRunner:
    allocator = ContiguousSlotAllocator(num_regions, REGION_LENGTH, CPU)
    return build_runner(model, allocator, num_regions * REGION_LENGTH)


def build_paged_runner(model, num_blocks: int = 40) -> ModelRunner:
    allocator = PagedSlotAllocator(num_blocks, BLOCK_SIZE, CPU)
    return build_runner(model, allocator, num_blocks * BLOCK_SIZE)


def compute_full_logits(model, token_ids: list[int]) -> torch.Tensor:
    with torch.inference_mode():
        return model.compute_logits(model(torch.tensor([token_ids])))[0]


def run_greedy(runner: ModelRunner, sequences: list[Sequence], steps: int) -> None:
    for _ in range(steps):
        for sequence in sequences:
            runner.allocator.reserve(sequence, sequence.num_tokens)
        for sequence, token_id in zip(sequences, runner.run_step(sequences), strict=True):
            sequence.token_ids.append(token_id)


@pytest.mark.parametrize("build", [build_contiguous_runner, build_paged_runner])
def test_batched_cached_steps_match_recomputing_each_sequence(tiny_model, build) -> None:
    runner = build(tiny_model)
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10], [11]]
    sequences = [
        Sequence.from_request(build_request(f"r{index}", prompt, max_new_tokens=6), CPU)
        for index, prompt in enumerate(prompts)
    ]

    run_greedy(runner, sequences, steps=6)

    for sequence in sequences:
        expected = list(sequence.token_ids[: sequence.num_prompt_tokens])
        for _ in range(6):
            expected.append(int(compute_full_logits(tiny_model, expected)[-1].argmax()))
        assert sequence.token_ids == expected


def test_cached_decode_logits_match_full_recompute(tiny_model) -> None:
    runner = build_contiguous_runner(tiny_model, num_regions=1)
    sequence = Sequence.from_request(build_request("r", [5, 9, 2, 7], max_new_tokens=4), CPU)

    run_greedy(runner, [sequence], steps=3)
    runner.allocator.reserve(sequence, sequence.num_tokens)
    inputs = build_step_inputs([sequence], runner.allocator, runner._pool, CPU)
    with torch.inference_mode():
        hidden_states = tiny_model(inputs.token_ids, inputs.context)
        cached_logits = tiny_model.compute_logits(hidden_states)[0, -1]

    expected_logits = compute_full_logits(tiny_model, sequence.token_ids)[-1]
    torch.testing.assert_close(cached_logits, expected_logits, atol=LOGITS_TOLERANCE, rtol=0)


def test_growing_sequences_get_interleaved_blocks(tiny_model) -> None:
    runner = build_paged_runner(tiny_model, num_blocks=12)
    first = Sequence.from_request(build_request("a", [1, 2, 3, 4], max_new_tokens=4), CPU)
    second = Sequence.from_request(build_request("b", [5, 6, 7, 8], max_new_tokens=4), CPU)

    run_greedy(runner, [first, second], steps=4)

    assert runner.allocator.block_table(first) == [0, 2]
    assert runner.allocator.block_table(second) == [1, 3]


def test_cached_generator_matches_naive_greedy_output(tiny_model) -> None:
    request = build_request("r", [3, 9, 27, 1], max_new_tokens=10, ignore_eos=True)
    naive = NaiveGenerator(tiny_model, frozenset(), CPU)
    cached = CachedGenerator(build_contiguous_runner(tiny_model, num_regions=1), frozenset(), CPU)

    naive_tokens = [event.token_id for event in naive.generate(request)]
    cached_tokens = [event.token_id for event in cached.generate(request)]

    assert cached_tokens == naive_tokens
    assert cached.vocab_size == tiny_model.config.vocab_size


def test_cached_generator_frees_memory_after_each_request(tiny_model) -> None:
    runner = build_contiguous_runner(tiny_model, num_regions=1)
    generator = CachedGenerator(runner, frozenset(), CPU)

    for index in range(3):
        list(generator.generate(build_request(f"r{index}", [1, 2], max_new_tokens=3)))

    assert runner.allocator.num_reserved_slots == 0


def test_kv_bytes_per_token_follow_the_formula() -> None:
    qwen3_0_6b = Qwen3Config(151936, 1024, 3072, 28, 16, 8, 128, 1e-6, 1e6, True)

    assert count_kv_bytes_per_token(qwen3_0_6b, torch.float16) == 2 * 28 * 8 * 128 * 2 == 114688
    assert count_slots_for_budget(build_tiny_config(), torch.float32, 1) == 1024 * 1024 // (
        2 * 2 * 2 * 16 * 4
    )
