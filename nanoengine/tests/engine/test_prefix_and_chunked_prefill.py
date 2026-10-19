"""Phase 5: shared prompt starts reuse KV blocks; long prompts are prefilled in chunks."""

import asyncio
from collections import deque

import pytest
import torch
from hypothesis import given, settings, strategies

from nanoengine.config import EngineSettings
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.memory.prefix_cache import (
    CachingBlockPool,
    PrefixCachingSlotAllocator,
    build_block_keys,
)
from nanoengine.engine.memory.slot_allocator import OutOfKVMemoryError
from nanoengine.engine.metrics import EngineMetrics
from nanoengine.engine.runtime import create_engine
from nanoengine.engine.scheduling.batch_scheduler import ChunkedPrefillScheduler
from nanoengine.engine.sequence_state import Sequence
from tests.engine.test_batch_engine_loop import (
    build_request,
    build_settings,
    collect_token_ids,
    expected_greedy_tokens,
)

CPU = torch.device("cpu")
BLOCK_SIZE = 4
SHARED_PREFIX = [5, 6, 7, 8, 9, 10, 11, 12]


def build_sequence(request_id: str, token_ids: list[int]) -> Sequence:
    request = GenerationRequest(request_id, token_ids, SamplingParams(max_new_tokens=4))
    return Sequence.from_request(request, CPU)


def cache_everything(allocator: PrefixCachingSlotAllocator, sequence: Sequence) -> None:
    """What a prefill step does to the allocator: all tokens are now in the cache."""
    allocator.reserve(sequence, sequence.num_tokens)
    sequence.num_cached_tokens = sequence.num_tokens
    allocator.on_tokens_cached(sequence)


def test_block_keys_include_the_whole_prefix() -> None:
    first = build_block_keys([1, 2, 3, 4, 9, 9, 9, 9], BLOCK_SIZE, 2)
    second = build_block_keys([0, 2, 3, 4, 9, 9, 9, 9], BLOCK_SIZE, 2)

    assert first[1][1] == second[1][1] == (9, 9, 9, 9)
    assert first[1] != second[1]


def test_pool_reuses_empty_blocks_first_then_evicts_the_oldest_cached() -> None:
    pool = CachingBlockPool(2)
    old, new = pool.allocate(), pool.allocate()
    pool.register(old, (0, (1,)))
    pool.register(new, (0, (2,)))
    pool.free(old)
    pool.free(new)

    reused = pool.allocate()

    assert reused == old
    assert pool.find((0, (1,))) is None and pool.find((0, (2,))) == new
    pool.share(new)
    assert pool.num_free_blocks == 0
    with pytest.raises(OutOfKVMemoryError):
        pool.allocate()
    pool.free(new)
    pool.free(reused)
    with pytest.raises(ValueError, match="not allocated"):
        pool.free(reused)


def test_second_request_shares_the_cached_prefix_blocks() -> None:
    allocator = PrefixCachingSlotAllocator(num_blocks=8, block_size=BLOCK_SIZE, device=CPU)
    first = build_sequence("first", [*SHARED_PREFIX, 1, 2])
    cache_everything(allocator, first)

    second = build_sequence("second", [*SHARED_PREFIX, 3])
    allocator.reserve(second, second.num_tokens)

    assert second.num_cached_tokens == 8
    assert allocator.block_table(second)[:2] == allocator.block_table(first)[:2]
    assert allocator.num_prefix_hit_tokens == 8
    allocator.free(first)
    allocator.free(second)
    assert allocator.num_free_blocks == 8


def test_an_exact_repeat_still_recomputes_its_last_token() -> None:
    allocator = PrefixCachingSlotAllocator(num_blocks=8, block_size=BLOCK_SIZE, device=CPU)
    cache_everything(allocator, build_sequence("first", SHARED_PREFIX))

    repeat = build_sequence("repeat", SHARED_PREFIX)
    allocator.reserve(repeat, repeat.num_tokens)

    assert repeat.num_cached_tokens == 4
    assert repeat.num_uncached_tokens == 4


def test_freed_cached_blocks_are_revived_and_counted_as_used() -> None:
    allocator = PrefixCachingSlotAllocator(num_blocks=3, block_size=BLOCK_SIZE, device=CPU)
    first = build_sequence("first", [*SHARED_PREFIX, 1])
    cache_everything(allocator, first)
    allocator.free(first)
    second = build_sequence("second", [*SHARED_PREFIX, 2])

    assert allocator.can_reserve(second, second.num_tokens)
    assert not allocator.can_reserve(build_sequence("other", list(range(30, 43))), 13)
    allocator.reserve(second, second.num_tokens)
    assert second.num_cached_tokens == 8 and allocator.num_free_blocks == 0


@settings(max_examples=50, deadline=None)
@given(
    strategies.lists(
        strategies.tuples(strategies.integers(0, 3), strategies.integers(1, 3)), max_size=30
    )
)
def test_reference_counts_never_lose_or_double_count_a_block(operations) -> None:
    allocator = PrefixCachingSlotAllocator(num_blocks=12, block_size=BLOCK_SIZE, device=CPU)
    live: dict[int, Sequence] = {}
    for index, (slot, prefix_blocks) in enumerate(operations):
        if slot in live:
            allocator.free(live.pop(slot))
            continue
        sequence = build_sequence(f"s{index}", SHARED_PREFIX[: prefix_blocks * 2] + [index, 1])
        if allocator.can_reserve(sequence, sequence.num_tokens):
            cache_everything(allocator, sequence)
            live[slot] = sequence
        tables = [allocator.block_table(sequence) for sequence in live.values()]
        distinct_owned = {block for table in tables for block in table}
        assert len(distinct_owned) + allocator.num_free_blocks == 12
    for sequence in live.values():
        allocator.free(sequence)
    assert allocator.num_free_blocks == 12


def test_chunked_scheduler_shares_the_token_budget_with_decodes() -> None:
    allocator = PrefixCachingSlotAllocator(num_blocks=40, block_size=BLOCK_SIZE, device=CPU)
    scheduler = ChunkedPrefillScheduler(allocator, max_num_seqs=8, max_num_batched_tokens=6)
    decoding = build_sequence("decoding", [1, 2])
    cache_everything(allocator, decoding)
    decoding.token_ids.append(3)
    long_prompt = build_sequence("long", list(range(100, 120)))
    running = [decoding]

    plan = scheduler.schedule(deque([long_prompt]), running, now=0.0)

    assert plan.sequences == [decoding]
    assert [(sequence.request_id, size) for sequence, size in plan.prefill_chunks] == [("long", 5)]


@pytest.mark.parametrize(
    "overrides",
    [
        {"prefix_cache": True},
        {"chunked_prefill": True, "max_num_batched_tokens": 5},
        {"prefix_cache": True, "chunked_prefill": True, "max_num_batched_tokens": 3},
    ],
)
@pytest.mark.anyio
async def test_phase_5_modes_match_generating_each_request_alone(tiny_model, overrides) -> None:
    metrics = EngineMetrics()
    settings_ = build_settings("paged", "continuous", **overrides)
    engine = create_engine(tiny_model, frozenset(), settings_, metrics)
    prompts = [[*SHARED_PREFIX, 1, 2, 3], [*SHARED_PREFIX, 4], list(range(20, 37)), [7]]
    await engine.start()

    first = await collect_token_ids(engine, build_request("warm", prompts[0]))
    others = await asyncio.gather(
        *(collect_token_ids(engine, build_request(f"r{i}", p)) for i, p in enumerate(prompts))
    )
    await engine.stop()

    assert [first, *others] == [
        expected_greedy_tokens(tiny_model, prompt) for prompt in [prompts[0], *prompts]
    ]
    hit_tokens = metrics.registry.get_sample_value("nanoengine_prefix_cache_hit_tokens")
    assert (hit_tokens > 0) == overrides.get("prefix_cache", False)
    mixed_steps = metrics.registry.get_sample_value(
        "nanoengine_step_latency_seconds_count", {"step_kind": "mixed"}
    )
    assert (mixed_steps is not None) == overrides.get("chunked_prefill", False)


def test_phase_5_flags_need_the_right_modes() -> None:
    with pytest.raises(ValueError, match="prefix_cache needs kv_cache=paged"):
        EngineSettings(kv_cache="contiguous", scheduler="continuous", prefix_cache=True)
    with pytest.raises(ValueError, match="chunked_prefill needs scheduler=continuous"):
        EngineSettings(kv_cache="paged", scheduler="static", chunked_prefill=True)
