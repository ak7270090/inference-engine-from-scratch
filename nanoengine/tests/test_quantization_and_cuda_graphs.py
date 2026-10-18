"""Phase 7: int8 weights and KV cache stay close to full precision; the CUDA graph decode path
(padded to fixed buckets) gives exactly the same tokens as the eager path."""

import asyncio

import pytest
import torch
from torch import nn

from nanoengine import quality_eval
from nanoengine.config import EngineSettings
from nanoengine.engine.generation.cuda_graph_runner import (
    DecodeGraphRunner,
    build_buckets,
    build_context_buckets,
    round_up_to_bucket,
)
from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.memory.kv_cache_pool import KVCachePool, count_kv_bytes_per_token
from nanoengine.engine.memory.paged_allocator import PagedSlotAllocator
from nanoengine.engine.metrics import EngineMetrics
from nanoengine.engine.runtime import create_engine
from nanoengine.engine.sequence_state import Sequence
from nanoengine.models.int8_quantization import (
    Int8Linear,
    quantize_linear_layers,
    quantize_vectors,
)
from nanoengine.models.qwen3 import Qwen3Config
from nanoengine.quality_eval import evaluate_quality, render_quality_markdown
from nanoengine.step_timing import CachedDecodeStepBenchmark, measure_step_timings
from tests.conftest import build_tiny_model
from tests.test_batch_engine_loop import (
    PROMPTS,
    build_request,
    build_settings,
    collect_token_ids,
    expected_greedy_tokens,
)
from tests.test_kv_cache import run_greedy

CPU = torch.device("cpu")


def test_int8_vectors_round_trip_within_half_a_step() -> None:
    values = torch.randn(6, 4, 16)

    quantized, scales = quantize_vectors(values)

    assert quantized.dtype == torch.int8 and scales.shape == (6, 4, 1)
    assert torch.all((quantized.float() * scales - values).abs() <= scales / 2 + 1e-6)


def test_int8_linear_stays_close_to_the_full_precision_layer() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(64, 32, bias=False)
    inputs = torch.randn(5, 64)

    quantized = Int8Linear.from_linear(linear)

    relative_error = (quantized(inputs) - linear(inputs)).norm() / linear(inputs).norm()
    assert quantized.weight_int8.dtype == torch.int8
    assert relative_error < 0.02


def test_quantized_model_keeps_its_predictions(tiny_model) -> None:
    token_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 12))
    with torch.inference_mode():
        full_logits = tiny_model.compute_logits(tiny_model(token_ids))
    quantized_model = build_tiny_model()

    replaced = quantize_linear_layers(quantized_model)
    with torch.inference_mode():
        int8_logits = quantized_model.compute_logits(quantized_model(token_ids))

    assert replaced == 7 * tiny_model.config.num_hidden_layers
    similarity = torch.nn.functional.cosine_similarity(full_logits, int8_logits, dim=-1)
    assert similarity.min() > 0.99


def test_int8_kv_cache_halves_the_bytes_per_token() -> None:
    qwen3_0_6b = Qwen3Config(151936, 1024, 3072, 28, 16, 8, 128, 1e-6, 1e6, True)

    full = count_kv_bytes_per_token(qwen3_0_6b, torch.float16)
    int8 = count_kv_bytes_per_token(qwen3_0_6b, torch.float16, is_int8_cache=True)

    assert int8 == 2 * 28 * 8 * (128 + 2)
    assert full / int8 > 1.95


def build_paged_runner(model, is_int8_cache: bool) -> ModelRunner:
    pool = KVCachePool(model.config, 64, CPU, torch.float32, is_int8_cache)
    return ModelRunner(model, pool, PagedSlotAllocator(16, 4, CPU), CPU)


def build_runner_with_graph_path(model, use_graph_path: bool) -> ModelRunner:
    allocator = PagedSlotAllocator(40, 4, CPU)
    pool = KVCachePool(model.config, 160, CPU, torch.float32)
    graphs = DecodeGraphRunner(model, pool, allocator, CPU, 8, 64)
    return ModelRunner(model, pool, allocator, CPU, graphs if use_graph_path else None)


def test_int8_kv_cache_matches_full_precision_logits_closely(tiny_model) -> None:
    prompts = [[3, 9, 27, 1, 5, 6], [2, 4]]
    results = []
    for is_int8 in (False, True):
        sequences = [
            Sequence.from_request(build_request(f"r{i}", prompt), CPU)
            for i, prompt in enumerate(prompts)
        ]
        run_greedy(build_paged_runner(tiny_model, is_int8), sequences, steps=8)
        results.append([sequence.token_ids for sequence in sequences])

    agreement = sum(
        full == int8
        for full_tokens, int8_tokens in zip(results[0], results[1], strict=True)
        for full, int8 in zip(full_tokens, int8_tokens, strict=True)
    ) / sum(len(tokens) for tokens in results[0])
    assert agreement >= 0.9


def test_buckets_cover_every_size_up_to_the_limit() -> None:
    assert build_buckets(48, (1, 2, 4, 8, 16, 32, 64)) == [1, 2, 4, 8, 16, 32, 48]
    assert build_context_buckets(2048) == [256, 512, 1024, 2048]
    assert build_context_buckets(100) == [100]
    assert round_up_to_bucket(3, [1, 2, 4, 8]) == 4


def test_padded_decode_path_gives_the_same_tokens_as_eager(tiny_model) -> None:
    outputs = []
    for use_graph_path in (False, True):
        runner = build_runner_with_graph_path(tiny_model, use_graph_path)
        sequences = [
            Sequence.from_request(build_request(f"r{i}", prompt), CPU)
            for i, prompt in enumerate(PROMPTS[:3])
        ]
        run_greedy(runner, sequences, steps=10)
        outputs.append([sequence.token_ids for sequence in sequences])

    assert outputs[0] == outputs[1]
    assert runner.decode_graphs.num_captured_graphs == 0  # CPU: same padded path, no capture
    runner.decode_graphs.capture_all_buckets()
    assert runner.decode_graphs.num_captured_graphs == 0
    assert runner.decode_graphs.can_run(sequences)
    assert not runner.decode_graphs.can_run([*sequences] * 3)


@pytest.mark.anyio
async def test_phase_7_engine_matches_alone_generation_with_graph_path(tiny_model) -> None:
    settings = build_settings("paged", "continuous", cuda_graphs=True, max_num_seqs=4)
    engine = create_engine(tiny_model, frozenset(), settings, EngineMetrics())
    await engine.start()

    outputs = await asyncio.gather(
        *(collect_token_ids(engine, build_request(f"r{i}", p)) for i, p in enumerate(PROMPTS))
    )
    await engine.stop()

    assert outputs == [expected_greedy_tokens(tiny_model, prompt) for prompt in PROMPTS]


def test_int8_settings_build_a_quantized_engine(tiny_model) -> None:
    settings = build_settings("paged", "continuous", quantization="int8", kv_cache_dtype="int8")

    engine = create_engine(tiny_model, frozenset(), settings, EngineMetrics())

    assert isinstance(tiny_model.model.layers[0].self_attn.q_proj, Int8Linear)
    assert engine._runner._pool.key_caches[0].dtype == torch.int8


def test_int8_kv_cache_and_graphs_need_a_kv_cache() -> None:
    with pytest.raises(ValueError, match="need a KV cache"):
        EngineSettings(kv_cache="none", cuda_graphs=True)


def test_quality_report_compares_against_full_precision(tiny_model, tmp_path) -> None:
    settings = EngineSettings(quantization="int8", kv_cache_dtype="int8", kv_cache="contiguous")
    quantized_model = build_tiny_model()
    quantize_linear_layers(quantized_model)
    token_ids = [index % tiny_model.config.vocab_size for index in range(200)]

    report = evaluate_quality(tiny_model, quantized_model, token_ids, settings)

    assert report.perplexity_full_precision > 0 and report.perplexity_configured > 0
    assert 0.0 <= report.greedy_token_agreement <= 1.0
    assert "Greedy token agreement" in render_quality_markdown(report)


def test_quality_eval_cli_writes_its_report(monkeypatch, tiny_model, tiny_tokenizer, tmp_path):
    monkeypatch.setenv("NANOENGINE_QUANTIZATION", "int8")
    monkeypatch.setattr(quality_eval, "download_model_files", lambda model_id, cache_dir: tmp_path)
    monkeypatch.setattr(
        quality_eval, "load_qwen3_model", lambda model_dir, device, dtype: tiny_model
    )
    monkeypatch.setattr(quality_eval, "load_text_tokenizer", lambda model_dir: tiny_tokenizer)
    monkeypatch.setattr(quality_eval, "QUALITY_TEXT", " ".join(f"w{i % 60}" for i in range(220)))

    quality_eval.main(["--output-dir", str(tmp_path)])

    assert (tmp_path / "quality-int8.json").exists()


def test_step_timing_measures_the_graph_path(tiny_model, tmp_path) -> None:
    benchmark = CachedDecodeStepBenchmark(tiny_model, CPU, torch.float32, 16, True, True)

    timings = measure_step_timings(benchmark, [4, 16], 1, CPU, tiny_model.config.vocab_size)

    assert benchmark.description == "KV cache, CUDA graphs"
    assert len(timings) == 2


def report_plenty_of_free_memory(device) -> tuple[int, int]:
    return 15 * 1024**3, 16 * 1024**3


def raise_out_of_memory(decode_graph) -> None:
    raise torch.cuda.OutOfMemoryError("pretend the GPU is full")


def test_a_failed_capture_falls_back_to_the_eager_path(tiny_model, monkeypatch) -> None:
    runner = build_runner_with_graph_path(tiny_model, use_graph_path=True)
    graphs = runner.decode_graphs
    monkeypatch.setattr(graphs, "_uses_cuda_graphs", True)
    monkeypatch.setattr(graphs, "_capture", raise_out_of_memory)
    monkeypatch.setattr(torch.cuda, "mem_get_info", report_plenty_of_free_memory)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    sequences = [Sequence.from_request(build_request("r", PROMPTS[0]), CPU)]

    run_greedy(runner, sequences, steps=4)

    assert graphs.num_captured_graphs == 0
    assert (
        sequences[0].token_ids[len(PROMPTS[0]) :]
        == expected_greedy_tokens(tiny_model, PROMPTS[0])[:4]
    )


def report_little_free_memory(device) -> tuple[int, int]:
    return 1024**3, 16 * 1024**3


def test_a_capture_that_would_eat_the_reserve_is_skipped(tiny_model, monkeypatch) -> None:
    runner = build_runner_with_graph_path(tiny_model, use_graph_path=True)
    graphs = runner.decode_graphs
    monkeypatch.setattr(graphs, "_uses_cuda_graphs", True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", report_little_free_memory)
    sequences = [Sequence.from_request(build_request("r", PROMPTS[1]), CPU)]

    run_greedy(runner, sequences, steps=3)

    assert graphs.num_captured_graphs == 0
    assert sequences[0].num_generated_tokens == 3
