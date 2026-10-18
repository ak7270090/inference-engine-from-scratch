"""Batched serving must give every request exactly what it would get alone."""

import asyncio

import pytest
import torch

from nanoengine.config import EngineSettings
from nanoengine.engine.generation.naive_generator import NaiveGenerator
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.metrics import EngineMetrics
from nanoengine.engine.runtime import create_engine
from nanoengine.engine.scheduling.batch_engine_loop import BatchEngineLoop
from nanoengine.engine.scheduling.request_queue import EngineOverloadedError
from nanoengine.engine.scheduling.serial_engine_loop import SerialEngineLoop

pytestmark = pytest.mark.anyio

CPU = torch.device("cpu")
PROMPTS = [[3, 9, 27], [1, 2, 3, 4, 5, 6, 7, 8], [11, 12], [40, 41, 42, 43, 44]]
NEW_TOKENS = 12


def build_request(request_id: str, prompt: list[int], max_new_tokens: int = NEW_TOKENS):
    return GenerationRequest(
        request_id, prompt, SamplingParams(max_new_tokens=max_new_tokens, ignore_eos=True)
    )


def build_settings(kv_cache: str, scheduler: str, **overrides) -> EngineSettings:
    values = {
        "kv_cache": kv_cache,
        "scheduler": scheduler,
        "max_model_len": 64,
        "kv_cache_memory_mb": 1,
        "kv_block_size": 4,
        "batch_fill_timeout_ms": 5,
    }
    return EngineSettings(**{**values, **overrides})


def expected_greedy_tokens(model, prompt: list[int]) -> list[int]:
    generator = NaiveGenerator(model, frozenset(), CPU)
    return [event.token_id for event in generator.generate(build_request("ref", prompt))]


async def collect_token_ids(engine, request: GenerationRequest) -> list[int]:
    return [event.token_id async for event in engine.submit(request)]


async def serve_all(engine, prompts: list[list[int]]) -> list[list[int]]:
    await engine.start()
    try:
        return await asyncio.gather(
            *(collect_token_ids(engine, build_request(f"r{i}", p)) for i, p in enumerate(prompts))
        )
    finally:
        await engine.stop()


@pytest.mark.parametrize(
    ("kv_cache", "scheduler"),
    [
        ("contiguous", "serial"),
        ("contiguous", "static"),
        ("contiguous", "continuous"),
        ("paged", "continuous"),
        ("paged", "static"),
    ],
)
async def test_every_mode_matches_generating_each_request_alone(
    tiny_model, kv_cache: str, scheduler: str
) -> None:
    metrics = EngineMetrics()
    engine = create_engine(tiny_model, frozenset(), build_settings(kv_cache, scheduler), metrics)

    outputs = await serve_all(engine, PROMPTS)

    assert outputs == [expected_greedy_tokens(tiny_model, prompt) for prompt in PROMPTS]
    peak = metrics.registry.get_sample_value("nanoengine_peak_running_requests")
    assert peak == (1 if scheduler == "serial" else len(PROMPTS))
    assert isinstance(engine, SerialEngineLoop if scheduler == "serial" else BatchEngineLoop)


async def test_preempted_sequences_are_recomputed_with_the_same_output(tiny_model) -> None:
    metrics = EngineMetrics()
    settings = build_settings("paged", "continuous", max_model_len=24)
    engine = create_engine(tiny_model, frozenset(), settings, metrics)
    engine._runner.allocator._blocks = type(engine._runner.allocator._blocks)(8)

    outputs = await serve_all(engine, PROMPTS)

    assert outputs == [expected_greedy_tokens(tiny_model, prompt) for prompt in PROMPTS]
    assert metrics.registry.get_sample_value("nanoengine_preemptions_total") > 0
    assert engine._runner.allocator.num_reserved_slots == 0


async def test_abandoned_request_frees_its_seat_and_memory(tiny_model) -> None:
    engine = create_engine(
        tiny_model, frozenset(), build_settings("paged", "continuous"), EngineMetrics()
    )
    await engine.start()

    abandoned = engine.submit(build_request("abandoned", [1, 2, 3], max_new_tokens=40))
    await anext(abandoned)
    await abandoned.aclose()
    finished = await collect_token_ids(engine, build_request("next", [4, 5]))
    await engine.stop()

    assert len(finished) == NEW_TOKENS
    assert engine._runner.allocator.num_reserved_slots == 0
    assert engine._running == [] and engine._waiting == type(engine._waiting)()


async def test_full_waiting_queue_rejects_new_requests(tiny_model) -> None:
    settings = build_settings("paged", "continuous", max_waiting_requests=1)
    engine = create_engine(tiny_model, frozenset(), settings, EngineMetrics())
    engine.submit(build_request("queued", [1]))

    with pytest.raises(EngineOverloadedError):
        engine.submit(build_request("rejected", [1]))


async def test_a_failing_step_reaches_every_request_in_it(tiny_model, monkeypatch) -> None:
    engine = create_engine(
        tiny_model, frozenset(), build_settings("paged", "continuous"), EngineMetrics()
    )
    monkeypatch.setattr(engine._runner, "run_step", raise_model_error)
    await engine.start()

    with pytest.raises(RuntimeError, match="model exploded"):
        await collect_token_ids(engine, build_request("broken", [1, 2]))
    await engine.stop()

    assert engine._runner.allocator.num_reserved_slots == 0


def raise_model_error(sequences) -> list[int]:
    raise RuntimeError("model exploded")


async def test_stop_without_start_does_nothing(tiny_model) -> None:
    engine = create_engine(
        tiny_model, frozenset(), build_settings("paged", "continuous"), EngineMetrics()
    )

    await engine.stop()


def test_batching_without_a_kv_cache_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs kv_cache"):
        EngineSettings(kv_cache="none", scheduler="continuous")
