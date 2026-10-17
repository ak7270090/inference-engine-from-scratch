import asyncio
import time
from collections.abc import Iterator

import pytest
import torch

from nanoengine.engine_interface import GenerationRequest, SamplingParams, TokenEvent
from nanoengine.engine_loop import EngineLoop, EngineOverloadedError
from nanoengine.engine_metrics import EngineMetrics
from nanoengine.naive_generator import NaiveGenerator

pytestmark = pytest.mark.anyio

FAKE_STEP_SECONDS = 0.02


class SlowFakeGenerator(NaiveGenerator):
    """Emits token 0, 1, 2, ... with a fixed delay and records when each request starts and ends."""

    def __init__(self) -> None:
        self.spans: dict[str, list[float]] = {}

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        self.spans[request.request_id] = [time.perf_counter()]
        max_new_tokens = request.sampling_params.max_new_tokens
        for step_index in range(max_new_tokens):
            time.sleep(FAKE_STEP_SECONDS)
            is_finished = step_index + 1 == max_new_tokens
            if is_finished:
                self.spans[request.request_id].append(time.perf_counter())
            yield TokenEvent(
                request.request_id, step_index, is_finished, "length" if is_finished else None
            )


class FailingGenerator(NaiveGenerator):
    def __init__(self) -> None:
        pass

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        raise RuntimeError("model exploded")
        yield


def build_request(request_id: str, max_new_tokens: int = 3) -> GenerationRequest:
    return GenerationRequest(request_id, [1, 2], SamplingParams(max_new_tokens=max_new_tokens))


async def collect_token_ids(engine: EngineLoop, request: GenerationRequest) -> list[int]:
    return [event.token_id async for event in engine.submit(request)]


async def test_requests_are_served_one_after_another() -> None:
    generator = SlowFakeGenerator()
    engine = EngineLoop(generator, EngineMetrics(), max_waiting_requests=4)
    await engine.start()

    results = await asyncio.gather(
        collect_token_ids(engine, build_request("first")),
        collect_token_ids(engine, build_request("second")),
    )
    await engine.stop()

    assert results == [[0, 1, 2], [0, 1, 2]]
    assert generator.spans["second"][0] >= generator.spans["first"][1]


async def test_full_queue_rejects_new_requests() -> None:
    engine = EngineLoop(SlowFakeGenerator(), EngineMetrics(), max_waiting_requests=1)
    engine.submit(build_request("queued"))

    with pytest.raises(EngineOverloadedError):
        engine.submit(build_request("rejected"))


async def test_generation_error_reaches_the_caller() -> None:
    engine = EngineLoop(FailingGenerator(), EngineMetrics(), max_waiting_requests=1)
    await engine.start()

    with pytest.raises(RuntimeError, match="model exploded"):
        await collect_token_ids(engine, build_request("broken"))
    await engine.stop()


async def test_abandoned_request_stops_generating() -> None:
    generator = SlowFakeGenerator()
    engine = EngineLoop(generator, EngineMetrics(), max_waiting_requests=4)
    await engine.start()

    token_events = engine.submit(build_request("abandoned", max_new_tokens=50))
    await anext(token_events)
    await token_events.aclose()
    remaining = await collect_token_ids(engine, build_request("next"))
    await engine.stop()

    assert remaining == [0, 1, 2]
    assert len(generator.spans["abandoned"]) == 1


async def test_stop_without_start_does_nothing() -> None:
    await EngineLoop(SlowFakeGenerator(), EngineMetrics(), max_waiting_requests=1).stop()


async def test_real_generator_runs_inside_the_loop(tiny_model) -> None:
    generator = NaiveGenerator(tiny_model, frozenset(), torch.device("cpu"))
    engine = EngineLoop(generator, EngineMetrics(), max_waiting_requests=1)
    await engine.start()

    token_ids = await collect_token_ids(engine, build_request("real", max_new_tokens=4))
    await engine.stop()

    assert len(token_ids) == 4
