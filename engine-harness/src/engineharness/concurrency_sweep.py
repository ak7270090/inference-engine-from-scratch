"""Run a workload with N users in parallel (closed loop) and repeat for each concurrency level."""

import asyncio
import time

import aiohttp

from engineharness.latency_metrics import WorkloadMetrics, summarize_timings
from engineharness.streaming_client import RequestTiming, stream_completion
from engineharness.workloads import CompletionRequest


async def run_closed_loop(
    session: aiohttp.ClientSession,
    requests: list[CompletionRequest],
    concurrency: int,
    model_id: str,
) -> tuple[list[RequestTiming], float]:
    """``concurrency`` users each send their next request as soon as the previous one ends."""
    queue: asyncio.Queue[CompletionRequest] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    timings: list[RequestTiming] = []
    started_at = time.perf_counter()
    await asyncio.gather(
        *(_run_user(session, queue, model_id, timings) for _ in range(concurrency))
    )
    return timings, time.perf_counter() - started_at


async def _run_user(
    session: aiohttp.ClientSession,
    queue: asyncio.Queue[CompletionRequest],
    model_id: str,
    timings: list[RequestTiming],
) -> None:
    """One simulated user: take the next request until the queue is empty."""
    while not queue.empty():
        request = queue.get_nowait()
        timings.append(await stream_completion(session, request, model_id))


async def run_concurrency_level(
    session: aiohttp.ClientSession,
    requests: list[CompletionRequest],
    concurrency: int,
    model_id: str,
    workload: str,
) -> WorkloadMetrics:
    """One closed-loop run, summarized."""
    timings, wall_seconds = await run_closed_loop(session, requests, concurrency, model_id)
    return summarize_timings(timings, wall_seconds, workload, concurrency)


async def run_concurrency_sweep(
    session: aiohttp.ClientSession,
    requests: list[CompletionRequest],
    concurrency_levels: list[int],
    model_id: str,
    workload: str,
) -> list[WorkloadMetrics]:
    """The same requests at every concurrency level, one level after another."""
    return [
        await run_concurrency_level(session, requests, level, model_id, workload)
        for level in concurrency_levels
    ]
