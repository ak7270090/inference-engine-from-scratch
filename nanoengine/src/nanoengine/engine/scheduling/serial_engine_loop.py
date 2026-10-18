"""``scheduler=serial``: one background task serves requests strictly one at a time."""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Protocol

from nanoengine.constants import ENGINE_NAME
from nanoengine.engine.interface import GenerationRequest, TokenEvent
from nanoengine.engine.metrics import STEP_KIND_DECODE, STEP_KIND_PREFILL, EngineMetrics
from nanoengine.engine.scheduling.request_queue import (
    EngineOverloadedError,
    RequestHandle,
    stream_token_events,
)

logger = logging.getLogger(__name__)


class TokenGenerator(Protocol):
    """``NaiveGenerator`` (no cache) or ``CachedGenerator`` (KV cache)."""

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]: ...


class SerialEngineLoop:
    def __init__(
        self, generator: TokenGenerator, metrics: EngineMetrics, max_waiting_requests: int
    ) -> None:
        self._generator = generator
        self._metrics = metrics
        self._waiting: asyncio.Queue[RequestHandle] = asyncio.Queue(maxsize=max_waiting_requests)
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._serve_forever())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task

    def submit(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]:
        """Queue ``request`` now (or raise if full) and return its token events as they arrive."""
        handle = RequestHandle(request)
        try:
            self._waiting.put_nowait(handle)
        except asyncio.QueueFull as error:
            raise EngineOverloadedError(
                f"{self._waiting.maxsize} requests already waiting"
            ) from error
        self._metrics.waiting_requests.set(self._waiting.qsize())
        return stream_token_events(handle)

    async def _serve_forever(self) -> None:
        while True:
            handle = await self._waiting.get()
            self._metrics.waiting_requests.set(self._waiting.qsize())
            if not handle.is_cancelled:
                await self._serve_request(handle)

    async def _serve_request(self, handle: RequestHandle) -> None:
        """Run each model step in a worker thread so the event loop keeps streaming."""
        request = handle.request
        logger.debug(
            "start request",
            extra={
                "request_id": request.request_id,
                "engine": ENGINE_NAME,
                "num_waiting": self._waiting.qsize(),
            },
        )
        self._metrics.set_running_requests(1)
        token_events = self._generator.generate(request)
        step_kind = STEP_KIND_PREFILL
        try:
            while not handle.is_cancelled:
                step_started_at = time.perf_counter()
                event = await asyncio.to_thread(next, token_events, None)
                if event is None:
                    break
                elapsed_seconds = time.perf_counter() - step_started_at
                self._metrics.step_latency_seconds.labels(step_kind).observe(elapsed_seconds)
                step_kind = STEP_KIND_DECODE
                self._metrics.generated_tokens.inc()
                handle.token_events.put_nowait(event)
                if event.is_finished:
                    self._metrics.finished_requests.labels(event.finish_reason).inc()
                    break
        except Exception as error:
            logger.exception("generation failed", extra={"request_id": request.request_id})
            handle.token_events.put_nowait(error)
        finally:
            token_events.close()
            self._metrics.set_running_requests(0)
