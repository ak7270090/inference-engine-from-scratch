"""``scheduler=static|continuous``: one background loop runs batched steps for many requests.

Every step: drop cancelled requests, let the scheduler pick the sequences, run one model step in a
worker thread (so streaming continues), then hand each sequence its new token.
"""

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import AsyncIterator

import torch

from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.interface import GenerationRequest, TokenEvent
from nanoengine.engine.metrics import (
    STEP_KIND_DECODE,
    STEP_KIND_MIXED,
    STEP_KIND_PREFILL,
    EngineMetrics,
)
from nanoengine.engine.sampling import find_finish_reason
from nanoengine.engine.scheduling.batch_scheduler import ContinuousBatchScheduler, StepPlan
from nanoengine.engine.scheduling.request_queue import (
    EngineOverloadedError,
    RequestHandle,
    stream_token_events,
)
from nanoengine.engine.sequence_state import Sequence

logger = logging.getLogger(__name__)


class BatchEngineLoop:
    def __init__(
        self,
        runner: ModelRunner,
        scheduler: ContinuousBatchScheduler,
        metrics: EngineMetrics,
        stop_token_ids: frozenset[int],
        max_waiting_requests: int,
        device: torch.device,
    ) -> None:
        self._runner = runner
        self._scheduler = scheduler
        self._metrics = metrics
        self._stop_token_ids = stop_token_ids
        self._max_waiting_requests = max_waiting_requests
        self._device = device
        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []
        self._handles: dict[str, RequestHandle] = {}
        self._has_work = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        metrics.kv_cache_total_slots.set(runner.allocator.num_total_slots)

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._serve_forever())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task

    def submit(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]:
        if len(self._waiting) >= self._max_waiting_requests:
            raise EngineOverloadedError(f"{self._max_waiting_requests} requests already waiting")
        handle = RequestHandle(request)
        self._handles[request.request_id] = handle
        self._waiting.append(Sequence.from_request(request, self._device))
        self._has_work.set()
        self._update_gauges()
        return stream_token_events(handle)

    async def _serve_forever(self) -> None:
        while True:
            self._drop_cancelled()
            if not self._waiting and not self._running:
                self._has_work.clear()
                await self._has_work.wait()
                continue
            plan = self._scheduler.schedule(self._waiting, self._running, time.perf_counter())
            self._metrics.preemptions.inc(len(plan.preempted))
            if not plan.sequences and not plan.prefill_chunks:
                await asyncio.sleep(plan.wait_seconds)
                continue
            await self._run_step(plan)
            self._update_gauges()

    async def _run_step(self, plan: StepPlan) -> None:
        step_started_at = time.perf_counter()
        try:
            new_tokens = await asyncio.to_thread(self._execute_plan, plan)
        except Exception as error:
            stepped = [*plan.sequences, *(sequence for sequence, _ in plan.prefill_chunks)]
            logger.exception("batched step failed", extra={"batch_size": len(stepped)})
            for sequence in stepped:
                self._fail(sequence, error)
            return
        if plan.prefill_chunks:
            step_kind = STEP_KIND_MIXED
        else:
            step_kind = STEP_KIND_PREFILL if plan.is_prefill else STEP_KIND_DECODE
        self._metrics.step_latency_seconds.labels(step_kind).observe(
            time.perf_counter() - step_started_at
        )
        for sequence, token_id in new_tokens:
            self._emit_token(sequence, token_id)

    def _execute_plan(self, plan: StepPlan) -> list[tuple[Sequence, int]]:
        """Runs in a worker thread. Whole sequences and prompt chunks are separate forward passes,
        so one-token decodes are not padded to the chunk length."""
        new_tokens: list[tuple[Sequence, int]] = []
        if plan.sequences:
            token_ids = self._runner.run_step(plan.sequences)
            new_tokens.extend(zip(plan.sequences, token_ids, strict=True))
        if plan.prefill_chunks:
            chunk_sequences = [sequence for sequence, _ in plan.prefill_chunks]
            chunk_sizes = [chunk_tokens for _, chunk_tokens in plan.prefill_chunks]
            token_ids = self._runner.run_step(chunk_sequences, chunk_sizes)
            new_tokens.extend(
                (sequence, token_id)
                for sequence, token_id in zip(chunk_sequences, token_ids, strict=True)
                if sequence.num_uncached_tokens == 0
            )
        return new_tokens

    def _emit_token(self, sequence: Sequence, token_id: int) -> None:
        sequence.token_ids.append(token_id)
        finish_reason = find_finish_reason(
            token_id, sequence.num_generated_tokens, sequence.sampling_params, self._stop_token_ids
        )
        handle = self._handles.get(sequence.request_id)
        if handle is not None:
            handle.token_events.put_nowait(
                TokenEvent(sequence.request_id, token_id, finish_reason is not None, finish_reason)
            )
            self._metrics.generated_tokens.inc()
        if finish_reason is not None:
            self._metrics.finished_requests.labels(finish_reason).inc()
            self._finish(sequence)

    def _fail(self, sequence: Sequence, error: Exception) -> None:
        handle = self._handles.get(sequence.request_id)
        if handle is not None:
            handle.token_events.put_nowait(error)
        self._finish(sequence)

    def _finish(self, sequence: Sequence) -> None:
        if sequence in self._running:
            self._running.remove(sequence)
        self._runner.allocator.free(sequence)
        self._handles.pop(sequence.request_id, None)

    def _drop_cancelled(self) -> None:
        """Free the seat and KV memory of requests whose client went away."""
        for sequence in [*self._waiting, *self._running]:
            handle = self._handles.get(sequence.request_id)
            if handle is None or not handle.is_cancelled:
                continue
            if sequence in self._waiting:
                self._waiting.remove(sequence)
            self._finish(sequence)

    def _update_gauges(self) -> None:
        self._metrics.waiting_requests.set(len(self._waiting))
        self._metrics.set_running_requests(len(self._running))
        self._metrics.kv_cache_reserved_slots.set(self._runner.allocator.num_reserved_slots)
        self._metrics.kv_cache_stored_tokens.set(
            sum(sequence.num_cached_tokens for sequence in self._running)
        )
        self._metrics.prefix_cache_hit_tokens.set(self._runner.allocator.num_prefix_hit_tokens)
