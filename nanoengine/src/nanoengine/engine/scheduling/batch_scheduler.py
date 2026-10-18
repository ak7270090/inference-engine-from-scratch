"""Decide which sequences run in the next step.

Continuous (iteration-level) batching re-decides every step: finished sequences leave, waiting
ones join as soon as there is a seat and KV memory. Static batching admits a batch, then runs it
until every sequence in it is done, leaving finished seats empty. Chunked prefill splits prompts
into pieces that share each step's token budget with the decoding sequences, so a long prompt
cannot freeze everyone else's stream.
"""

from collections import deque
from dataclasses import dataclass, field

from nanoengine.engine.memory.slot_allocator import SlotAllocator
from nanoengine.engine.sequence_state import Sequence

UNLIMITED_PREFILL_TOKENS = 10**9


@dataclass(frozen=True)
class StepPlan:
    """``sequences`` cache all their uncached tokens; ``prefill_chunks`` cache only a slice."""

    sequences: list[Sequence]
    is_prefill: bool = False
    wait_seconds: float = 0.0
    preempted: list[Sequence] = field(default_factory=list)
    prefill_chunks: list[tuple[Sequence, int]] = field(default_factory=list)


class ContinuousBatchScheduler:
    def __init__(
        self, allocator: SlotAllocator, max_num_seqs: int, max_prefill_tokens: int
    ) -> None:
        self._allocator = allocator
        self._max_num_seqs = max_num_seqs
        self._max_prefill_tokens = max_prefill_tokens

    def schedule(self, waiting: deque[Sequence], running: list[Sequence], now: float) -> StepPlan:
        """Prefill new arrivals first (they wait for their first token), otherwise decode all."""
        admitted = self.admit_waiting(waiting, running)
        if admitted:
            return StepPlan(admitted, is_prefill=True)
        return self.plan_decode(waiting, running)

    def admit_waiting(self, waiting: deque[Sequence], running: list[Sequence]) -> list[Sequence]:
        """Move waiting sequences to running while seats, KV memory and the prefill budget last."""
        admitted: list[Sequence] = []
        prefill_tokens = 0
        while waiting and len(running) < self._max_num_seqs:
            candidate = waiting[0]
            if admitted and prefill_tokens + candidate.num_tokens > self._max_prefill_tokens:
                break
            if not self._allocator.can_reserve(candidate, candidate.num_tokens):
                break
            self._allocator.reserve(candidate, candidate.num_tokens)
            waiting.popleft()
            running.append(candidate)
            admitted.append(candidate)
            prefill_tokens += candidate.num_tokens
        return admitted

    def plan_decode(self, waiting: deque[Sequence], running: list[Sequence]) -> StepPlan:
        """Give every running sequence room for one more token, preempting the newest if needed."""
        preempted = self.reserve_next_tokens(waiting, running, list(running))
        return StepPlan(list(running), preempted=preempted)

    def reserve_next_tokens(
        self, waiting: deque[Sequence], running: list[Sequence], growing: list[Sequence]
    ) -> list[Sequence]:
        """Reserve one more token for each of ``growing``; returns the preempted sequences."""
        preempted: list[Sequence] = []
        for sequence in growing:
            if sequence not in running:
                continue
            while not self._allocator.can_reserve(sequence, sequence.num_tokens):
                victim = running.pop()
                self._allocator.free(victim)
                victim.forget_cache()
                waiting.appendleft(victim)
                preempted.append(victim)
                if victim is sequence:
                    break
            if sequence in running:
                self._allocator.reserve(sequence, sequence.num_tokens)
        return preempted


class ChunkedPrefillScheduler(ContinuousBatchScheduler):
    """Every step: one token for each decoding sequence plus prompt chunks, together at most
    ``max_num_batched_tokens``. Prompts reserve their KV memory when admitted."""

    def __init__(
        self, allocator: SlotAllocator, max_num_seqs: int, max_num_batched_tokens: int
    ) -> None:
        super().__init__(allocator, max_num_seqs, UNLIMITED_PREFILL_TOKENS)
        self._max_num_batched_tokens = max_num_batched_tokens

    def schedule(self, waiting: deque[Sequence], running: list[Sequence], now: float) -> StepPlan:
        self.admit_waiting(waiting, running)
        decoding = [sequence for sequence in running if sequence.num_uncached_tokens == 1]
        preempted = self.reserve_next_tokens(waiting, running, decoding)
        decoding = [sequence for sequence in decoding if sequence in running]
        token_budget = max(self._max_num_batched_tokens - len(decoding), 0)
        prefill_chunks = []
        for sequence in running:
            if sequence.num_uncached_tokens > 1 and token_budget > 0:
                chunk_tokens = min(sequence.num_uncached_tokens, token_budget)
                prefill_chunks.append((sequence, chunk_tokens))
                token_budget -= chunk_tokens
        return StepPlan(decoding, preempted=preempted, prefill_chunks=prefill_chunks)


class StaticBatchScheduler(ContinuousBatchScheduler):
    def __init__(
        self,
        allocator: SlotAllocator,
        max_num_seqs: int,
        max_prefill_tokens: int,
        batch_fill_timeout_seconds: float,
    ) -> None:
        super().__init__(allocator, max_num_seqs, max_prefill_tokens)
        self._batch_fill_timeout_seconds = batch_fill_timeout_seconds

    def schedule(self, waiting: deque[Sequence], running: list[Sequence], now: float) -> StepPlan:
        """Run the current batch to the end; start a new one when full or the fill wait is over."""
        if running:
            return self.plan_decode(waiting, running)
        if not waiting:
            return StepPlan([])
        waited_seconds = now - waiting[0].arrival_time
        if len(waiting) < self._max_num_seqs and waited_seconds < self._batch_fill_timeout_seconds:
            return StepPlan([], wait_seconds=self._batch_fill_timeout_seconds - waited_seconds)
        return StepPlan(self.admit_waiting(waiting, running), is_prefill=True)
