from collections import deque

import torch

from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.memory.contiguous_allocator import ContiguousSlotAllocator
from nanoengine.engine.memory.paged_allocator import PagedSlotAllocator
from nanoengine.engine.scheduling.batch_scheduler import (
    ContinuousBatchScheduler,
    StaticBatchScheduler,
)
from nanoengine.engine.sequence_state import Sequence

CPU = torch.device("cpu")


def build_sequence(request_id: str, num_tokens: int, arrival_time: float = 0.0) -> Sequence:
    request = GenerationRequest(request_id, [1] * num_tokens, SamplingParams(max_new_tokens=8))
    sequence = Sequence.from_request(request, CPU)
    sequence.arrival_time = arrival_time
    return sequence


def mark_prefilled(sequences: list[Sequence]) -> None:
    for sequence in sequences:
        sequence.num_cached_tokens = sequence.num_tokens
        sequence.token_ids.append(7)


def test_continuous_admits_until_seats_run_out_then_decodes_everyone() -> None:
    allocator = ContiguousSlotAllocator(num_regions=8, region_length=32, device=CPU)
    scheduler = ContinuousBatchScheduler(allocator, max_num_seqs=2, max_prefill_tokens=100)
    waiting = deque(build_sequence(f"r{index}", 4) for index in range(3))
    running: list[Sequence] = []

    prefill = scheduler.schedule(waiting, running, now=0.0)
    mark_prefilled(prefill.sequences)
    decode = scheduler.schedule(waiting, running, now=0.0)

    assert prefill.is_prefill and [s.request_id for s in prefill.sequences] == ["r0", "r1"]
    assert not decode.is_prefill and decode.sequences == running
    assert [s.request_id for s in waiting] == ["r2"]


def test_continuous_splits_prefills_by_the_token_budget() -> None:
    allocator = ContiguousSlotAllocator(num_regions=8, region_length=32, device=CPU)
    scheduler = ContinuousBatchScheduler(allocator, max_num_seqs=8, max_prefill_tokens=10)
    waiting = deque(build_sequence(f"r{index}", 6) for index in range(3))

    plan = scheduler.schedule(waiting, [], now=0.0)

    assert len(plan.sequences) == 1


def test_static_waits_to_fill_the_batch_then_runs_it_to_the_end() -> None:
    allocator = ContiguousSlotAllocator(num_regions=8, region_length=32, device=CPU)
    scheduler = StaticBatchScheduler(
        allocator, max_num_seqs=4, max_prefill_tokens=100, batch_fill_timeout_seconds=1.0
    )
    waiting = deque([build_sequence("early", 4, arrival_time=10.0)])
    running: list[Sequence] = []

    too_soon = scheduler.schedule(waiting, running, now=10.4)
    started = scheduler.schedule(waiting, running, now=11.5)
    mark_prefilled(started.sequences)
    waiting.append(build_sequence("late", 4, arrival_time=11.6))
    while_running = scheduler.schedule(waiting, running, now=20.0)

    assert too_soon.sequences == [] and 0.5 < too_soon.wait_seconds <= 0.6
    assert [s.request_id for s in started.sequences] == ["early"]
    assert [s.request_id for s in while_running.sequences] == ["early"]
    assert [s.request_id for s in waiting] == ["late"]
    assert scheduler.schedule(deque(), [], now=0.0).sequences == []


def test_paged_decode_preempts_the_newest_sequence_when_blocks_run_out() -> None:
    allocator = PagedSlotAllocator(num_blocks=2, block_size=4, device=CPU)
    scheduler = ContinuousBatchScheduler(allocator, max_num_seqs=4, max_prefill_tokens=100)
    old, new = build_sequence("old", 3), build_sequence("new", 3)
    waiting: deque[Sequence] = deque([old, new])
    running: list[Sequence] = []
    mark_prefilled(scheduler.schedule(waiting, running, now=0.0).sequences)  # prefill both
    mark_prefilled(scheduler.schedule(waiting, running, now=0.0).sequences)  # first decode

    plan = scheduler.schedule(waiting, running, now=0.0)

    assert [s.request_id for s in plan.preempted] == ["new"]
    assert running == [old] and list(waiting) == [new]
    assert new.num_cached_tokens == 0
    assert allocator.block_table(old) == [0, 1]
