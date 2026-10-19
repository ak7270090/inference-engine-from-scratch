import pytest
import torch
from hypothesis import given, settings, strategies

from nanoengine.config import EngineSettings
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.memory.block_allocator import BlockAllocator
from nanoengine.engine.memory.contiguous_allocator import ContiguousSlotAllocator
from nanoengine.engine.memory.paged_allocator import PagedSlotAllocator
from nanoengine.engine.memory.slot_allocator import OutOfKVMemoryError
from nanoengine.engine.runtime import create_slot_allocator
from nanoengine.engine.sequence_state import Sequence

CPU = torch.device("cpu")


def build_sequence(request_id: str, num_tokens: int = 1) -> Sequence:
    request = GenerationRequest(request_id, [1] * num_tokens, SamplingParams(max_new_tokens=1))
    return Sequence.from_request(request, CPU)


def test_block_allocator_refuses_when_empty_and_on_double_free() -> None:
    allocator = BlockAllocator(2)
    blocks = [allocator.allocate(), allocator.allocate()]

    with pytest.raises(OutOfKVMemoryError):
        allocator.allocate()
    allocator.free(blocks[0])
    with pytest.raises(ValueError, match="not allocated"):
        allocator.free(blocks[0])
    assert (allocator.num_free_blocks, allocator.num_used_blocks) == (1, 1)


def test_contiguous_allocator_reserves_a_whole_region_per_sequence() -> None:
    allocator = ContiguousSlotAllocator(num_regions=2, region_length=8, device=CPU)
    first, second, third = (build_sequence(name) for name in "abc")

    allocator.reserve(first, 2)
    allocator.reserve(second, 8)

    assert allocator.num_reserved_slots == 16
    assert not allocator.can_reserve(third, 1)
    assert not allocator.can_reserve(first, 9)
    with pytest.raises(OutOfKVMemoryError):
        allocator.reserve(third, 1)
    allocator.free(first)
    assert allocator.can_reserve(third, 1)
    slots = allocator.build_context_slots([second], 3).tolist()[0]
    assert slots == [slots[0], slots[0] + 1, slots[0] + 2] and slots[0] % 8 == 0


def test_paged_allocator_grows_one_block_at_a_time() -> None:
    allocator = PagedSlotAllocator(num_blocks=4, block_size=4, device=CPU)
    sequence = build_sequence("a")

    allocator.reserve(sequence, 4)
    allocator.reserve(sequence, 5)

    assert allocator.block_table(sequence) == [0, 1]
    assert allocator.num_reserved_slots == 8
    assert allocator.build_context_slots([sequence], 6).tolist() == [[0, 1, 2, 3, 4, 5]]
    with pytest.raises(OutOfKVMemoryError):
        allocator.reserve(sequence, 17)
    allocator.free(sequence)
    assert allocator.num_free_blocks == 4


@settings(max_examples=60, deadline=None)
@given(
    strategies.lists(
        strategies.tuples(
            strategies.integers(0, 4), strategies.integers(1, 20), strategies.booleans()
        ),
        max_size=40,
    )
)
def test_paged_allocator_never_loses_or_shares_a_block(operations) -> None:
    allocator = PagedSlotAllocator(num_blocks=10, block_size=4, device=CPU)
    sequences = [build_sequence(f"s{index}") for index in range(5)]
    for sequence_index, num_tokens, should_free in operations:
        sequence = sequences[sequence_index]
        if should_free:
            allocator.free(sequence)
        elif allocator.can_reserve(sequence, num_tokens):
            allocator.reserve(sequence, num_tokens)
        tables = [allocator.block_table(sequence) for sequence in sequences]
        owned = [block for table in tables for block in table]
        assert len(owned) == len(set(owned))
        assert len(owned) + allocator.num_free_blocks == 10


def test_runtime_sizes_the_pool_from_the_memory_budget(tiny_model) -> None:
    contiguous, contiguous_slots = create_slot_allocator(
        tiny_model,
        EngineSettings(
            kv_cache="contiguous", scheduler="continuous", max_model_len=1024, max_num_seqs=64
        ),
    )
    paged, paged_slots = create_slot_allocator(
        tiny_model, EngineSettings(kv_cache="paged", scheduler="continuous", max_model_len=1024)
    )
    serial, serial_slots = create_slot_allocator(
        tiny_model, EngineSettings(kv_cache="contiguous", max_model_len=1024)
    )

    assert isinstance(contiguous, ContiguousSlotAllocator) and contiguous.num_regions == 64
    assert isinstance(paged, PagedSlotAllocator) and paged_slots % 16 == 0
    assert serial_slots == 1024
    assert contiguous_slots == 64 * 1024


@pytest.mark.parametrize("kv_cache", ["contiguous", "paged"])
def test_runtime_rejects_a_budget_smaller_than_one_sequence(tiny_model, kv_cache: str) -> None:
    too_small = EngineSettings(
        kv_cache=kv_cache, scheduler="continuous", max_model_len=10**7, kv_cache_memory_mb=1
    )

    with pytest.raises(ValueError, match="raise the budget"):
        create_slot_allocator(tiny_model, too_small)
