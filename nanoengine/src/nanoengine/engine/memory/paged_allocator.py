"""Paged KV cache (PagedAttention's memory manager): blocks are handed out only as a sequence grows.

Each sequence has a block table: logical block ``i`` (tokens ``i*block_size`` …) lives in physical
block ``table[i]``, anywhere in the pool. Waste is at most one partly filled block per sequence.
"""

import math

import torch

from nanoengine.engine.memory.block_allocator import BlockAllocator
from nanoengine.engine.memory.slot_allocator import OutOfKVMemoryError
from nanoengine.engine.sequence_state import Sequence


def pad_block_table(table: list[int], length: int) -> list[int]:
    """Cut or pad with block 0; padded positions are masked out of attention."""
    return table[:length] + [0] * (length - len(table[:length]))


class PagedSlotAllocator:
    def __init__(self, num_blocks: int, block_size: int, device: torch.device) -> None:
        self.block_size = block_size
        self._blocks = BlockAllocator(num_blocks)
        self._block_tables: dict[str, list[int]] = {}
        self._device = device

    @property
    def num_total_slots(self) -> int:
        return self._blocks.num_blocks * self.block_size

    @property
    def num_reserved_slots(self) -> int:
        return self._blocks.num_used_blocks * self.block_size

    @property
    def num_free_blocks(self) -> int:
        return self._blocks.num_free_blocks

    def block_table(self, sequence: Sequence) -> list[int]:
        return list(self._block_tables.get(sequence.request_id, []))

    def count_missing_blocks(self, sequence: Sequence, num_tokens: int) -> int:
        owned = len(self._block_tables.get(sequence.request_id, []))
        return max(0, math.ceil(num_tokens / self.block_size) - owned)

    def can_reserve(self, sequence: Sequence, num_tokens: int) -> bool:
        return self.count_missing_blocks(sequence, num_tokens) <= self._blocks.num_free_blocks

    def reserve(self, sequence: Sequence, num_tokens: int) -> None:
        if not self.can_reserve(sequence, num_tokens):
            raise OutOfKVMemoryError(f"no blocks for {num_tokens} tokens of {sequence.request_id}")
        table = self._block_tables.setdefault(sequence.request_id, [])
        for _ in range(self.count_missing_blocks(sequence, num_tokens)):
            table.append(self._blocks.allocate())

    def free(self, sequence: Sequence) -> None:
        for block in self._block_tables.pop(sequence.request_id, []):
            self._blocks.free(block)

    @property
    def num_prefix_hit_tokens(self) -> int:
        """Prompt tokens served from the prefix cache instead of computed (0 without one)."""
        return 0

    def on_tokens_cached(self, sequence: Sequence) -> None:
        """Called after a step wrote more of ``sequence`` into the cache."""

    def build_context_slots(self, sequences: list[Sequence], context_length: int) -> torch.Tensor:
        """One slot per token position ``p``: ``table[p // size] * size + p % size``."""
        blocks_per_row = math.ceil(context_length / self.block_size)
        padded_tables = [
            pad_block_table(self._block_tables[sequence.request_id], blocks_per_row)
            for sequence in sequences
        ]
        block_tables = torch.tensor(padded_tables, device=self._device)
        positions = torch.arange(context_length, device=self._device)
        blocks = block_tables[:, positions // self.block_size]
        return blocks * self.block_size + positions % self.block_size
