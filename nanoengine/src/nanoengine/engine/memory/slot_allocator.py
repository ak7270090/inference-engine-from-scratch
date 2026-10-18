"""What the scheduler and model runner need from a KV memory manager, contiguous or paged."""

from typing import Protocol

import torch

from nanoengine.engine.sequence_state import Sequence


class OutOfKVMemoryError(Exception):
    """No free region or block is left in the KV cache pool."""


class SlotAllocator(Protocol):
    @property
    def num_total_slots(self) -> int: ...

    @property
    def num_reserved_slots(self) -> int: ...

    def can_reserve(self, sequence: Sequence, num_tokens: int) -> bool:
        """Whether ``sequence`` can hold ``num_tokens`` tokens (new or growing sequence)."""
        ...

    def reserve(self, sequence: Sequence, num_tokens: int) -> None: ...

    def free(self, sequence: Sequence) -> None: ...

    def build_context_slots(self, sequences: list[Sequence], context_length: int) -> torch.Tensor:
        """``[len(sequences), context_length]``: the slot holding each token position."""
        ...
