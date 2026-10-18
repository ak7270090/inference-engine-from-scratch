"""Contiguous KV cache: every sequence reserves a whole ``max_model_len`` region up front.

Simple and fast to index, but a 20-token answer still holds memory for ``max_model_len`` tokens,
which caps how many sequences fit. The paged allocator removes that waste.
"""

import torch

from nanoengine.engine.memory.slot_allocator import OutOfKVMemoryError
from nanoengine.engine.sequence_state import Sequence


class ContiguousSlotAllocator:
    def __init__(self, num_regions: int, region_length: int, device: torch.device) -> None:
        self.num_regions = num_regions
        self.region_length = region_length
        self._device = device
        self._free_regions = list(range(num_regions - 1, -1, -1))
        self._region_by_request: dict[str, int] = {}

    @property
    def num_total_slots(self) -> int:
        return self.num_regions * self.region_length

    @property
    def num_reserved_slots(self) -> int:
        return len(self._region_by_request) * self.region_length

    def can_reserve(self, sequence: Sequence, num_tokens: int) -> bool:
        if num_tokens > self.region_length:
            return False
        return sequence.request_id in self._region_by_request or bool(self._free_regions)

    def reserve(self, sequence: Sequence, num_tokens: int) -> None:
        if not self.can_reserve(sequence, num_tokens):
            raise OutOfKVMemoryError(f"no region for {num_tokens} tokens of {sequence.request_id}")
        if sequence.request_id not in self._region_by_request:
            self._region_by_request[sequence.request_id] = self._free_regions.pop()

    def free(self, sequence: Sequence) -> None:
        region = self._region_by_request.pop(sequence.request_id, None)
        if region is not None:
            self._free_regions.append(region)

    def build_context_slots(self, sequences: list[Sequence], context_length: int) -> torch.Tensor:
        region_starts = torch.tensor(
            [
                self._region_by_request[sequence.request_id] * self.region_length
                for sequence in sequences
            ],
            device=self._device,
        )
        offsets = torch.arange(context_length, device=self._device)
        return region_starts[:, None] + offsets[None, :]
