"""Free list of fixed-size KV blocks: hand one out, take it back, never give one out twice."""

from collections import deque

from nanoengine.engine.memory.slot_allocator import OutOfKVMemoryError


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free_blocks: deque[int] = deque(range(num_blocks))
        self._used_blocks: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return len(self._used_blocks)

    def allocate(self) -> int:
        if not self._free_blocks:
            raise OutOfKVMemoryError("no free KV blocks")
        block = self._free_blocks.popleft()
        self._used_blocks.add(block)
        return block

    def free(self, block: int) -> None:
        if block not in self._used_blocks:
            raise ValueError(f"block {block} is not allocated")
        self._used_blocks.remove(block)
        self._free_blocks.append(block)
