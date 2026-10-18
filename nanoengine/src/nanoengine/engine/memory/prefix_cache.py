"""Automatic prefix caching: requests that start with the same tokens share KV blocks.

A full block is identified by its tokens plus its parent block's key, so a key names a whole
prefix, not just 16 tokens. When a block is filled it is registered under its key. A new request
walks its prompt block by block and takes every block already registered, then computes only the
rest. Freed blocks keep their content until the pool needs space (least recently used first).

Only full blocks are shared and the last prompt token is always recomputed (its logits are needed
to sample), so no step ever writes into a shared block: copy-on-write is not needed.
"""

import math
from collections import OrderedDict, deque

import torch

from nanoengine.engine.memory.paged_allocator import PagedSlotAllocator
from nanoengine.engine.memory.slot_allocator import OutOfKVMemoryError
from nanoengine.engine.sequence_state import Sequence

BlockKey = tuple[int, tuple[int, ...]]
ROOT_PARENT_HASH = 0


def build_block_keys(token_ids: list[int], block_size: int, num_blocks: int) -> list[BlockKey]:
    """Keys of the first ``num_blocks`` full blocks; each includes its parent's hash."""
    keys: list[BlockKey] = []
    parent_hash = ROOT_PARENT_HASH
    for block_index in range(num_blocks):
        block_tokens = tuple(token_ids[block_index * block_size : (block_index + 1) * block_size])
        key = (parent_hash, block_tokens)
        keys.append(key)
        parent_hash = hash(key)
    return keys


class CachingBlockPool:
    """Blocks with reference counts. A free block either holds nothing or holds cached content
    that can be reused until it is evicted for a new allocation."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._reference_counts = [0] * num_blocks
        self._empty_blocks: deque[int] = deque(range(num_blocks))
        self._evictable_blocks: OrderedDict[int, None] = OrderedDict()
        self._block_by_key: dict[BlockKey, int] = {}
        self._key_by_block: dict[int, BlockKey] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._empty_blocks) + len(self._evictable_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def is_free(self, block: int) -> bool:
        return self._reference_counts[block] == 0

    def allocate(self) -> int:
        """An empty block if any, otherwise evict the least recently used cached block."""
        if self._empty_blocks:
            block = self._empty_blocks.popleft()
        elif self._evictable_blocks:
            block, _ = self._evictable_blocks.popitem(last=False)
            del self._block_by_key[self._key_by_block.pop(block)]
        else:
            raise OutOfKVMemoryError("no free KV blocks")
        self._reference_counts[block] = 1
        return block

    def free(self, block: int) -> None:
        if self._reference_counts[block] <= 0:
            raise ValueError(f"block {block} is not allocated")
        self._reference_counts[block] -= 1
        if self._reference_counts[block] > 0:
            return
        if block in self._key_by_block:
            self._evictable_blocks[block] = None
        else:
            self._empty_blocks.append(block)

    def find(self, key: BlockKey) -> int | None:
        return self._block_by_key.get(key)

    def share(self, block: int) -> None:
        """Take one more reference to a cached block, reviving it if it was free."""
        if self._reference_counts[block] == 0:
            del self._evictable_blocks[block]
        self._reference_counts[block] += 1

    def register(self, block: int, key: BlockKey) -> None:
        """Remember that ``block`` holds the prefix ``key``; the first block with a key wins."""
        if key in self._block_by_key or block in self._key_by_block:
            return
        self._block_by_key[key] = block
        self._key_by_block[block] = key


class PrefixCachingSlotAllocator(PagedSlotAllocator):
    def __init__(self, num_blocks: int, block_size: int, device: torch.device) -> None:
        super().__init__(num_blocks, block_size, device)
        self._blocks = CachingBlockPool(num_blocks)
        self._registered_block_counts: dict[str, int] = {}
        self._prefix_hit_tokens = 0

    @property
    def num_prefix_hit_tokens(self) -> int:
        return self._prefix_hit_tokens

    def find_cached_blocks(self, sequence: Sequence) -> list[int]:
        """The longest run of cached full blocks at the start, leaving at least one token."""
        num_full_blocks = (sequence.num_tokens - 1) // self.block_size
        cached_blocks = []
        for key in build_block_keys(sequence.token_ids, self.block_size, num_full_blocks):
            block = self._blocks.find(key)
            if block is None:
                break
            cached_blocks.append(block)
        return cached_blocks

    def can_reserve(self, sequence: Sequence, num_tokens: int) -> bool:
        if sequence.request_id in self._block_tables:
            return super().can_reserve(sequence, num_tokens)
        cached_blocks = self.find_cached_blocks(sequence)
        revived_blocks = sum(1 for block in cached_blocks if self._blocks.is_free(block))
        missing_blocks = math.ceil(num_tokens / self.block_size) - len(cached_blocks)
        return missing_blocks <= self._blocks.num_free_blocks - revived_blocks

    def reserve(self, sequence: Sequence, num_tokens: int) -> None:
        """A new sequence first takes its cached prefix blocks and skips computing those tokens."""
        if sequence.request_id not in self._block_tables:
            if not self.can_reserve(sequence, num_tokens):
                raise OutOfKVMemoryError(f"no blocks for {sequence.request_id}")
            cached_blocks = self.find_cached_blocks(sequence)
            for block in cached_blocks:
                self._blocks.share(block)
            self._block_tables[sequence.request_id] = cached_blocks
            self._registered_block_counts[sequence.request_id] = len(cached_blocks)
            sequence.num_cached_tokens = len(cached_blocks) * self.block_size
            self._prefix_hit_tokens += sequence.num_cached_tokens
        super().reserve(sequence, num_tokens)

    def on_tokens_cached(self, sequence: Sequence) -> None:
        """Register blocks that just became full so later requests can reuse them."""
        table = self._block_tables.get(sequence.request_id, [])
        registered = self._registered_block_counts.get(sequence.request_id, 0)
        num_full_blocks = min(sequence.num_cached_tokens // self.block_size, len(table))
        if num_full_blocks <= registered:
            return
        keys = build_block_keys(sequence.token_ids, self.block_size, num_full_blocks)
        for block_index in range(registered, num_full_blocks):
            self._blocks.register(table[block_index], keys[block_index])
        self._registered_block_counts[sequence.request_id] = num_full_blocks

    def free(self, sequence: Sequence) -> None:
        super().free(sequence)
        self._registered_block_counts.pop(sequence.request_id, None)
