"""The KV cache tensors: one flat pool of token slots per layer, sized from a memory budget."""

import torch

from nanoengine.models.qwen3 import Qwen3Config

BYTES_PER_MEGABYTE = 1024 * 1024


def count_kv_bytes_per_token(config: Qwen3Config, dtype: torch.dtype) -> int:
    """2 (key and value) × layers × KV heads × head dim × bytes per element."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * element_size
    )


def count_slots_for_budget(config: Qwen3Config, dtype: torch.dtype, memory_megabytes: int) -> int:
    return memory_megabytes * BYTES_PER_MEGABYTE // count_kv_bytes_per_token(config, dtype)


class KVCachePool:
    """``key_caches[layer]`` and ``value_caches[layer]`` are ``[num_slots, kv_heads, head_dim]``."""

    def __init__(
        self, config: Qwen3Config, num_slots: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        shape = (num_slots, config.num_key_value_heads, config.head_dim)
        self.num_slots = num_slots
        self.key_caches = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)
        ]
        self.value_caches = [
            torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)
        ]
