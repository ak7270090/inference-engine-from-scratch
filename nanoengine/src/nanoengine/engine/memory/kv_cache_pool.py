"""The KV cache tensors: one flat pool of token slots per layer, sized from a memory budget."""

import torch

from nanoengine.models.qwen3 import Qwen3Config

BYTES_PER_MEGABYTE = 1024 * 1024
INT8_BYTES = 1


def count_kv_bytes_per_token(
    config: Qwen3Config, dtype: torch.dtype, is_int8_cache: bool = False
) -> int:
    """2 (key and value) × layers × KV heads × bytes per head vector.

    A head vector is ``head_dim`` numbers in ``dtype``, or ``head_dim`` int8 numbers plus one
    scale in ``dtype`` for an int8 cache.
    """
    element_size = torch.empty((), dtype=dtype).element_size()
    if is_int8_cache:
        bytes_per_head = config.head_dim * INT8_BYTES + element_size
    else:
        bytes_per_head = config.head_dim * element_size
    return 2 * config.num_hidden_layers * config.num_key_value_heads * bytes_per_head


def count_slots_for_budget(
    config: Qwen3Config, dtype: torch.dtype, memory_megabytes: int, is_int8_cache: bool = False
) -> int:
    bytes_per_token = count_kv_bytes_per_token(config, dtype, is_int8_cache)
    return memory_megabytes * BYTES_PER_MEGABYTE // bytes_per_token


class KVCachePool:
    """Per layer key and value caches, each ``[num_slots + 1, kv_heads, head_dim]``.

    The extra last row is a scratch slot: padded tokens of a batch write there, so writes never
    need a mask. An int8 cache adds per layer scales ``[num_slots + 1, kv_heads, 1]``.
    """

    def __init__(
        self,
        config: Qwen3Config,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
        is_int8_cache: bool = False,
    ) -> None:
        shape = (num_slots + 1, config.num_key_value_heads, config.head_dim)
        scale_shape = (num_slots + 1, config.num_key_value_heads, 1)
        cache_dtype = torch.int8 if is_int8_cache else dtype
        num_layers = config.num_hidden_layers
        self.num_slots = num_slots
        self.scratch_slot = num_slots
        self.key_caches = [
            torch.zeros(shape, device=device, dtype=cache_dtype) for _ in range(num_layers)
        ]
        self.value_caches = [
            torch.zeros(shape, device=device, dtype=cache_dtype) for _ in range(num_layers)
        ]
        self.key_scales: list[torch.Tensor] | None = None
        self.value_scales: list[torch.Tensor] | None = None
        if is_int8_cache:
            self.key_scales = [
                torch.zeros(scale_shape, device=device, dtype=dtype) for _ in range(num_layers)
            ]
            self.value_scales = [
                torch.zeros(scale_shape, device=device, dtype=dtype) for _ in range(num_layers)
            ]
