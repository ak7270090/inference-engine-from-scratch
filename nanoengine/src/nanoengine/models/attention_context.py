"""What cached attention needs for one step: where to write new keys and values, what to read.

The KV cache is one flat pool of token slots per layer, ``[num_slots, kv_heads, head_dim]``. A
contiguous cache gives each sequence a reserved range of slots; a paged cache gives it blocks of
slots anywhere in the pool. Attention does not care which: it only sees slot numbers.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional

from nanoengine.models.int8_quantization import quantize_vectors


@dataclass(frozen=True)
class AttentionContext:
    """Padded batch of sequences; ``b`` sequences, ``q`` new tokens, ``c`` context tokens max."""

    positions: torch.Tensor  # [b, q] position of each new token (RoPE), 0 for padding
    slot_mapping: torch.Tensor  # [b, q] slot for each new token's key / value; pads: scratch slot
    context_slots: torch.Tensor  # [b, c] slots of every token a query may attend to
    attention_mask: torch.Tensor  # [b, 1, q, c] True where a query may attend to a key
    key_caches: list[torch.Tensor]  # per layer [num_slots, kv_heads, head_dim]
    value_caches: list[torch.Tensor]
    key_scales: list[torch.Tensor] | None = (
        None  # int8 cache only: per layer [num_slots, kv_heads, 1]
    )
    value_scales: list[torch.Tensor] | None = None


def build_attention_mask(
    positions: torch.Tensor, context_lengths: torch.Tensor, context_length: int
) -> torch.Tensor:
    """Causal and length mask: a query at position p sees keys 0..p of its own sequence."""
    key_positions = torch.arange(context_length, device=positions.device)
    is_causal = key_positions[None, None, :] <= positions[:, :, None]
    is_inside = key_positions[None, None, :] < context_lengths[:, None, None]
    return (is_causal & is_inside).unsqueeze(1)


def write_to_cache(
    cache: torch.Tensor,
    scales: torch.Tensor | None,
    target_slots: torch.Tensor,
    vectors: torch.Tensor,
) -> None:
    """``vectors`` is ``[n, kv_heads, d]``; an int8 cache also stores one scale per vector."""
    if scales is None:
        cache.index_copy_(0, target_slots, vectors)
        return
    quantized, vector_scales = quantize_vectors(vectors)
    cache.index_copy_(0, target_slots, quantized)
    scales.index_copy_(0, target_slots, vector_scales)


def read_from_cache(
    cache: torch.Tensor, scales: torch.Tensor | None, slots: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Gather ``slots`` (``[b, c]``) into ``[b, c, kv_heads, d]``, dequantizing an int8 cache."""
    if scales is None:
        return cache[slots]
    return cache[slots].to(dtype) * scales[slots]


def run_cached_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    context: AttentionContext,
    layer_index: int,
) -> torch.Tensor:
    """Write this step's keys / values into the pool, then attend over each sequence's slots.

    ``queries`` is ``[b, heads, q, d]``; ``keys`` and ``values`` are ``[b, kv_heads, q, d]``.
    """
    key_scales = context.key_scales[layer_index] if context.key_scales else None
    value_scales = context.value_scales[layer_index] if context.value_scales else None
    # padding goes to a scratch slot; masking it out instead would make the GPU stop and report
    # how many tokens are real, once per layer, stalling the queue of launched work
    target_slots = context.slot_mapping.reshape(-1)
    vector_shape = (-1, keys.shape[1], keys.shape[3])
    write_to_cache(
        context.key_caches[layer_index],
        key_scales,
        target_slots,
        keys.transpose(1, 2).reshape(vector_shape),
    )
    write_to_cache(
        context.value_caches[layer_index],
        value_scales,
        target_slots,
        values.transpose(1, 2).reshape(vector_shape),
    )

    heads_per_kv_head = queries.shape[1] // keys.shape[1]
    context_keys = read_from_cache(
        context.key_caches[layer_index], key_scales, context.context_slots, queries.dtype
    )
    context_values = read_from_cache(
        context.value_caches[layer_index], value_scales, context.context_slots, queries.dtype
    )
    context_keys = context_keys.transpose(1, 2).repeat_interleave(heads_per_kv_head, dim=1)
    context_values = context_values.transpose(1, 2).repeat_interleave(heads_per_kv_head, dim=1)
    return functional.scaled_dot_product_attention(
        queries, context_keys, context_values, attn_mask=context.attention_mask
    )
