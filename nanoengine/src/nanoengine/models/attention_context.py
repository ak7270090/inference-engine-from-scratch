"""What cached attention needs for one step: where to write new keys and values, what to read.

The KV cache is one flat pool of token slots per layer, ``[num_slots, kv_heads, head_dim]``. A
contiguous cache gives each sequence a reserved range of slots; a paged cache gives it blocks of
slots anywhere in the pool. Attention does not care which: it only sees slot numbers.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class AttentionContext:
    """Padded batch of sequences; ``b`` sequences, ``q`` new tokens, ``c`` context tokens max."""

    positions: torch.Tensor  # [b, q] position of each new token (RoPE), 0 for padding
    slot_mapping: torch.Tensor  # [b, q] slot that receives each new token's key / value, -1 = pad
    context_slots: torch.Tensor  # [b, c] slots of every token a query may attend to
    attention_mask: torch.Tensor  # [b, 1, q, c] True where a query may attend to a key
    key_caches: list[torch.Tensor]  # per layer [num_slots, kv_heads, head_dim]
    value_caches: list[torch.Tensor]


def build_attention_mask(
    positions: torch.Tensor, context_lengths: torch.Tensor, context_length: int
) -> torch.Tensor:
    """Causal and length mask: a query at position p sees keys 0..p of its own sequence."""
    key_positions = torch.arange(context_length, device=positions.device)
    is_causal = key_positions[None, None, :] <= positions[:, :, None]
    is_inside = key_positions[None, None, :] < context_lengths[:, None, None]
    return (is_causal & is_inside).unsqueeze(1)


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
    key_cache = context.key_caches[layer_index]
    value_cache = context.value_caches[layer_index]
    is_real_token = context.slot_mapping >= 0
    target_slots = context.slot_mapping[is_real_token]
    key_cache[target_slots] = keys.transpose(1, 2)[is_real_token]
    value_cache[target_slots] = values.transpose(1, 2)[is_real_token]

    heads_per_kv_head = queries.shape[1] // keys.shape[1]
    context_keys = key_cache[context.context_slots].transpose(1, 2)
    context_values = value_cache[context.context_slots].transpose(1, 2)
    context_keys = context_keys.repeat_interleave(heads_per_kv_head, dim=1)
    context_values = context_values.repeat_interleave(heads_per_kv_head, dim=1)
    return functional.scaled_dot_product_attention(
        queries, context_keys, context_values, attn_mask=context.attention_mask
    )
