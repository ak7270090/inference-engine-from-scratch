"""Pack the next tokens of several sequences into one padded step for the model."""

from dataclasses import dataclass

import torch

from nanoengine.engine.memory.kv_cache_pool import KVCachePool
from nanoengine.engine.memory.slot_allocator import SlotAllocator
from nanoengine.engine.sequence_state import Sequence
from nanoengine.models.attention_context import AttentionContext, build_attention_mask

PADDING_TOKEN_ID = 0


@dataclass(frozen=True)
class StepInputs:
    token_ids: torch.Tensor  # [b, q]
    context: AttentionContext
    last_token_indices: torch.Tensor  # [b] index of each sequence's last real token in q


def build_step_inputs(
    sequences: list[Sequence],
    allocator: SlotAllocator,
    pool: KVCachePool,
    device: torch.device,
    token_counts: list[int] | None = None,
) -> StepInputs:
    """Each sequence brings its next ``token_counts[i]`` uncached tokens (default: all of them).

    A whole prompt is a prefill, a slice of one is a prefill chunk, one token is a decode step.
    """
    new_token_counts = token_counts or [sequence.num_uncached_tokens for sequence in sequences]
    end_positions = [
        sequence.num_cached_tokens + count
        for sequence, count in zip(sequences, new_token_counts, strict=True)
    ]
    query_length = max(new_token_counts)
    context_length = max(end_positions)
    token_rows, position_rows = [], []
    for sequence, end in zip(sequences, end_positions, strict=True):
        start = sequence.num_cached_tokens
        padding = [PADDING_TOKEN_ID] * (query_length - (end - start))
        token_rows.append(sequence.token_ids[start:end] + padding)
        position_rows.append(list(range(start, end)) + padding)

    token_ids = torch.tensor(token_rows, device=device)
    positions = torch.tensor(position_rows, device=device)
    counts = torch.tensor(new_token_counts, device=device)
    is_real_token = torch.arange(query_length, device=device)[None, :] < counts[:, None]
    context_slots = allocator.build_context_slots(sequences, context_length)
    slot_mapping = context_slots.gather(1, positions)
    slot_mapping = slot_mapping.masked_fill(~is_real_token, pool.scratch_slot)
    context_lengths = torch.tensor(end_positions, device=device)
    context = AttentionContext(
        positions=positions,
        slot_mapping=slot_mapping,
        context_slots=context_slots,
        attention_mask=build_attention_mask(positions, context_lengths, context_length),
        key_caches=pool.key_caches,
        value_caches=pool.value_caches,
        key_scales=pool.key_scales,
        value_scales=pool.value_scales,
    )
    return StepInputs(token_ids=token_ids, context=context, last_token_indices=counts - 1)
