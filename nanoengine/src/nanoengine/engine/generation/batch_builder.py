"""Pack the uncached tokens of several sequences into one padded step for the model."""

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
) -> StepInputs:
    """Prefill sequences bring all their uncached tokens, decoding sequences bring one."""
    new_token_counts = [sequence.num_uncached_tokens for sequence in sequences]
    query_length = max(new_token_counts)
    context_length = max(sequence.num_tokens for sequence in sequences)
    token_rows, position_rows = [], []
    for sequence, new_token_count in zip(sequences, new_token_counts, strict=True):
        padding = [PADDING_TOKEN_ID] * (query_length - new_token_count)
        token_rows.append(sequence.token_ids[sequence.num_cached_tokens :] + padding)
        position_rows.append(list(range(sequence.num_cached_tokens, sequence.num_tokens)) + padding)

    token_ids = torch.tensor(token_rows, device=device)
    positions = torch.tensor(position_rows, device=device)
    counts = torch.tensor(new_token_counts, device=device)
    is_real_token = torch.arange(query_length, device=device)[None, :] < counts[:, None]
    context_slots = allocator.build_context_slots(sequences, context_length)
    slot_mapping = context_slots.gather(1, positions)
    slot_mapping = slot_mapping.masked_fill(~is_real_token, pool.scratch_slot)
    context_lengths = torch.tensor([sequence.num_tokens for sequence in sequences], device=device)
    context = AttentionContext(
        positions=positions,
        slot_mapping=slot_mapping,
        context_slots=context_slots,
        attention_mask=build_attention_mask(positions, context_lengths, context_length),
        key_caches=pool.key_caches,
        value_caches=pool.value_caches,
    )
    return StepInputs(token_ids=token_ids, context=context, last_token_indices=counts - 1)
