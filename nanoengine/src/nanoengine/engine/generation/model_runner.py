"""Run one model step over a batch of sequences and sample one new token for each."""

import torch

from nanoengine.engine.generation.batch_builder import build_step_inputs
from nanoengine.engine.memory.kv_cache_pool import KVCachePool
from nanoengine.engine.memory.slot_allocator import SlotAllocator
from nanoengine.engine.sampling import GREEDY_TEMPERATURE, sample_next_token
from nanoengine.engine.sequence_state import Sequence
from nanoengine.models.qwen3 import Qwen3ForCausalLM


class ModelRunner:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        pool: KVCachePool,
        allocator: SlotAllocator,
        device: torch.device,
    ) -> None:
        self._model = model
        self._pool = pool
        self.allocator = allocator
        self._device = device

    @property
    def vocab_size(self) -> int:
        return self._model.config.vocab_size

    @torch.inference_mode()
    def run_step(self, sequences: list[Sequence]) -> list[int]:
        """Caller has reserved KV memory for every token; returns the sampled token per sequence."""
        inputs = build_step_inputs(sequences, self.allocator, self._pool, self._device)
        hidden_states = self._model(inputs.token_ids, inputs.context)
        batch_rows = torch.arange(len(sequences), device=self._device)
        logits = self._model.compute_logits(hidden_states[batch_rows, inputs.last_token_indices])
        token_ids = self._sample(logits, sequences)
        for sequence in sequences:
            sequence.num_cached_tokens = sequence.num_tokens
        return token_ids

    def _sample(self, logits: torch.Tensor, sequences: list[Sequence]) -> list[int]:
        """One argmax for the whole batch (one GPU sync); per-sequence sampling only when needed."""
        token_ids = torch.argmax(logits, dim=-1).tolist()
        for row, sequence in enumerate(sequences):
            if sequence.sampling_params.temperature != GREEDY_TEMPERATURE:
                token_ids[row] = sample_next_token(
                    logits[row], sequence.sampling_params, sequence.sampling_generator
                )
        return token_ids
