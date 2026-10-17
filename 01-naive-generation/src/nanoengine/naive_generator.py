"""Generation without a KV cache: the whole sequence goes through the model for every new token."""

from collections.abc import Iterator

import torch

from nanoengine.constants import FINISH_REASON_LENGTH, FINISH_REASON_STOP
from nanoengine.engine_interface import GenerationRequest, SamplingParams, TokenEvent
from nanoengine.qwen3_model import Qwen3ForCausalLM
from nanoengine.sampling import create_sampling_generator, sample_next_token


class NaiveGenerator:
    def __init__(
        self, model: Qwen3ForCausalLM, stop_token_ids: frozenset[int], device: torch.device
    ) -> None:
        self._model = model
        self._stop_token_ids = stop_token_ids
        self._device = device

    @property
    def vocab_size(self) -> int:
        return self._model.config.vocab_size

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """Yield one event per new token; the last one has ``is_finished`` set."""
        params = request.sampling_params
        generator = create_sampling_generator(params.seed, self._device)
        token_ids = list(request.prompt_token_ids)
        for step_index in range(params.max_new_tokens):
            next_token_id = self.run_full_sequence_step(token_ids, params, generator)
            token_ids.append(next_token_id)
            finish_reason = self._find_finish_reason(next_token_id, step_index, params)
            yield TokenEvent(
                request.request_id, next_token_id, finish_reason is not None, finish_reason
            )
            if finish_reason is not None:
                return

    @torch.inference_mode()
    def run_full_sequence_step(
        self, token_ids: list[int], params: SamplingParams, generator: torch.Generator
    ) -> int:
        input_ids = torch.tensor([token_ids], device=self._device)
        hidden_states = self._model(input_ids)
        logits = self._model.compute_logits(hidden_states[0, -1])
        return sample_next_token(logits, params, generator)

    def _find_finish_reason(
        self, token_id: int, step_index: int, params: SamplingParams
    ) -> str | None:
        if not params.ignore_eos and token_id in self._stop_token_ids:
            return FINISH_REASON_STOP
        if step_index + 1 == params.max_new_tokens:
            return FINISH_REASON_LENGTH
        return None
