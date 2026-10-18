"""Generation with a KV cache, one request at a time: prefill the prompt once, then one token
per step that reads the cached keys and values instead of recomputing them."""

from collections.abc import Iterator

import torch

from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.interface import GenerationRequest, TokenEvent
from nanoengine.engine.sampling import find_finish_reason
from nanoengine.engine.sequence_state import Sequence


class CachedGenerator:
    def __init__(
        self, runner: ModelRunner, stop_token_ids: frozenset[int], device: torch.device
    ) -> None:
        self._runner = runner
        self._stop_token_ids = stop_token_ids
        self._device = device

    @property
    def vocab_size(self) -> int:
        return self._runner.vocab_size

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        """The first step is the prefill (whole prompt), every later step decodes one token."""
        sequence = Sequence.from_request(request, self._device)
        try:
            while True:
                self._runner.allocator.reserve(sequence, sequence.num_tokens)
                token_id = self._runner.run_step([sequence])[0]
                sequence.token_ids.append(token_id)
                finish_reason = find_finish_reason(
                    token_id,
                    sequence.num_generated_tokens,
                    sequence.sampling_params,
                    self._stop_token_ids,
                )
                yield TokenEvent(
                    request.request_id, token_id, finish_reason is not None, finish_reason
                )
                if finish_reason is not None:
                    return
        finally:
            self._runner.allocator.free(sequence)
