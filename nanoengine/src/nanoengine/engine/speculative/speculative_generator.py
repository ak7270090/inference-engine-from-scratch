"""Serial generation with speculative decoding: each target pass can produce several tokens.

Per step: the proposer guesses ``k`` tokens, the target scores the last real token plus all ``k``
guesses in one pass, the verifier keeps a prefix of the guesses and adds one target token, and
the KV entries of rejected guesses are dropped by moving ``num_cached_tokens`` back.
"""

from collections.abc import Iterator

import torch

from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.interface import GenerationRequest, TokenEvent
from nanoengine.engine.metrics import EngineMetrics
from nanoengine.engine.sampling import find_finish_reason
from nanoengine.engine.sequence_state import Sequence
from nanoengine.engine.speculative.proposers import Proposal, Proposer
from nanoengine.engine.speculative.verifier import verify_proposal


class SpeculativeGenerator:
    def __init__(
        self,
        runner: ModelRunner,
        proposer: Proposer,
        num_speculative_tokens: int,
        stop_token_ids: frozenset[int],
        device: torch.device,
        metrics: EngineMetrics | None = None,
    ) -> None:
        self._runner = runner
        self._proposer = proposer
        self._num_speculative_tokens = num_speculative_tokens
        self._stop_token_ids = stop_token_ids
        self._device = device
        self._metrics = metrics

    @property
    def vocab_size(self) -> int:
        return self._runner.vocab_size

    def generate(self, request: GenerationRequest) -> Iterator[TokenEvent]:
        sequence = Sequence.from_request(request, self._device)
        try:
            yield from self._generate(sequence)
        finally:
            self._runner.allocator.free(sequence)
            self._proposer.release(sequence)

    def _generate(self, sequence: Sequence) -> Iterator[TokenEvent]:
        """The first step prefills the prompt; every later step verifies a batch of guesses."""
        self._runner.allocator.reserve(sequence, sequence.num_tokens)
        new_tokens = self._runner.run_step([sequence])
        while True:
            for token_id in new_tokens:
                sequence.token_ids.append(token_id)
                finish_reason = find_finish_reason(
                    token_id,
                    sequence.num_generated_tokens,
                    sequence.sampling_params,
                    self._stop_token_ids,
                )
                yield TokenEvent(
                    sequence.request_id, token_id, finish_reason is not None, finish_reason
                )
                if finish_reason is not None:
                    return
            new_tokens = self._run_speculative_step(sequence)

    def _run_speculative_step(self, sequence: Sequence) -> list[int]:
        remaining = sequence.sampling_params.max_new_tokens - sequence.num_generated_tokens
        num_guesses = max(0, min(self._num_speculative_tokens, remaining - 1))
        proposal = self._proposer.propose(sequence, num_guesses) if num_guesses else Proposal([])
        length_before = sequence.num_tokens
        sequence.token_ids.extend(proposal.token_ids)
        self._runner.allocator.reserve(sequence, sequence.num_tokens)
        target_logits = self._runner.score_tokens(sequence, 1 + len(proposal.token_ids))
        new_tokens = verify_proposal(
            proposal, target_logits, sequence.sampling_params, sequence.sampling_generator
        )
        num_kept_guesses = len(new_tokens) - 1
        del sequence.token_ids[length_before:]
        sequence.num_cached_tokens = length_before + num_kept_guesses
        self._record_acceptance(len(proposal.token_ids), num_kept_guesses)
        return new_tokens

    def _record_acceptance(self, num_guesses: int, num_kept: int) -> None:
        if self._metrics is not None:
            self._metrics.speculative_proposed_tokens.inc(num_guesses)
            self._metrics.speculative_accepted_tokens.inc(num_kept)
