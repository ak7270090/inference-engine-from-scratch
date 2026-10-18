"""Cheap guesses for the next few tokens.

``NgramProposer`` (prompt lookup): if the last few tokens appeared earlier, guess that what
followed them comes again. Free, and very good when the answer copies the prompt (code edits).

``DraftModelProposer``: a small model of the same family runs ahead of the target. It keeps its
own KV cache per request and rolls it back to the tokens the target actually kept.
"""

from dataclasses import dataclass
from typing import Protocol

import torch

from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.sampling import (
    GREEDY_TEMPERATURE,
    compute_token_probabilities,
    sample_from_probabilities,
)
from nanoengine.engine.sequence_state import Sequence


@dataclass(frozen=True)
class Proposal:
    token_ids: list[int]
    # [len(token_ids), vocab] distribution each guess was drawn from; None = deterministic guesses
    probabilities: torch.Tensor | None = None


class Proposer(Protocol):
    def propose(self, sequence: Sequence, num_tokens: int) -> Proposal: ...

    def release(self, sequence: Sequence) -> None:
        """The request finished; drop anything kept for it."""
        ...


class NgramProposer:
    def __init__(self, max_ngram: int, min_ngram: int = 1) -> None:
        self._max_ngram = max_ngram
        self._min_ngram = min_ngram

    def propose(self, sequence: Sequence, num_tokens: int) -> Proposal:
        """Longest recent n-gram that occurred before; guess the tokens that followed it."""
        token_ids = sequence.token_ids
        for ngram_size in range(self._max_ngram, self._min_ngram - 1, -1):
            if len(token_ids) <= ngram_size:
                continue
            suffix = token_ids[-ngram_size:]
            for start in range(len(token_ids) - ngram_size - 1, -1, -1):
                if token_ids[start : start + ngram_size] == suffix:
                    follow_start = start + ngram_size
                    guesses = token_ids[follow_start : follow_start + num_tokens]
                    if guesses:
                        return Proposal(guesses)
        return Proposal([])

    def release(self, sequence: Sequence) -> None:
        return None


class DraftModelProposer:
    def __init__(self, runner: ModelRunner, device: torch.device) -> None:
        self._runner = runner
        self._device = device
        self._draft_sequences: dict[str, Sequence] = {}

    def propose(self, sequence: Sequence, num_tokens: int) -> Proposal:
        """Run the draft model ``num_tokens`` steps; the first step catches up on new tokens."""
        draft = self._sync_draft(sequence)
        params = sequence.sampling_params
        guesses: list[int] = []
        distributions = []
        for _ in range(num_tokens):
            self._runner.allocator.reserve(draft, draft.num_tokens)
            logits = self._runner.score_tokens(draft, draft.num_uncached_tokens)[-1]
            probabilities = compute_token_probabilities(logits, params)
            if params.temperature == GREEDY_TEMPERATURE:
                guess = int(torch.argmax(probabilities).item())
            else:
                guess = sample_from_probabilities(probabilities, sequence.sampling_generator)
            draft.token_ids.append(guess)
            guesses.append(guess)
            distributions.append(probabilities)
        return Proposal(guesses, torch.stack(distributions) if distributions else None)

    def release(self, sequence: Sequence) -> None:
        draft = self._draft_sequences.pop(sequence.request_id, None)
        if draft is not None:
            self._runner.allocator.free(draft)

    def _sync_draft(self, sequence: Sequence) -> Sequence:
        """Make the draft's tokens equal the target's; keep its cache for the common prefix."""
        draft = self._draft_sequences.get(sequence.request_id)
        if draft is None:
            draft = Sequence.from_request(sequence.request, self._device)
            self._draft_sequences[sequence.request_id] = draft
        common = 0
        for draft_token, target_token in zip(draft.token_ids, sequence.token_ids, strict=False):
            if draft_token != target_token:
                break
            common += 1
        draft.token_ids = list(sequence.token_ids)
        draft.num_cached_tokens = min(draft.num_cached_tokens, common)
        return draft
