"""The engine interface every phase keeps: requests in, one token event per generated token out."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int
    is_finished: bool
    finish_reason: str | None


class InferenceEngine(Protocol):
    def submit(self, request: GenerationRequest) -> AsyncIterator[TokenEvent]: ...
