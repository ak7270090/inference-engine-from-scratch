"""One request's generation state while the engine works on it."""

import time
from dataclasses import dataclass, field

import torch

from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.sampling import create_sampling_generator


@dataclass(eq=False)
class Sequence:
    """Prompt plus generated tokens; ``num_cached_tokens`` of them have keys / values cached.

    Compared by identity, so ``sequence in running`` finds this exact sequence.
    """

    request: GenerationRequest
    sampling_generator: torch.Generator
    token_ids: list[int]
    num_prompt_tokens: int
    num_cached_tokens: int = 0
    arrival_time: float = field(default_factory=time.perf_counter)

    @classmethod
    def from_request(cls, request: GenerationRequest, device: torch.device) -> "Sequence":
        return cls(
            request=request,
            sampling_generator=create_sampling_generator(request.sampling_params.seed, device),
            token_ids=list(request.prompt_token_ids),
            num_prompt_tokens=len(request.prompt_token_ids),
        )

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def sampling_params(self) -> SamplingParams:
        return self.request.sampling_params

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def num_uncached_tokens(self) -> int:
        return len(self.token_ids) - self.num_cached_tokens

    def forget_cache(self) -> None:
        """After preemption: every token must be recomputed (prefilled) again."""
        self.num_cached_tokens = 0
