"""Standard workloads as lists of completion requests with exact prompt and output token counts.

Prompts are sent as token IDs (the OpenAI completions API accepts a list of ints), so prompt
lengths are exact for every engine, and ``ignore_eos`` keeps output lengths fixed.
"""

import random
from dataclasses import dataclass
from typing import Protocol

from engineharness.constants import (
    CORPUS_TEXT,
    SCALE_CPU,
    SCALE_T4,
    WORKLOAD_BURST,
    WORKLOAD_EQUIVALENCE,
    WORKLOAD_LONG_PROMPT,
    WORKLOAD_MIXED_LENGTHS,
    WORKLOAD_SHARED_PREFIX,
    WORKLOAD_SHORT_CHAT,
)


class TextTokenizer(Protocol):
    """Anything that turns text into token IDs."""

    def encode(self, text: str) -> list[int]: ...


class HubTokenizer:
    """The model's own tokenizer, downloaded from the Hugging Face Hub (``tokenizer.json`` only)."""

    def __init__(self, tokenizer_id: str) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_pretrained(tokenizer_id)

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids


@dataclass(frozen=True)
class WorkloadSpec:
    """The shape of one workload: how many requests and how many tokens in and out."""

    name: str
    request_count: int
    prompt_tokens: int
    output_tokens_min: int
    output_tokens_max: int
    shared_prefix_tokens: int = 0


@dataclass(frozen=True)
class CompletionRequest:
    """One request of a workload."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    temperature: float = 0.0
    seed: int | None = None
    ignore_eos: bool = True

    def build_payload(self, model_id: str, is_streaming: bool) -> dict:
        payload = {
            "model": model_id,
            "prompt": list(self.prompt_token_ids),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": is_streaming,
            "ignore_eos": self.ignore_eos,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if is_streaming:
            payload["stream_options"] = {"include_usage": True}
        return payload


WORKLOAD_SPECS: dict[str, dict[str, WorkloadSpec]] = {
    SCALE_CPU: {
        WORKLOAD_SHORT_CHAT: WorkloadSpec(WORKLOAD_SHORT_CHAT, 8, 32, 32, 32),
        WORKLOAD_LONG_PROMPT: WorkloadSpec(WORKLOAD_LONG_PROMPT, 4, 256, 8, 8),
        WORKLOAD_SHARED_PREFIX: WorkloadSpec(WORKLOAD_SHARED_PREFIX, 8, 144, 16, 16, 128),
        WORKLOAD_MIXED_LENGTHS: WorkloadSpec(WORKLOAD_MIXED_LENGTHS, 8, 32, 8, 64),
        WORKLOAD_BURST: WorkloadSpec(WORKLOAD_BURST, 16, 32, 32, 32),
    },
    SCALE_T4: {
        WORKLOAD_SHORT_CHAT: WorkloadSpec(WORKLOAD_SHORT_CHAT, 64, 64, 128, 128),
        WORKLOAD_LONG_PROMPT: WorkloadSpec(WORKLOAD_LONG_PROMPT, 16, 1024, 32, 32),
        WORKLOAD_SHARED_PREFIX: WorkloadSpec(WORKLOAD_SHARED_PREFIX, 64, 544, 64, 64, 512),
        WORKLOAD_MIXED_LENGTHS: WorkloadSpec(WORKLOAD_MIXED_LENGTHS, 64, 64, 16, 512),
        WORKLOAD_BURST: WorkloadSpec(WORKLOAD_BURST, 64, 64, 128, 128),
    },
}


def get_workload_spec(scale: str, workload_name: str) -> WorkloadSpec:
    """Look up a standard workload; raises ``KeyError`` for an unknown scale or name."""
    return WORKLOAD_SPECS[scale][workload_name]


def build_token_stream(tokenizer: TextTokenizer, token_count: int, rng: random.Random) -> list[int]:
    """At least ``token_count`` token IDs of natural-looking text, shuffled by ``rng``."""
    token_ids: list[int] = []
    while len(token_ids) < token_count:
        words = CORPUS_TEXT.split()
        rng.shuffle(words)
        token_ids.extend(tokenizer.encode(" " + " ".join(words)))
    return token_ids[:token_count]


def build_requests(
    spec: WorkloadSpec, tokenizer: TextTokenizer, seed: int
) -> list[CompletionRequest]:
    """Requests for ``spec``; the same seed always gives the same prompts and lengths."""
    rng = random.Random(seed)
    shared_prefix = build_token_stream(tokenizer, spec.shared_prefix_tokens, rng)
    unique_tokens = spec.prompt_tokens - spec.shared_prefix_tokens
    requests = []
    for index in range(spec.request_count):
        prompt = shared_prefix + build_token_stream(tokenizer, unique_tokens, rng)
        max_tokens = rng.randint(spec.output_tokens_min, spec.output_tokens_max)
        requests.append(
            CompletionRequest(
                request_id=f"{spec.name}-{index}",
                prompt_token_ids=tuple(prompt),
                max_tokens=max_tokens,
            )
        )
    return requests


def build_equivalence_requests(
    tokenizer: TextTokenizer, prompt_count: int, output_tokens: int, seed: int
) -> list[CompletionRequest]:
    """Greedy requests with varied prompt lengths, compared token by token across phases."""
    rng = random.Random(seed)
    requests = []
    for index in range(prompt_count):
        prompt_length = rng.randint(8, 96)
        requests.append(
            CompletionRequest(
                request_id=f"{WORKLOAD_EQUIVALENCE}-{index}",
                prompt_token_ids=tuple(build_token_stream(tokenizer, prompt_length, rng)),
                max_tokens=output_tokens,
            )
        )
    return requests
