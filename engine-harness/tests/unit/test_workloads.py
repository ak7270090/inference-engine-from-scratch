import random

import pytest

from engineharness.constants import SCALE_CPU, SCALE_T4, WORKLOAD_SHARED_PREFIX
from engineharness.workloads import (
    WORKLOAD_SPECS,
    CompletionRequest,
    WorkloadSpec,
    build_equivalence_requests,
    build_requests,
    build_token_stream,
    get_workload_spec,
)


def test_token_stream_has_exactly_the_requested_length(word_tokenizer) -> None:
    tokens = build_token_stream(word_tokenizer, 137, random.Random(0))

    assert len(tokens) == 137


@pytest.mark.parametrize("scale", [SCALE_CPU, SCALE_T4])
def test_every_standard_workload_builds_exact_prompt_lengths(word_tokenizer, scale) -> None:
    for spec in WORKLOAD_SPECS[scale].values():
        requests = build_requests(spec, word_tokenizer, seed=7)

        assert len(requests) == spec.request_count
        assert all(len(r.prompt_token_ids) == spec.prompt_tokens for r in requests)
        assert all(
            spec.output_tokens_min <= r.max_tokens <= spec.output_tokens_max for r in requests
        )


def test_shared_prefix_requests_start_with_the_same_tokens(word_tokenizer) -> None:
    spec = get_workload_spec(SCALE_CPU, WORKLOAD_SHARED_PREFIX)

    requests = build_requests(spec, word_tokenizer, seed=3)

    prefixes = {r.prompt_token_ids[: spec.shared_prefix_tokens] for r in requests}
    tails = {r.prompt_token_ids[spec.shared_prefix_tokens :] for r in requests}
    assert len(prefixes) == 1
    assert len(tails) == len(requests)


def test_same_seed_gives_same_requests(word_tokenizer) -> None:
    spec = WorkloadSpec("mixed", 5, 20, 4, 40)

    assert build_requests(spec, word_tokenizer, 11) == build_requests(spec, word_tokenizer, 11)
    assert build_requests(spec, word_tokenizer, 11) != build_requests(spec, word_tokenizer, 12)


def test_equivalence_requests_are_greedy_and_fixed_length(word_tokenizer) -> None:
    requests = build_equivalence_requests(word_tokenizer, prompt_count=4, output_tokens=16, seed=1)

    assert [r.request_id for r in requests] == [f"equivalence-{i}" for i in range(4)]
    assert all(r.temperature == 0.0 and r.max_tokens == 16 for r in requests)


def test_streaming_payload_asks_for_usage_and_ignores_eos() -> None:
    request = CompletionRequest("r", (5, 6), max_tokens=8, seed=42)

    payload = request.build_payload("qwen", is_streaming=True)

    assert payload["prompt"] == [5, 6]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["ignore_eos"] is True
    assert payload["seed"] == 42
    assert "stream_options" not in request.build_payload("qwen", is_streaming=False)


def test_unknown_workload_raises() -> None:
    with pytest.raises(KeyError):
        get_workload_spec(SCALE_CPU, "nope")
