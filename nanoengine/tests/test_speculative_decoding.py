"""Phase 6: speculative decoding must give exactly the target model's output."""

import pytest
import torch

from nanoengine import speculative_benchmark
from nanoengine.config import EngineSettings
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.metrics import EngineMetrics
from nanoengine.engine.runtime import create_engine
from nanoengine.engine.sampling import create_sampling_generator
from nanoengine.engine.sequence_state import Sequence
from nanoengine.engine.speculative.proposers import NgramProposer, Proposal
from nanoengine.engine.speculative.verifier import verify_proposal
from nanoengine.speculative_benchmark import render_results_markdown, run_benchmark
from tests.conftest import MODEL_SEED, build_tiny_model
from tests.test_batch_engine_loop import (
    PROMPTS,
    build_settings,
    expected_greedy_tokens,
    serve_all,
)

CPU = torch.device("cpu")
GREEDY = SamplingParams(max_new_tokens=8)
SAMPLED = SamplingParams(max_new_tokens=8, temperature=1.0)
CHI_SQUARE_CRITICAL_4_DEGREES_999 = 18.47
CHI_SQUARE_TRIALS = 6000


def build_sequence(token_ids: list[int]) -> Sequence:
    return Sequence.from_request(GenerationRequest("s", token_ids, GREEDY), CPU)


def test_ngram_proposer_repeats_what_followed_the_last_ngram() -> None:
    proposer = NgramProposer(max_ngram=2)

    repeated = proposer.propose(build_sequence([1, 2, 3, 4, 5, 9, 1, 2]), 3)
    unseen = proposer.propose(build_sequence([1, 2, 3, 4]), 3)

    assert repeated.token_ids == [3, 4, 5]
    assert unseen.token_ids == []


def test_greedy_verification_keeps_matching_guesses_and_adds_the_target_token() -> None:
    logits = torch.full((4, 6), -10.0)
    for row, best in enumerate([2, 4, 1, 5]):
        logits[row, best] = 10.0
    generator = create_sampling_generator(0, CPU)

    all_kept = verify_proposal(Proposal([2, 4, 1]), logits, GREEDY, generator)
    first_wrong = verify_proposal(Proposal([2, 3, 1]), logits, GREEDY, generator)

    assert all_kept == [2, 4, 1, 5]
    assert first_wrong == [2, 4]


def test_sampled_tokens_follow_the_target_distribution() -> None:
    """Chi-square: with a draft that disagrees with the target, the first verified token is
    still distributed like the target's own samples."""
    target_probabilities = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])
    draft_probabilities = torch.tensor([0.05, 0.1, 0.15, 0.3, 0.4])
    target_logits = torch.log(target_probabilities).repeat(2, 1)
    generator = create_sampling_generator(7, CPU)
    counts = torch.zeros(5)

    for _ in range(CHI_SQUARE_TRIALS):
        guess = int(torch.multinomial(draft_probabilities, 1, generator=generator).item())
        proposal = Proposal([guess], draft_probabilities[None, :])
        counts[verify_proposal(proposal, target_logits, SAMPLED, generator)[0]] += 1

    expected = target_probabilities * CHI_SQUARE_TRIALS
    chi_square = float(((counts - expected) ** 2 / expected).sum())
    assert chi_square < CHI_SQUARE_CRITICAL_4_DEGREES_999


@pytest.mark.parametrize("proposer_kind", ["ngram", "draft", "draft_identical"])
@pytest.mark.anyio
async def test_speculative_output_equals_the_target_alone(tiny_model, proposer_kind) -> None:
    metrics = EngineMetrics()
    speculative = "ngram" if proposer_kind == "ngram" else "draft"
    draft_model = (
        tiny_model if proposer_kind == "draft_identical" else build_tiny_model(seed=MODEL_SEED + 1)
    )
    settings = build_settings(
        "contiguous", "serial", speculative=speculative, num_speculative_tokens=3
    )
    engine = create_engine(tiny_model, frozenset(), settings, metrics, draft_model=draft_model)

    outputs = await serve_all(engine, PROMPTS)

    assert outputs == [expected_greedy_tokens(tiny_model, prompt) for prompt in PROMPTS]
    proposed = metrics.registry.get_sample_value("nanoengine_speculative_proposed_tokens_total")
    accepted = metrics.registry.get_sample_value("nanoengine_speculative_accepted_tokens_total")
    assert 0 <= accepted <= proposed
    if proposer_kind == "draft_identical":
        assert accepted == proposed > 0


@pytest.mark.anyio
async def test_speculation_works_on_a_paged_cache_with_stop_tokens(tiny_model) -> None:
    stop_token = expected_greedy_tokens(tiny_model, PROMPTS[0])[3]
    settings = build_settings("paged", "serial", speculative="ngram")
    engine = create_engine(tiny_model, frozenset({stop_token}), settings, EngineMetrics())
    request = GenerationRequest("stop", PROMPTS[0], SamplingParams(max_new_tokens=12))
    await engine.start()

    events = [event async for event in engine.submit(request)]
    await engine.stop()

    assert events[-1].finish_reason == "stop" and events[-1].token_id == stop_token
    assert len(events) <= 4


def test_speculation_needs_serial_scheduling_without_prefix_cache() -> None:
    with pytest.raises(ValueError, match="speculative decoding needs"):
        EngineSettings(kv_cache="paged", scheduler="continuous", speculative="ngram")
    with pytest.raises(ValueError, match="speculative decoding needs"):
        EngineSettings(kv_cache="paged", prefix_cache=True, speculative="draft")


def test_benchmark_reports_speed_up_and_acceptance(tiny_model, tiny_tokenizer, monkeypatch) -> None:
    monkeypatch.setattr(speculative_benchmark, "GENERATED_TOKENS", 8)
    monkeypatch.setattr(speculative_benchmark, "REPEATS", 1)
    monkeypatch.setattr(
        speculative_benchmark, "PROMPT_STYLES", {"repeat": "w1 w2 w3 w1 w2 w3 w1", "chat": "w9"}
    )
    settings = EngineSettings(kv_cache="contiguous", speculative="ngram", max_model_len=64)

    results = run_benchmark(tiny_model, None, tiny_tokenizer, settings)

    assert [result.prompt_style for result in results] == ["repeat", "chat"]
    assert all(result.speculative_tokens_per_second > 0 for result in results)
    assert "Tokens per pass" in render_results_markdown(results, settings)


def test_benchmark_cli_writes_its_report(monkeypatch, tiny_model, tiny_tokenizer, tmp_path) -> None:
    monkeypatch.setenv("NANOENGINE_KV_CACHE", "contiguous")
    monkeypatch.setenv("NANOENGINE_SPECULATIVE", "ngram")
    monkeypatch.setenv("NANOENGINE_MAX_MODEL_LEN", "64")
    monkeypatch.setattr(speculative_benchmark, "GENERATED_TOKENS", 4)
    monkeypatch.setattr(speculative_benchmark, "REPEATS", 1)
    monkeypatch.setattr(speculative_benchmark, "PROMPT_STYLES", {"repeat": "w1 w2 w1 w2"})
    monkeypatch.setattr(
        speculative_benchmark, "download_model_files", lambda model_id, cache_dir: tmp_path
    )
    monkeypatch.setattr(
        speculative_benchmark, "load_qwen3_model", lambda model_dir, device, dtype: tiny_model
    )
    monkeypatch.setattr(
        speculative_benchmark, "load_text_tokenizer", lambda model_dir: tiny_tokenizer
    )

    speculative_benchmark.main(["--output-dir", str(tmp_path)])

    assert (tmp_path / "speculative-benchmark.json").exists()
