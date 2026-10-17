import torch
from transformers import GenerationConfig
from transformers import Qwen3ForCausalLM as HuggingFaceQwen3ForCausalLM

from nanoengine.constants import FINISH_REASON_LENGTH, FINISH_REASON_STOP
from nanoengine.engine_interface import GenerationRequest, SamplingParams
from nanoengine.naive_generator import NaiveGenerator
from nanoengine.sampling import apply_top_p, create_sampling_generator, sample_next_token
from tests.test_qwen3_model import build_hugging_face_twin

CPU = torch.device("cpu")


def build_request(prompt: list[int], **sampling_options) -> GenerationRequest:
    return GenerationRequest("request-1", prompt, SamplingParams(**sampling_options))


def test_greedy_sampling_picks_the_largest_logit() -> None:
    logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
    generator = create_sampling_generator(0, CPU)

    assert sample_next_token(logits, SamplingParams(max_new_tokens=1), generator) == 1


def test_same_seed_gives_the_same_samples() -> None:
    logits = torch.zeros(50)
    params = SamplingParams(max_new_tokens=1, temperature=1.0, seed=3)

    first_generator = create_sampling_generator(3, CPU)
    second_generator = create_sampling_generator(3, CPU)
    first = [sample_next_token(logits, params, first_generator) for _ in range(10)]
    second = [sample_next_token(logits, params, second_generator) for _ in range(10)]

    assert first == second
    assert len(set(first)) > 1


def test_unseeded_generator_still_samples() -> None:
    params = SamplingParams(max_new_tokens=1, temperature=0.5, top_p=0.9)

    token_id = sample_next_token(torch.zeros(5), params, create_sampling_generator(None, CPU))

    assert 0 <= token_id < 5


def test_top_p_keeps_only_the_most_likely_tokens() -> None:
    probabilities = torch.tensor([0.05, 0.6, 0.3, 0.05])

    kept = apply_top_p(probabilities, 0.8)

    assert kept.nonzero().flatten().tolist() == [1, 2]
    torch.testing.assert_close(kept.sum(), torch.tensor(1.0))


def test_tiny_top_p_always_picks_the_top_token() -> None:
    logits = torch.tensor([0.0, 5.0, 1.0])
    params = SamplingParams(max_new_tokens=1, temperature=1.0, top_p=0.01)
    generator = create_sampling_generator(1, CPU)

    assert {sample_next_token(logits, params, generator) for _ in range(20)} == {1}


def test_greedy_generation_matches_transformers_generate(tiny_model) -> None:
    our_model, their_model = build_hugging_face_twin(tie_word_embeddings=True)
    prompt = [3, 9, 27, 1]
    generator = NaiveGenerator(our_model, frozenset(), CPU)

    our_tokens = [
        event.token_id for event in generator.generate(build_request(prompt, max_new_tokens=8))
    ]
    their_output = HuggingFaceQwen3ForCausalLM.generate(
        their_model,
        torch.tensor([prompt]),
        generation_config=GenerationConfig(
            max_new_tokens=8, do_sample=False, eos_token_id=None, pad_token_id=0
        ),
    )

    assert our_tokens == their_output[0, len(prompt) :].tolist()


def test_generation_stops_at_max_tokens_with_length_reason(tiny_model) -> None:
    generator = NaiveGenerator(tiny_model, frozenset(), CPU)

    events = list(generator.generate(build_request([1, 2], max_new_tokens=3)))

    assert len(events) == 3
    assert [event.is_finished for event in events] == [False, False, True]
    assert events[-1].finish_reason == FINISH_REASON_LENGTH


def test_generation_stops_at_a_stop_token_unless_eos_is_ignored(tiny_model) -> None:
    first_token_id = next(
        NaiveGenerator(tiny_model, frozenset(), CPU).generate(
            build_request([1, 2], max_new_tokens=1)
        )
    ).token_id
    generator = NaiveGenerator(tiny_model, frozenset({first_token_id}), CPU)

    stopped = list(generator.generate(build_request([1, 2], max_new_tokens=5)))
    ignored = list(generator.generate(build_request([1, 2], max_new_tokens=5, ignore_eos=True)))

    assert len(stopped) == 1
    assert stopped[0].finish_reason == FINISH_REASON_STOP
    assert len(ignored) == 5
