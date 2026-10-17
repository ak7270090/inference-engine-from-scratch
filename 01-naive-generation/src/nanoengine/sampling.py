"""Pick the next token from logits: greedy at temperature 0, otherwise temperature + top-p."""

import torch

from nanoengine.engine_interface import SamplingParams

GREEDY_TEMPERATURE = 0.0


def create_sampling_generator(seed: int | None, device: torch.device) -> torch.Generator:
    """A per-request random generator, so a fixed seed always gives the same tokens."""
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def apply_top_p(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of most likely tokens whose probabilities add up to ``top_p``."""
    sorted_probabilities, sorted_token_ids = torch.sort(probabilities, descending=True)
    probability_before = torch.cumsum(sorted_probabilities, dim=-1) - sorted_probabilities
    sorted_probabilities[probability_before >= top_p] = 0.0
    kept = torch.zeros_like(probabilities).scatter(-1, sorted_token_ids, sorted_probabilities)
    return kept / kept.sum()


def sample_next_token(
    logits: torch.Tensor, params: SamplingParams, generator: torch.Generator
) -> int:
    """``logits`` is ``[vocab_size]`` for the last position."""
    if params.temperature == GREEDY_TEMPERATURE:
        return int(torch.argmax(logits).item())
    probabilities = torch.softmax(logits.float() / params.temperature, dim=-1)
    if params.top_p < 1.0:
        probabilities = apply_top_p(probabilities, params.top_p)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())
