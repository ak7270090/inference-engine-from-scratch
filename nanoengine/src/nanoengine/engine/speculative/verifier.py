"""Check guesses against the target model so the output distribution stays exactly the target's.

Guess ``i`` (drawn with probability ``q``) is kept with probability ``min(1, p / q)``, where ``p``
is the target's probability of that token. At the first rejection the token is redrawn from
``max(0, p - q)`` (normalized), and the rest are dropped. If every guess is kept, the target's
logits after the last guess give one extra token for free. With greedy decoding ``p`` and ``q``
are one-hot, so this reduces to "keep guesses while they equal the target's argmax".
"""

import torch

from nanoengine.engine.interface import SamplingParams
from nanoengine.engine.sampling import (
    GREEDY_TEMPERATURE,
    compute_token_probabilities,
    sample_from_probabilities,
)
from nanoengine.engine.speculative.proposers import Proposal


def pick_token(probabilities: torch.Tensor, params: SamplingParams, generator) -> int:
    if params.temperature == GREEDY_TEMPERATURE:
        return int(torch.argmax(probabilities).item())
    return sample_from_probabilities(probabilities, generator)


def verify_proposal(
    proposal: Proposal,
    target_logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator,
) -> list[int]:
    """``target_logits`` is ``[len(guesses) + 1, vocab]``; returns the kept guesses plus one token
    from the target (the correction at the first rejection, or the bonus after the last guess)."""
    target_probabilities = compute_token_probabilities(target_logits, params)
    vocab_size = target_probabilities.shape[-1]
    kept: list[int] = []
    for index, guess in enumerate(proposal.token_ids):
        if proposal.probabilities is None:
            draft_probabilities = torch.nn.functional.one_hot(
                torch.tensor(guess, device=target_probabilities.device), vocab_size
            ).float()
        else:
            draft_probabilities = proposal.probabilities[index].float()
        target_probability = target_probabilities[index, guess]
        draft_probability = draft_probabilities[guess].clamp(min=1e-12)
        acceptance = (target_probability / draft_probability).clamp(max=1.0).item()
        if acceptance >= 1.0 or (
            acceptance > 0.0
            and torch.rand((), device=generator.device, generator=generator).item() < acceptance
        ):
            kept.append(guess)
            continue
        residual = (target_probabilities[index] - draft_probabilities).clamp(min=0.0)
        if residual.sum() <= 0:
            residual = target_probabilities[index]
        residual = residual / residual.sum()
        return [*kept, pick_token(residual, params, generator)]
    bonus = pick_token(target_probabilities[len(proposal.token_ids)], params, generator)
    return [*kept, bonus]
