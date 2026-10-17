"""Greedy outputs per prompt, saved as the reference by phase 01 and compared by later phases."""

import json
from dataclasses import dataclass
from os.path import commonprefix
from pathlib import Path

import aiohttp

from engineharness.streaming_client import stream_completion
from engineharness.workloads import CompletionRequest


@dataclass(frozen=True)
class EquivalenceResult:
    """How many outputs match the reference exactly, and how much of the rest matches."""

    prompt_count: int
    identical_count: int
    mean_common_prefix_ratio: float
    mismatched_request_ids: tuple[str, ...]

    @property
    def agreement(self) -> float:
        return self.identical_count / self.prompt_count if self.prompt_count else 0.0


async def collect_greedy_outputs(
    session: aiohttp.ClientSession, requests: list[CompletionRequest], model_id: str
) -> dict[str, str]:
    """Run requests one at a time (batch size 1) and return ``request_id -> text``."""
    outputs = {}
    for request in requests:
        timing = await stream_completion(session, request, model_id)
        if timing.error:
            raise RuntimeError(f"{request.request_id} failed: {timing.error}")
        outputs[request.request_id] = timing.text
    return outputs


def compare_outputs(reference: dict[str, str], actual: dict[str, str]) -> EquivalenceResult:
    """Compare every reference prompt; a missing output counts as a full mismatch."""
    mismatched = []
    prefix_ratios = []
    for request_id, expected_text in reference.items():
        actual_text = actual.get(request_id, "")
        if actual_text != expected_text:
            mismatched.append(request_id)
        shared_length = len(commonprefix([expected_text, actual_text]))
        prefix_ratios.append(shared_length / len(expected_text) if expected_text else 1.0)
    prompt_count = len(reference)
    return EquivalenceResult(
        prompt_count=prompt_count,
        identical_count=prompt_count - len(mismatched),
        mean_common_prefix_ratio=sum(prefix_ratios) / prompt_count if prompt_count else 0.0,
        mismatched_request_ids=tuple(mismatched),
    )


def save_reference(path: Path, outputs: dict[str, str], source: str) -> None:
    """Write the reference outputs with a note on which engine produced them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"source": source, "outputs": outputs}
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_reference(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))["outputs"]
