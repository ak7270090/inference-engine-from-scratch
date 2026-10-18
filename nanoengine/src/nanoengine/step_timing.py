"""Time one decode step at growing sequence lengths: ``python -m nanoengine.step_timing``.

Without a KV cache (``kv_cache=none``) every step recomputes the whole sequence, so step time grows
with length. With a KV cache a step computes one token and only reads the cached keys and values.
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch

from nanoengine.config import load_engine_settings
from nanoengine.constants import KV_CACHE_NONE, TORCH_DTYPES
from nanoengine.engine.generation.model_runner import ModelRunner
from nanoengine.engine.generation.naive_generator import NaiveGenerator
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.memory.contiguous_allocator import ContiguousSlotAllocator
from nanoengine.engine.memory.kv_cache_pool import KVCachePool
from nanoengine.engine.sampling import create_sampling_generator
from nanoengine.engine.sequence_state import Sequence
from nanoengine.models.qwen3 import Qwen3ForCausalLM
from nanoengine.models.weight_loader import download_model_files, load_qwen3_model

DEFAULT_SEQUENCE_LENGTHS = (16, 64, 128, 256, 512, 1024)
DEFAULT_REPEATS = 5
WARMUP_STEPS = 2
STEP_TIMING_FILE_STEM = "step-time-vs-length"
KV_CACHE_LABEL_SUFFIX = "-kv-cache"


@dataclass(frozen=True)
class StepTiming:
    sequence_length: int
    median_step_ms: float
    min_step_ms: float


class StepBenchmark(Protocol):
    description: str

    def run_step(self, token_ids: list[int]) -> None: ...


class NaiveStepBenchmark:
    """One no-cache step: the whole sequence goes through the model."""

    description = "no KV cache"

    def __init__(self, model: Qwen3ForCausalLM, device: torch.device) -> None:
        self._generator = NaiveGenerator(model, frozenset(), device)
        self._params = SamplingParams(max_new_tokens=1)
        self._sampling_generator = create_sampling_generator(0, device)

    def run_step(self, token_ids: list[int]) -> None:
        self._generator.run_full_sequence_step(token_ids, self._params, self._sampling_generator)


class CachedDecodeStepBenchmark:
    """One decode step with ``len(token_ids) - 1`` tokens already in the KV cache."""

    description = "KV cache"

    def __init__(
        self, model: Qwen3ForCausalLM, device: torch.device, dtype: torch.dtype, max_length: int
    ) -> None:
        self._allocator = ContiguousSlotAllocator(1, max_length, device)
        pool = KVCachePool(model.config, max_length, device, dtype)
        self._runner = ModelRunner(model, pool, self._allocator, device)
        self._device = device

    def run_step(self, token_ids: list[int]) -> None:
        request = GenerationRequest("step-timing", token_ids, SamplingParams(max_new_tokens=1))
        sequence = Sequence.from_request(request, self._device)
        sequence.num_cached_tokens = len(token_ids) - 1
        self._allocator.reserve(sequence, sequence.num_tokens)
        self._runner.run_step([sequence])
        self._allocator.free(sequence)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_step_timings(
    benchmark: StepBenchmark,
    sequence_lengths: list[int],
    repeats: int,
    device: torch.device,
    vocab_size: int,
) -> list[StepTiming]:
    timings = []
    for sequence_length in sequence_lengths:
        token_ids = [index % vocab_size for index in range(sequence_length)]
        for _ in range(WARMUP_STEPS):
            benchmark.run_step(token_ids)
        step_ms = []
        for _ in range(repeats):
            synchronize(device)
            started_at = time.perf_counter()
            benchmark.run_step(token_ids)
            synchronize(device)
            step_ms.append((time.perf_counter() - started_at) * 1000)
        timings.append(StepTiming(sequence_length, statistics.median(step_ms), min(step_ms)))
    return timings


def render_step_timings_markdown(
    timings: list[StepTiming], device_label: str, description: str
) -> str:
    lines = [
        f"# Step time vs sequence length ({device_label}, {description})",
        "",
        "| Sequence length | Median step (ms) | Min step (ms) |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {timing.sequence_length} | {timing.median_step_ms:.1f} | {timing.min_step_ms:.1f} |"
        for timing in timings
    )
    return "\n".join(lines) + "\n"


def draw_step_timings_chart(
    timings: list[StepTiming], device_label: str, description: str, chart_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axis = pyplot.subplots(figsize=(6, 4))
    axis.plot(
        [timing.sequence_length for timing in timings],
        [timing.median_step_ms for timing in timings],
        marker="o",
    )
    axis.set_xlabel("sequence length (tokens)")
    axis.set_ylabel("one decode step (ms)")
    axis.set_title(f"Decode step time, {description} ({device_label})")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=120)
    pyplot.close(figure)


def write_step_timings(
    timings: list[StepTiming], device_label: str, description: str, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{STEP_TIMING_FILE_STEM}-{device_label}"
    stem.with_suffix(".md").write_text(
        render_step_timings_markdown(timings, device_label, description), encoding="utf-8"
    )
    stem.with_suffix(".json").write_text(
        json.dumps([asdict(timing) for timing in timings], indent=2) + "\n", encoding="utf-8"
    )
    draw_step_timings_chart(timings, device_label, description, stem.with_suffix(".png"))


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nanoengine.step_timing")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_SEQUENCE_LENGTHS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parsed = parser.parse_args(arguments)

    settings = load_engine_settings()
    device = torch.device(settings.device)
    dtype = TORCH_DTYPES[settings.dtype]
    model_dir = download_model_files(settings.model_id, settings.model_cache_dir)
    model = load_qwen3_model(model_dir, device, dtype)
    device_label = f"{device.type}-{settings.dtype}"
    if settings.kv_cache == KV_CACHE_NONE:
        benchmark: StepBenchmark = NaiveStepBenchmark(model, device)
    else:
        benchmark = CachedDecodeStepBenchmark(model, device, dtype, max(parsed.lengths))
        device_label += KV_CACHE_LABEL_SUFFIX
    timings = measure_step_timings(
        benchmark, parsed.lengths, parsed.repeats, device, model.config.vocab_size
    )
    write_step_timings(timings, device_label, benchmark.description, parsed.output_dir)


if __name__ == "__main__":
    main()
