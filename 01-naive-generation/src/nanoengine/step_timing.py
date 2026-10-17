"""Time one no-cache step at growing sequence lengths: ``python -m nanoengine.step_timing``.

Without a KV cache every step re-reads the whole sequence, so step time grows with length.
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nanoengine.config import load_engine_settings
from nanoengine.constants import TORCH_DTYPES
from nanoengine.engine_interface import SamplingParams
from nanoengine.naive_generator import NaiveGenerator
from nanoengine.sampling import create_sampling_generator
from nanoengine.weight_loader import download_model_files, load_qwen3_model

DEFAULT_SEQUENCE_LENGTHS = (16, 64, 128, 256, 512, 1024)
DEFAULT_REPEATS = 5
WARMUP_STEPS = 2
STEP_TIMING_FILE_STEM = "step-time-vs-length"


@dataclass(frozen=True)
class StepTiming:
    sequence_length: int
    median_step_ms: float
    min_step_ms: float


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_step_timings(
    generator: NaiveGenerator, sequence_lengths: list[int], repeats: int, device: torch.device
) -> list[StepTiming]:
    params = SamplingParams(max_new_tokens=1)
    sampling_generator = create_sampling_generator(0, device)
    timings = []
    for sequence_length in sequence_lengths:
        token_ids = [index % generator.vocab_size for index in range(sequence_length)]
        for _ in range(WARMUP_STEPS):
            generator.run_full_sequence_step(token_ids, params, sampling_generator)
        step_ms = []
        for _ in range(repeats):
            synchronize(device)
            started_at = time.perf_counter()
            generator.run_full_sequence_step(token_ids, params, sampling_generator)
            synchronize(device)
            step_ms.append((time.perf_counter() - started_at) * 1000)
        timings.append(StepTiming(sequence_length, statistics.median(step_ms), min(step_ms)))
    return timings


def render_step_timings_markdown(timings: list[StepTiming], device_label: str) -> str:
    lines = [
        f"# Step time vs sequence length ({device_label}, no KV cache)",
        "",
        "| Sequence length | Median step (ms) | Min step (ms) |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {timing.sequence_length} | {timing.median_step_ms:.1f} | {timing.min_step_ms:.1f} |"
        for timing in timings
    )
    return "\n".join(lines) + "\n"


def draw_step_timings_chart(timings: list[StepTiming], device_label: str, chart_path: Path) -> None:
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
    axis.set_title(f"No KV cache: step time grows with length ({device_label})")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=120)
    pyplot.close(figure)


def write_step_timings(timings: list[StepTiming], device_label: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{STEP_TIMING_FILE_STEM}-{device_label}"
    stem.with_suffix(".md").write_text(
        render_step_timings_markdown(timings, device_label), encoding="utf-8"
    )
    stem.with_suffix(".json").write_text(
        json.dumps([asdict(timing) for timing in timings], indent=2) + "\n", encoding="utf-8"
    )
    draw_step_timings_chart(timings, device_label, stem.with_suffix(".png"))


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nanoengine.step_timing")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_SEQUENCE_LENGTHS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parsed = parser.parse_args(arguments)

    settings = load_engine_settings()
    device = torch.device(settings.device)
    model_dir = download_model_files(settings.model_id, settings.model_cache_dir)
    model = load_qwen3_model(model_dir, device, TORCH_DTYPES[settings.dtype])
    generator = NaiveGenerator(model, frozenset(), device)
    timings = measure_step_timings(generator, parsed.lengths, parsed.repeats, device)
    write_step_timings(timings, f"{device.type}-{settings.dtype}", parsed.output_dir)


if __name__ == "__main__":
    main()
