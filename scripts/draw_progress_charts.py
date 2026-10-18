"""Draw the cross-phase comparison charts: ``python scripts/draw_progress_charts.py``.

Only phases whose results exist are drawn, so the charts can be redrawn after every new run.
Needs matplotlib (the engine-harness venv has it).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "results"
IMAGES_DIR = REPO_ROOT / "docs" / "images"

SURFACE_COLOR = "#fcfcfb"
INK_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e6e5e1"
# one fixed color per engine, so an engine keeps its color on every chart
ENGINES = (
    ("phase-01-naive-t4", "1: naive", "#2a78d6"),
    ("phase-02-kv-cache-t4", "2: KV cache", "#eb6834"),
    ("phase-03-static-batching-t4", "3: static batching", "#1baf7a"),
    ("phase-03-continuous-batching-t4", "3: continuous batching", "#eda100"),
    ("phase-04-paged-t4", "4: paged KV cache", "#e87ba4"),
    ("phase-05-prefix-chunked-t4", "5: prefix cache + chunked prefill", "#008300"),
    ("phase-07-cuda-graphs-t4", "7: + CUDA graphs", "#6250d6"),
    ("vllm-t4", "vLLM 0.31.0", "#e34948"),
)
# phases 2-4 share the same single-sequence decode step, so one line stands for all of them
STEP_TIME_ENGINES = {"phase-01-naive-t4", "phase-02-kv-cache-t4", "phase-07-cuda-graphs-t4"}
SPECULATIVE_RUNS = (
    ("phase-06-ngram-t4", "n-gram, Qwen3-0.6B"),
    ("phase-06-draft-t4", "0.6B draft → 1.7B target"),
)


def read_json(path: Path) -> list[dict] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def style_axis(axis, title: str, x_label: str, y_label: str) -> None:
    axis.set_facecolor(SURFACE_COLOR)
    axis.set_title(title, color=INK_COLOR, loc="left", fontsize=11)
    axis.set_xlabel(x_label, color=MUTED_COLOR)
    axis.set_ylabel(y_label, color=MUTED_COLOR)
    axis.grid(color=GRID_COLOR, zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED_COLOR)
    axis.tick_params(colors=MUTED_COLOR)


def save(figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(IMAGES_DIR / name, dpi=130, facecolor=SURFACE_COLOR)
    pyplot.close(figure)
    print(f"wrote docs/images/{name}")


def draw_step_time_chart() -> None:
    figure, axis = pyplot.subplots(figsize=(7, 4.2), facecolor=SURFACE_COLOR)
    for directory, label, color in ENGINES:
        if directory not in STEP_TIME_ENGINES:
            continue
        for path in sorted(
            (RESULTS_DIR / directory / "results").glob("step-time-vs-length-*.json")
        ):
            timings = read_json(path)
            lengths = [timing["sequence_length"] for timing in timings]
            step_ms = [timing["median_step_ms"] for timing in timings]
            axis.plot(
                lengths, step_ms, marker="o", markersize=5, linewidth=2, color=color, label=label
            )
    style_axis(
        axis,
        "One decode step on a T4 vs sequence length",
        "sequence length (tokens)",
        "ms per step",
    )
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, labelcolor=INK_COLOR)
    save(figure, "progress-step-time.png")


def draw_throughput_chart(workload: str, file_name: str, title: str) -> None:
    figure, axis = pyplot.subplots(figsize=(7, 4.2), facecolor=SURFACE_COLOR)
    for directory, label, color in ENGINES:
        levels = read_json(RESULTS_DIR / directory / "results" / workload / "metrics.json")
        if not levels:
            continue
        users = [level["concurrency"] for level in levels]
        throughput = [level["total_tokens_per_second"] for level in levels]
        axis.plot(
            users, throughput, marker="o", markersize=5, linewidth=2, color=color, label=label
        )
    style_axis(axis, title, "concurrent users", "total tokens / s")
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, labelcolor=INK_COLOR, fontsize=8)
    save(figure, file_name)


def draw_speculative_chart() -> None:
    runs = [
        (label, read_json(RESULTS_DIR / directory / "results" / "speculative-benchmark.json"))
        for directory, label in SPECULATIVE_RUNS
    ]
    runs = [(label, results) for label, results in runs if results]
    if not runs:
        return
    styles = [result["prompt_style"] for result in runs[0][1]]
    figure, axis = pyplot.subplots(figsize=(7, 4.2), facecolor=SURFACE_COLOR)
    width = 0.36
    for run_index, ((label, results), color) in enumerate(
        zip(runs, ("#2a78d6", "#eb6834"), strict=False)
    ):
        offsets = [index + (run_index - 0.5) * (width + 0.02) for index in range(len(styles))]
        speed_ups = [result["speed_up"] for result in results]
        bars = axis.bar(offsets, speed_ups, width, color=color, label=label, zorder=2)
        for bar, speed_up in zip(bars, speed_ups, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                speed_up + 0.05,
                f"{speed_up:.2f}x",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK_COLOR,
            )
    axis.axhline(1.0, color=MUTED_COLOR, linewidth=1, linestyle="--", zorder=1)
    style_axis(
        axis,
        "Speculative decoding on a T4: one user, speed-up over no speculation",
        "prompt style",
        "speed-up",
    )
    axis.set_xticks(range(len(styles)), [style.replace("_", " ") for style in styles])
    axis.legend(frameon=False, labelcolor=INK_COLOR)
    save(figure, "progress-speculative.png")


def main() -> None:
    draw_step_time_chart()
    draw_throughput_chart(
        "short_chat",
        "progress-throughput-short-chat.png",
        "short_chat on a T4: total throughput as users grow",
    )
    draw_throughput_chart(
        "mixed_lengths",
        "progress-throughput-mixed-lengths.png",
        "mixed_lengths (16-512 output tokens) on a T4",
    )
    draw_speculative_chart()


if __name__ == "__main__":
    main()
