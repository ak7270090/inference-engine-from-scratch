"""Draw the cross-phase comparison charts: ``python scripts/draw_progress_charts.py``.

Only phases whose results exist are drawn, so the charts can be redrawn after every new run.
Every bar and point is labeled with its value. Needs matplotlib (the engine-harness venv has it).
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "results"
IMAGES_DIR = REPO_ROOT / "docs" / "images"

SURFACE_COLOR = "#fcfcfb"
INK_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e6e5e1"
OURS_COLOR = "#2a78d6"
BEST_COLOR = "#1baf7a"
VLLM_COLOR = "#e34948"
BASELINE_COLOR = "#b9b8b3"
STEP_TIME_ENGINES = (
    ("phase-01-naive-t4", "no KV cache (phase 1)", "#eb6834"),
    ("phase-02-kv-cache-t4", "KV cache (phase 2)", OURS_COLOR),
    ("phase-07-cuda-graphs-t4", "KV cache + CUDA graph (phase 7)", BEST_COLOR),
)
MIXED_ENGINES = (
    ("phase-03-static-batching-t4", "static batching"),
    ("phase-03-continuous-batching-t4", "continuous batching"),
    ("phase-04-paged-t4", "+ paged KV cache"),
    ("vllm-t4", "vLLM 0.31.0"),
)
SPECULATIVE_RUNS = (
    ("phase-06-ngram-t4", "n-gram guesses, Qwen3-0.6B"),
    ("phase-06-draft-t4", "draft model: 0.6B guesses for 1.7B"),
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


def throughput_at(directory: str, workload: str, users: int) -> float | None:
    levels = read_json(RESULTS_DIR / directory / "results" / workload / "metrics.json") or []
    for level in levels:
        if level["concurrency"] == users:
            return level["total_tokens_per_second"]
    return None


# one fixed color per engine, so an engine keeps its color on every line chart
LINE_ENGINES = (
    ("phase-01-naive-t4", "1: naive", "#2a78d6"),
    ("phase-02-kv-cache-t4", "2: KV cache", "#eb6834"),
    ("phase-03-static-batching-t4", "3: static batching", "#1baf7a"),
    ("phase-03-continuous-batching-t4", "3: continuous batching", "#eda100"),
    ("phase-04-paged-t4", "4: paged KV cache", "#e87ba4"),
    ("phase-05-prefix-chunked-t4", "5: prefix cache + chunked prefill", "#008300"),
    ("phase-07-cuda-graphs-t4", "7: + CUDA graphs", "#6250d6"),
    ("vllm-t4", "vLLM 0.31.0", VLLM_COLOR),
)
USER_LEVELS = (1, 4, 16, 64)
LABEL_GAP = 66  # tokens/s between stacked labels at the same user count


def spread_labels(points: list[tuple[float, str, str]]) -> list[tuple[float, float, str, str]]:
    """Points (y, text, color) at one x -> (y, label_y, text, color) with labels at least
    LABEL_GAP apart, so labels of lines that meet don't cover each other."""
    placed = []
    previous = None
    for value, label, color in sorted(points):
        label_y = value if previous is None else max(value, previous + LABEL_GAP)
        placed.append((value, label_y, label, color))
        previous = label_y
    return placed


def draw_throughput_chart() -> None:
    """Total tokens/s as users grow, one line per engine, every point labeled (users, tok/s)."""
    figure, axis = pyplot.subplots(figsize=(11, 7), facecolor=SURFACE_COLOR)
    points_by_users: dict[int, list[tuple[float, str, str]]] = {users: [] for users in USER_LEVELS}
    for directory, label, color in LINE_ENGINES:
        series = [(users, throughput_at(directory, "short_chat", users)) for users in USER_LEVELS]
        series = [(users, value) for users, value in series if value is not None]
        if not series:
            continue
        axis.plot(
            [users for users, _ in series],
            [value for _, value in series],
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=color,
            label=label,
            zorder=3,
        )
        for users, value in series:
            points_by_users[users].append((value, f"({users}, {value:,.0f})", color))
    for users, points in points_by_users.items():
        label_x = users * 1.13
        for value, label_y, text, color in spread_labels(points):
            if label_y != value:
                axis.plot([users, label_x], [value, label_y], color=color, linewidth=0.7, zorder=2)
            axis.text(
                label_x,
                label_y,
                text,
                va="center",
                fontsize=8.5,
                color=color,
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": SURFACE_COLOR,
                    "edgecolor": "none",
                },
            )
    style_axis(
        axis,
        "short_chat (64 tokens in, 128 out) on one T4: total throughput as users grow",
        "concurrent users",
        "total tokens / s, all users together (higher is better)",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(USER_LEVELS), [str(users) for users in USER_LEVELS])
    axis.set_xlim(0.8, 110)
    axis.set_ylim(0, 2150)
    axis.legend(frameon=False, labelcolor=INK_COLOR, fontsize=9, loc="upper left")
    save(figure, "progress-throughput-short-chat.png")


def draw_mixed_lengths_chart() -> None:
    """Grouped bars at 16 and 64 users for the batching phases and vLLM."""
    figure, axis = pyplot.subplots(figsize=(8, 4.4), facecolor=SURFACE_COLOR)
    colors = ("#b9b8b3", OURS_COLOR, BEST_COLOR, VLLM_COLOR)
    width = 0.2
    for index, ((directory, label), color) in enumerate(zip(MIXED_ENGINES, colors, strict=True)):
        values = [throughput_at(directory, "mixed_lengths", users) or 0 for users in (16, 64)]
        offsets = [group + (index - 1.5) * width for group in range(2)]
        bars = axis.bar(offsets, values, width * 0.95, color=color, label=label, zorder=2)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 15,
                f"{value:,.0f}",
                ha="center",
                fontsize=9,
                color=INK_COLOR,
            )
    style_axis(
        axis,
        "mixed_lengths (answers of 16-512 tokens) on one T4",
        "concurrent users",
        "total tokens / s",
    )
    axis.set_xticks([0, 1], ["16 users", "64 users"])
    axis.legend(frameon=False, labelcolor=INK_COLOR, fontsize=9)
    save(figure, "progress-throughput-mixed-lengths.png")


def draw_step_time_chart() -> None:
    figure, axis = pyplot.subplots(figsize=(8, 4.4), facecolor=SURFACE_COLOR)
    for directory, label, color in STEP_TIME_ENGINES:
        for path in sorted(
            (RESULTS_DIR / directory / "results").glob("step-time-vs-length-*.json")
        ):
            timings = read_json(path)
            lengths = [timing["sequence_length"] for timing in timings]
            step_ms = [timing["median_step_ms"] for timing in timings]
            axis.plot(
                lengths,
                step_ms,
                marker="o",
                markersize=6,
                linewidth=2,
                color=color,
                label=label,
                zorder=3,
            )
            for length, value in zip(lengths, step_ms, strict=True):
                if directory == "phase-01-naive-t4" and length < 512:
                    continue  # same as the KV-cache line here; its label would overlap
                below = directory == "phase-07-cuda-graphs-t4"
                axis.annotate(
                    f"{value:.0f} ms",
                    (length, value),
                    textcoords="offset points",
                    xytext=(0, -15 if below else 8),
                    ha="center",
                    fontsize=9,
                    color=color,
                )
    style_axis(
        axis,
        "Time for one decode step (1 user) on a T4, as the text gets longer",
        "tokens so far (prompt + answer)",
        "ms per step (lower is better)",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks([16, 64, 128, 256, 512, 1024], ["16", "64", "128", "256", "512", "1,024"])
    axis.set_ylim(0, 115)
    axis.annotate(
        "no KV cache = same as\nKV cache up to 256",
        (128, 34),
        textcoords="offset points",
        xytext=(0, 26),
        ha="center",
        fontsize=8,
        color="#eb6834",
    )
    axis.legend(frameon=False, labelcolor=INK_COLOR, loc="upper left")
    save(figure, "progress-step-time.png")


def draw_speculative_chart() -> None:
    """One panel per method: tokens/s without and with speculation, per prompt style."""
    runs = [
        (label, read_json(RESULTS_DIR / directory / "results" / "speculative-benchmark.json"))
        for directory, label in SPECULATIVE_RUNS
    ]
    runs = [(label, results) for label, results in runs if results]
    if not runs:
        return
    figure, axes = pyplot.subplots(
        1, len(runs), figsize=(11, 4.6), facecolor=SURFACE_COLOR, sharey=True
    )
    for axis, (label, results) in zip(axes, runs, strict=True):
        styles = [result["prompt_style"].replace("_", " ") for result in results]
        width = 0.36
        for group, result in enumerate(results):
            before, after = (
                result["baseline_tokens_per_second"],
                result["speculative_tokens_per_second"],
            )
            color = BEST_COLOR if after > before else VLLM_COLOR
            for offset, value, bar_color_ in (
                (-width / 2, before, BASELINE_COLOR),
                (width / 2, after, color),
            ):
                axis.bar(group + offset, value, width * 0.95, color=bar_color_, zorder=2)
                axis.text(
                    group + offset,
                    value + 1.5,
                    f"{value:.0f} tok/s",
                    ha="center",
                    fontsize=9,
                    color=INK_COLOR,
                )
            verdict = f"{result['speed_up']:.2f}× {'faster' if after > before else 'slower'}"
            axis.text(
                group,
                max(before, after) + 9,
                verdict,
                ha="center",
                fontsize=11,
                fontweight="bold",
                color=color,
            )
            axis.text(
                group,
                -9,
                f"guesses accepted: {result['acceptance_rate']:.0%}",
                ha="center",
                fontsize=8,
                color=MUTED_COLOR,
            )
        style_axis(axis, label, "", "tokens / s for one user (higher is better)")
        axis.set_xticks(range(len(styles)), styles)
        axis.set_ylim(-14, 105)
    handles = [
        Patch(color=BASELINE_COLOR, label="no speculation"),
        Patch(color=BEST_COLOR, label="with speculation: faster"),
        Patch(color=VLLM_COLOR, label="with speculation: slower"),
    ]
    axes[1].legend(handles=handles, frameon=False, labelcolor=INK_COLOR, loc="upper right")
    figure.suptitle(
        "Speculative decoding on a T4: guess a few tokens cheaply, check them in one pass",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK_COLOR,
    )
    save(figure, "progress-speculative.png")


# users that fit in the 4 GB KV budget (phase 4 write-up):
# one 2,048-slot region per user vs 16-slot blocks
USERS_FIT_CONTIGUOUS = 18
USERS_FIT_PAGED = 64
INTERFERENCE_TARGET = 2.0


def ttft_p50_ms(directory: str, workload: str, users: int) -> float:
    levels = read_json(RESULTS_DIR / directory / "results" / workload / "metrics.json") or []
    return next(level["ttft"]["p50_ms"] for level in levels if level["concurrency"] == users)


def report_numbers(directory: str, suite: str) -> tuple[float, float]:
    """The 'before -> after' pair a suite row reports, e.g. 'TTFT p50 141 -> 46 ms'."""
    report = (RESULTS_DIR / directory / "results" / "harness-report.md").read_text(encoding="utf-8")
    row = next(line for line in report.splitlines() if line.startswith(f"| {suite} "))
    before, after = re.search(r"([\d.]+) -> ([\d.]+)", row).groups()
    return float(before), float(after)


def draw_before_after_chart() -> None:
    """Four panels, one per scheduling / memory technique: the number it changed."""
    static_ttft = ttft_p50_ms("phase-03-static-batching-t4", "mixed_lengths", 16)
    continuous_ttft = ttft_p50_ms("phase-03-continuous-batching-t4", "mixed_lengths", 16)
    unique_ttft, shared_ttft = report_numbers("phase-05-prefix-chunked-t4", "prefix_reuse")
    gap_p4 = report_numbers("phase-04-paged-t4", "long_prompt_interference")
    gap_p5 = report_numbers("phase-05-prefix-chunked-t4", "long_prompt_interference")
    panels = (
        (
            "Continuous batching: time to first token\n(16 users, answers of 16-512 tokens)",
            [
                ("static\nbatching", static_ttft, BASELINE_COLOR),
                ("continuous\nbatching", continuous_ttft, BEST_COLOR),
            ],
            "ms (lower is better)",
            "{:,.0f} ms",
        ),
        (
            "Paged KV cache: users that fit\nin 4 GB of KV memory",
            [
                ("one fixed\nregion each", USERS_FIT_CONTIGUOUS, BASELINE_COLOR),
                ("16-token\nblocks", USERS_FIT_PAGED, BEST_COLOR),
            ],
            "users (higher is better)",
            "{:.0f} users",
        ),
        (
            "Prefix cache: time to first token\n(512 of 544 prompt tokens shared)",
            [
                ("prompt\ncomputed", unique_ttft, BASELINE_COLOR),
                ("prefix\ncache hit", shared_ttft, BEST_COLOR),
            ],
            "ms (lower is better)",
            "{:.0f} ms",
        ),
        (
            "Chunked prefill: how much other users\nslow down while long prompts arrive",
            [
                ("whole prompt\nin one step", gap_p4[1] / gap_p4[0], BASELINE_COLOR),
                ("prompt in\nchunks", gap_p5[1] / gap_p5[0], BEST_COLOR),
            ],
            "× slower token gaps (lower is better)",
            "{:.1f}×",
        ),
    )
    figure, axes = pyplot.subplots(2, 2, figsize=(11, 8.5), facecolor=SURFACE_COLOR)
    for axis, (title, bars, y_label, value_format) in zip(axes.flat, panels, strict=True):
        names = [name for name, _, _ in bars]
        values = [value for _, value, _ in bars]
        axis.bar(names, values, color=[color for _, _, color in bars], width=0.6, zorder=2)
        top = max(values)
        for index, value in enumerate(values):
            axis.text(
                index,
                value + top * 0.02,
                value_format.format(value),
                ha="center",
                fontsize=11,
                fontweight="bold",
                color=INK_COLOR,
            )
        style_axis(axis, title, "", y_label)
        axis.title.set_fontsize(11)
        axis.set_ylim(0, top * 1.18)
    axes[1][1].axhline(INTERFERENCE_TARGET, color=VLLM_COLOR, linewidth=1, linestyle="--", zorder=1)
    axes[1][1].text(
        0.5,
        INTERFERENCE_TARGET + 0.05,
        "target: under 2×",
        ha="center",
        fontsize=9,
        color=VLLM_COLOR,
    )
    figure.suptitle(
        "What each scheduling and memory technique changed, measured on one T4",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK_COLOR,
    )
    save(figure, "progress-before-after.png")


def main() -> None:
    draw_throughput_chart()
    draw_step_time_chart()
    draw_mixed_lengths_chart()
    draw_speculative_chart()
    draw_before_after_chart()


if __name__ == "__main__":
    main()
