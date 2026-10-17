"""Write suite outcomes and concurrency sweeps into ``results/`` as Markdown, JSON and a chart."""

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from engineharness.constants import (
    METRICS_JSON_FILE_NAME,
    OUTCOME_SKIPPED,
    REPORT_FILE_NAME,
    SWEEP_CHART_FILE_NAME,
    SWEEP_REPORT_FILE_NAME,
)
from engineharness.latency_metrics import WorkloadMetrics


@dataclass(frozen=True)
class SuiteResult:
    """What one suite measured; ``outcome`` is filled in once pytest knows it."""

    suite: str
    workload: str
    measured: str
    threshold: str
    detail: str
    outcome: str = OUTCOME_SKIPPED


class HarnessReport:
    """Rows recorded by suites during a run, keyed by pytest node ID."""

    def __init__(self) -> None:
        self._results: dict[str, SuiteResult] = {}

    def record(self, node_id: str, result: SuiteResult) -> None:
        self._results[node_id] = result

    def set_outcome(self, node_id: str, outcome: str) -> None:
        if node_id in self._results:
            self._results[node_id] = replace(self._results[node_id], outcome=outcome)

    @property
    def results(self) -> list[SuiteResult]:
        return list(self._results.values())

    def render_markdown(self, target_url: str) -> str:
        lines = [
            f"# Harness report: {target_url}",
            "",
            f"Generated {_utc_now()}.",
            "",
            "| Suite | Workload | Measured | Threshold | Outcome | Detail |",
            "|---|---|---|---|---|---|",
        ]
        lines.extend(
            f"| {result.suite} | {result.workload} | {result.measured} | {result.threshold} "
            f"| {result.outcome} | {result.detail} |"
            for result in self.results
        )
        return "\n".join(lines) + "\n"

    def write(self, results_dir: Path, target_url: str) -> Path:
        results_dir.mkdir(parents=True, exist_ok=True)
        report_path = results_dir / REPORT_FILE_NAME
        report_path.write_text(self.render_markdown(target_url), encoding="utf-8")
        return report_path


def render_sweep_markdown(metrics: list[WorkloadMetrics], target_url: str) -> str:
    """One row per concurrency level: throughput, per-user speed and latency percentiles."""
    lines = [
        f"# Concurrency sweep: {metrics[0].workload if metrics else '-'} on {target_url}",
        "",
        f"Generated {_utc_now()}. Latencies in ms; ITL is measured between streamed chunks.",
        "",
        "| Users | Requests | Errors | Total tok/s | Per-user tok/s | TTFT p50 | TTFT p99 "
        "| ITL p50 | ITL p99 | E2E p50 | E2E p99 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {level.concurrency} | {level.request_count} | {level.error_count} "
        f"| {level.total_tokens_per_second:.1f} | {level.per_user_tokens_per_second:.1f} "
        f"| {level.ttft.p50_ms:.0f} | {level.ttft.p99_ms:.0f} | {level.inter_token.p50_ms:.1f} "
        f"| {level.inter_token.p99_ms:.1f} | {level.end_to_end.p50_ms:.0f} "
        f"| {level.end_to_end.p99_ms:.0f} |"
        for level in metrics
    )
    return "\n".join(lines) + "\n"


def write_sweep_results(
    results_dir: Path, metrics: list[WorkloadMetrics], target_url: str
) -> list[Path]:
    """Markdown table, raw JSON and a throughput-vs-latency chart for one sweep."""
    workload_dir = results_dir / (metrics[0].workload if metrics else "empty")
    workload_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = workload_dir / SWEEP_REPORT_FILE_NAME
    markdown_path.write_text(render_sweep_markdown(metrics, target_url), encoding="utf-8")
    json_path = workload_dir / METRICS_JSON_FILE_NAME
    json_path.write_text(
        json.dumps([level.to_dict() for level in metrics], indent=2) + "\n", encoding="utf-8"
    )
    chart_path = workload_dir / SWEEP_CHART_FILE_NAME
    draw_sweep_chart(metrics, chart_path)
    return [markdown_path, json_path, chart_path]


def draw_sweep_chart(metrics: list[WorkloadMetrics], chart_path: Path) -> None:
    """Total tokens/s against inter-token latency p50, one point per concurrency level."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axis = pyplot.subplots(figsize=(6, 4))
    throughputs = [level.total_tokens_per_second for level in metrics]
    latencies = [level.inter_token.p50_ms for level in metrics]
    axis.plot(throughputs, latencies, marker="o")
    for level, throughput, latency in zip(metrics, throughputs, latencies, strict=True):
        axis.annotate(
            f"{level.concurrency} users",
            (throughput, latency),
            textcoords="offset points",
            xytext=(5, 5),
        )
    axis.set_xlabel("total tokens / s")
    axis.set_ylabel("inter-token latency p50 (ms)")
    axis.set_title(metrics[0].workload if metrics else "")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=120)
    pyplot.close(figure)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
