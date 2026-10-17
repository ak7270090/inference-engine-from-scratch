"""Turn per-request timings into TTFT, inter-token latency, end-to-end time and throughput."""

import statistics
from dataclasses import asdict, dataclass

from engineharness.streaming_client import RequestTiming

MILLISECONDS_PER_SECOND = 1000.0


def calculate_percentile(values: list[float], percent: float) -> float:
    """Linear-interpolation percentile (same method as ``numpy.percentile``'s default)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


@dataclass(frozen=True)
class LatencySummary:
    """Percentiles of one latency, in milliseconds."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float

    @classmethod
    def from_seconds(cls, values_seconds: list[float]) -> "LatencySummary":
        values_ms = [value * MILLISECONDS_PER_SECOND for value in values_seconds]
        return cls(
            p50_ms=calculate_percentile(values_ms, 50),
            p95_ms=calculate_percentile(values_ms, 95),
            p99_ms=calculate_percentile(values_ms, 99),
            mean_ms=statistics.fmean(values_ms) if values_ms else 0.0,
        )


@dataclass(frozen=True)
class WorkloadMetrics:
    """Everything the harness reports for one workload at one concurrency level."""

    workload: str
    concurrency: int
    request_count: int
    error_count: int
    total_completion_tokens: int
    wall_seconds: float
    total_tokens_per_second: float
    per_user_tokens_per_second: float
    ttft: LatencySummary
    inter_token: LatencySummary
    end_to_end: LatencySummary

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_timings(
    timings: list[RequestTiming], wall_seconds: float, workload: str, concurrency: int
) -> WorkloadMetrics:
    """Aggregate successful requests; failed ones only count towards ``error_count``."""
    successful = [timing for timing in timings if timing.is_successful]
    total_tokens = sum(timing.completion_tokens for timing in successful)
    decode_speeds = [
        speed for timing in successful if (speed := timing.decode_tokens_per_second) is not None
    ]
    inter_token_gaps = [gap for timing in successful for gap in timing.inter_chunk_gaps_seconds]
    return WorkloadMetrics(
        workload=workload,
        concurrency=concurrency,
        request_count=len(timings),
        error_count=len(timings) - len(successful),
        total_completion_tokens=total_tokens,
        wall_seconds=wall_seconds,
        total_tokens_per_second=total_tokens / wall_seconds if wall_seconds > 0 else 0.0,
        per_user_tokens_per_second=statistics.median(decode_speeds) if decode_speeds else 0.0,
        ttft=LatencySummary.from_seconds([timing.ttft_seconds for timing in successful]),
        inter_token=LatencySummary.from_seconds(inter_token_gaps),
        end_to_end=LatencySummary.from_seconds([timing.e2e_seconds for timing in successful]),
    )
