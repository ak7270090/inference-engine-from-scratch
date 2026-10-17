import pytest

from engineharness.latency_metrics import LatencySummary, calculate_percentile, summarize_timings
from engineharness.streaming_client import RequestTiming


def build_timing(request_id: str, sent_at: float, arrivals: list[float]) -> RequestTiming:
    return RequestTiming(
        request_id=request_id,
        sent_at=sent_at,
        finished_at=arrivals[-1] + 0.01,
        chunk_arrival_times=arrivals,
        chunk_texts=["x"] * len(arrivals),
        completion_tokens=len(arrivals),
    )


@pytest.mark.parametrize(
    ("values", "percent", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], 50, 2.5),
        ([1.0, 2.0, 3.0, 4.0], 100, 4.0),
        ([1.0, 2.0, 3.0, 4.0], 0, 1.0),
        ([10.0], 99, 10.0),
        ([], 50, 0.0),
    ],
)
def test_percentile_uses_linear_interpolation(values, percent, expected) -> None:
    assert calculate_percentile(values, percent) == pytest.approx(expected)


def test_latency_summary_converts_seconds_to_milliseconds() -> None:
    summary = LatencySummary.from_seconds([0.1, 0.2, 0.3])

    assert summary.p50_ms == pytest.approx(200.0)
    assert summary.mean_ms == pytest.approx(200.0)


def test_summarize_timings_computes_ttft_itl_and_throughput() -> None:
    timings = [
        build_timing("a", 0.0, [0.2, 0.3, 0.4, 0.5]),
        build_timing("b", 0.0, [0.4, 0.6, 0.8, 1.0]),
    ]

    metrics = summarize_timings(timings, wall_seconds=2.0, workload="short_chat", concurrency=2)

    assert metrics.request_count == 2
    assert metrics.error_count == 0
    assert metrics.total_completion_tokens == 8
    assert metrics.total_tokens_per_second == pytest.approx(4.0)
    assert metrics.ttft.p50_ms == pytest.approx(300.0)
    assert metrics.inter_token.p50_ms == pytest.approx(150.0)
    assert metrics.per_user_tokens_per_second == pytest.approx((10.0 + 5.0) / 2)


def test_failed_requests_only_count_as_errors() -> None:
    failed = RequestTiming("bad", sent_at=0.0, error="HTTP 500")
    timings = [build_timing("ok", 0.0, [0.1, 0.2]), failed]

    metrics = summarize_timings(timings, wall_seconds=1.0, workload="w", concurrency=1)

    assert metrics.error_count == 1
    assert metrics.total_completion_tokens == 2
    assert metrics.to_dict()["ttft"]["p50_ms"] == pytest.approx(100.0)


def test_zero_wall_time_gives_zero_throughput() -> None:
    metrics = summarize_timings([], wall_seconds=0.0, workload="w", concurrency=1)

    assert metrics.total_tokens_per_second == 0.0
    assert metrics.per_user_tokens_per_second == 0.0
