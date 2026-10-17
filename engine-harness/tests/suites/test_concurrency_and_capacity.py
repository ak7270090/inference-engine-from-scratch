"""Many users at once: throughput must grow with users, and a burst must not cause errors."""

import pytest

from engineharness.concurrency_sweep import run_concurrency_level, run_concurrency_sweep
from engineharness.constants import (
    SUITE_CAPACITY,
    SUITE_CONCURRENCY,
    WORKLOAD_BURST,
    WORKLOAD_SHORT_CHAT,
)
from engineharness.report_writer import SuiteResult, write_sweep_results
from engineharness.workloads import build_requests, get_workload_spec

pytestmark = pytest.mark.anyio


@pytest.mark.concurrency
async def test_total_throughput_grows_with_users(
    request, harness_settings, harness_report, engine_session, engine_tokenizer
) -> None:
    spec = get_workload_spec(harness_settings.workload_scale, WORKLOAD_SHORT_CHAT)
    requests = build_requests(spec, engine_tokenizer, harness_settings.seed)
    levels = harness_settings.concurrency_levels

    metrics = await run_concurrency_sweep(
        engine_session, requests, levels, harness_settings.model_id, WORKLOAD_SHORT_CHAT
    )
    write_sweep_results(harness_settings.results_dir, metrics, harness_settings.target_url)
    speedup = metrics[-1].total_tokens_per_second / max(metrics[0].total_tokens_per_second, 1e-9)

    harness_report.record(
        request.node.nodeid,
        SuiteResult(
            SUITE_CONCURRENCY,
            WORKLOAD_SHORT_CHAT,
            f"{metrics[0].total_tokens_per_second:.1f} -> "
            f"{metrics[-1].total_tokens_per_second:.1f} tok/s ({speedup:.1f}x)",
            f">= {harness_settings.concurrency_min_speedup:.1f}x at {levels[-1]} users",
            f"levels {levels}",
        ),
    )
    assert all(level.error_count == 0 for level in metrics)
    assert speedup >= harness_settings.concurrency_min_speedup


@pytest.mark.capacity
async def test_a_burst_of_requests_is_served_without_errors(
    request, harness_settings, harness_report, engine_session, engine_tokenizer
) -> None:
    spec = get_workload_spec(harness_settings.workload_scale, WORKLOAD_BURST)
    requests = build_requests(spec, engine_tokenizer, harness_settings.seed)

    metrics = await run_concurrency_level(
        engine_session, requests, len(requests), harness_settings.model_id, WORKLOAD_BURST
    )
    error_rate = metrics.error_count / metrics.request_count

    harness_report.record(
        request.node.nodeid,
        SuiteResult(
            SUITE_CAPACITY,
            WORKLOAD_BURST,
            f"{metrics.error_count}/{metrics.request_count} errors, "
            f"{metrics.total_tokens_per_second:.1f} tok/s",
            f"error rate <= {harness_settings.capacity_max_error_rate:.0%}",
            f"E2E p99 {metrics.end_to_end.p99_ms:.0f} ms",
        ),
    )
    assert error_rate <= harness_settings.capacity_max_error_rate
