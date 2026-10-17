"""Streaming format and single-user latency: the basics every phase must get right."""

import pytest

from engineharness.concurrency_sweep import run_concurrency_level
from engineharness.constants import SUITE_LATENCY, SUITE_STREAMING, WORKLOAD_SHORT_CHAT
from engineharness.report_writer import SuiteResult, write_sweep_results
from engineharness.streaming_client import stream_completion
from engineharness.workloads import build_requests, get_workload_spec

pytestmark = pytest.mark.anyio

LATENCY_SAMPLE_REQUESTS = 4


@pytest.mark.streaming
async def test_tokens_stream_one_chunk_at_a_time_and_end_with_done(
    request, harness_settings, harness_report, engine_session, engine_tokenizer
) -> None:
    spec = get_workload_spec(harness_settings.workload_scale, WORKLOAD_SHORT_CHAT)
    completion_request = build_requests(spec, engine_tokenizer, harness_settings.seed)[0]

    timing = await stream_completion(engine_session, completion_request, harness_settings.model_id)

    harness_report.record(
        request.node.nodeid,
        SuiteResult(
            SUITE_STREAMING,
            WORKLOAD_SHORT_CHAT,
            f"{len(timing.chunk_arrival_times)} chunks, {timing.completion_tokens} tokens",
            f"{completion_request.max_tokens} tokens, [DONE]",
            timing.error or "ok",
        ),
    )
    assert timing.error is None
    assert timing.saw_done_marker
    assert timing.completion_tokens == completion_request.max_tokens
    assert len(timing.chunk_arrival_times) >= 2


@pytest.mark.latency
async def test_single_user_latency_is_measured_without_errors(
    request, harness_settings, harness_report, engine_session, engine_tokenizer
) -> None:
    spec = get_workload_spec(harness_settings.workload_scale, WORKLOAD_SHORT_CHAT)
    requests = build_requests(spec, engine_tokenizer, harness_settings.seed)[
        :LATENCY_SAMPLE_REQUESTS
    ]

    metrics = await run_concurrency_level(
        engine_session, requests, 1, harness_settings.model_id, WORKLOAD_SHORT_CHAT
    )
    write_sweep_results(
        harness_settings.results_dir / "latency", [metrics], harness_settings.target_url
    )

    harness_report.record(
        request.node.nodeid,
        SuiteResult(
            SUITE_LATENCY,
            WORKLOAD_SHORT_CHAT,
            f"TTFT p50 {metrics.ttft.p50_ms:.0f} ms, ITL p50 {metrics.inter_token.p50_ms:.1f} ms, "
            f"{metrics.per_user_tokens_per_second:.1f} tok/s",
            "no errors",
            f"{metrics.request_count} requests at 1 user",
        ),
    )
    assert metrics.error_count == 0
    assert metrics.ttft.p50_ms > 0
