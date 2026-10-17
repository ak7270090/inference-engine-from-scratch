"""Prompt reuse and fairness: shared prompt starts should be cheap, long prompts should not freeze
other users' streams."""

import asyncio
from dataclasses import replace

import pytest

from engineharness.concurrency_sweep import run_concurrency_level
from engineharness.constants import (
    SUITE_LONG_PROMPT_INTERFERENCE,
    SUITE_PREFIX_REUSE,
    WORKLOAD_LONG_PROMPT,
    WORKLOAD_SHARED_PREFIX,
    WORKLOAD_SHORT_CHAT,
)
from engineharness.report_writer import SuiteResult
from engineharness.workloads import build_requests, get_workload_spec

pytestmark = pytest.mark.anyio

INTERFERENCE_USERS = 2


@pytest.mark.prefix_reuse
async def test_shared_prompt_start_cuts_time_to_first_token(
    request, harness_settings, harness_report, engine_session, engine_tokenizer
) -> None:
    shared_spec = get_workload_spec(harness_settings.workload_scale, WORKLOAD_SHARED_PREFIX)
    unshared_spec = replace(shared_spec, shared_prefix_tokens=0)
    shared = build_requests(shared_spec, engine_tokenizer, harness_settings.seed)
    unshared = build_requests(unshared_spec, engine_tokenizer, harness_settings.seed + 1)
    model_id = harness_settings.model_id

    await run_concurrency_level(engine_session, shared[:1], 1, model_id, WORKLOAD_SHARED_PREFIX)
    with_reuse = await run_concurrency_level(
        engine_session, shared[1:], 1, model_id, WORKLOAD_SHARED_PREFIX
    )
    without_reuse = await run_concurrency_level(
        engine_session, unshared[1:], 1, model_id, WORKLOAD_SHARED_PREFIX
    )
    reduction = 1 - with_reuse.ttft.p50_ms / max(without_reuse.ttft.p50_ms, 1e-9)

    harness_report.record(
        request.node.nodeid,
        SuiteResult(
            SUITE_PREFIX_REUSE,
            WORKLOAD_SHARED_PREFIX,
            f"TTFT p50 {without_reuse.ttft.p50_ms:.0f} -> {with_reuse.ttft.p50_ms:.0f} ms "
            f"({reduction:.0%} lower)",
            f">= {harness_settings.prefix_min_ttft_reduction:.0%} lower",
            f"{shared_spec.shared_prefix_tokens} shared of {shared_spec.prompt_tokens} tokens",
        ),
    )
    assert with_reuse.error_count == without_reuse.error_count == 0
    assert reduction >= harness_settings.prefix_min_ttft_reduction


@pytest.mark.long_prompt_interference
async def test_long_prompts_do_not_freeze_other_streams(
    request, harness_settings, harness_report, engine_session, engine_tokenizer
) -> None:
    scale = harness_settings.workload_scale
    chats = build_requests(
        get_workload_spec(scale, WORKLOAD_SHORT_CHAT), engine_tokenizer, harness_settings.seed
    )[: INTERFERENCE_USERS * 2]
    long_prompts = build_requests(
        get_workload_spec(scale, WORKLOAD_LONG_PROMPT), engine_tokenizer, harness_settings.seed
    )
    model_id = harness_settings.model_id

    alone = await run_concurrency_level(
        engine_session, chats, INTERFERENCE_USERS, model_id, WORKLOAD_SHORT_CHAT
    )
    disturbed, _ = await asyncio.gather(
        run_concurrency_level(
            engine_session, chats, INTERFERENCE_USERS, model_id, WORKLOAD_SHORT_CHAT
        ),
        run_concurrency_level(engine_session, long_prompts, 1, model_id, WORKLOAD_LONG_PROMPT),
    )
    ratio = disturbed.inter_token.p99_ms / max(alone.inter_token.p99_ms, 1e-9)

    harness_report.record(
        request.node.nodeid,
        SuiteResult(
            SUITE_LONG_PROMPT_INTERFERENCE,
            WORKLOAD_SHORT_CHAT,
            f"ITL p99 {alone.inter_token.p99_ms:.0f} -> {disturbed.inter_token.p99_ms:.0f} ms "
            f"({ratio:.1f}x)",
            f"<= {harness_settings.interference_max_itl_ratio:.1f}x",
            f"{len(long_prompts)} long prompts arriving",
        ),
    )
    assert alone.error_count == disturbed.error_count == 0
    assert ratio <= harness_settings.interference_max_itl_ratio
