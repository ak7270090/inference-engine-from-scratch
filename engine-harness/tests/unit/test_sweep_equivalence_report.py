import json

import aiohttp
import pytest
from stub_engine import STUB_MODEL_ID

from engineharness.__main__ import main, sweep_workload, wait_for_engine, write_reference
from engineharness.concurrency_sweep import run_closed_loop, run_concurrency_sweep
from engineharness.config import load_harness_settings
from engineharness.output_equivalence import (
    collect_greedy_outputs,
    compare_outputs,
    load_reference,
    save_reference,
)
from engineharness.report_writer import HarnessReport, SuiteResult, write_sweep_results
from engineharness.workloads import CompletionRequest

pytestmark = pytest.mark.anyio


def build_requests(count: int, max_tokens: int = 4) -> list[CompletionRequest]:
    return [CompletionRequest(f"r{i}", (i, i + 1), max_tokens) for i in range(count)]


async def test_more_users_finish_the_same_requests_sooner(stub_session) -> None:
    requests = build_requests(8)

    serial_timings, serial_wall = await run_closed_loop(stub_session, requests, 1, STUB_MODEL_ID)
    parallel_timings, parallel_wall = await run_closed_loop(
        stub_session, requests, 8, STUB_MODEL_ID
    )

    assert len(serial_timings) == len(parallel_timings) == 8
    assert parallel_wall < serial_wall / 3


async def test_sweep_returns_one_row_per_level_and_writes_results(stub_session, tmp_path) -> None:
    metrics = await run_concurrency_sweep(
        stub_session, build_requests(4), [1, 2], STUB_MODEL_ID, "w"
    )

    paths = write_sweep_results(tmp_path, metrics, "http://stub")

    assert [level.concurrency for level in metrics] == [1, 2]
    assert all(level.error_count == 0 for level in metrics)
    assert all(path.exists() for path in paths)
    rows = json.loads(paths[1].read_text(encoding="utf-8"))
    assert rows[1]["concurrency"] == 2
    assert "| 2 | 4 | 0 |" in paths[0].read_text(encoding="utf-8")


async def test_greedy_outputs_round_trip_through_the_reference_file(stub_session, tmp_path) -> None:
    outputs = await collect_greedy_outputs(stub_session, build_requests(3), STUB_MODEL_ID)
    reference_path = tmp_path / "ref" / "greedy.json"

    save_reference(reference_path, outputs, source="stub")

    assert load_reference(reference_path) == outputs
    assert compare_outputs(outputs, outputs).agreement == 1.0


async def test_collecting_outputs_fails_loudly_on_an_error(stub_session) -> None:
    with pytest.raises(RuntimeError, match="r0 failed"):
        await collect_greedy_outputs(stub_session, build_requests(1, max_tokens=0), STUB_MODEL_ID)


def test_compare_outputs_counts_mismatches_and_partial_prefixes() -> None:
    reference = {"a": "hello world", "b": "abcd", "c": "same"}
    actual = {"a": "hello there", "c": "same"}

    result = compare_outputs(reference, actual)

    assert result.identical_count == 1
    assert result.agreement == pytest.approx(1 / 3)
    assert result.mismatched_request_ids == ("a", "b")
    assert result.mean_common_prefix_ratio == pytest.approx((6 / 11 + 0 + 1) / 3)


def test_compare_outputs_of_nothing_is_zero_agreement() -> None:
    assert compare_outputs({}, {}).agreement == 0.0


def test_harness_report_renders_rows_with_outcomes(tmp_path) -> None:
    report = HarnessReport()
    report.record("t1", SuiteResult("latency", "short_chat", "TTFT p50 200 ms", "-", "ok"))
    report.set_outcome("t1", "PASS")
    report.set_outcome("unknown", "FAIL")

    path = report.write(tmp_path, "http://engine")

    text = path.read_text(encoding="utf-8")
    assert "| latency | short_chat | TTFT p50 200 ms | - | PASS | ok |" in text
    assert len(report.results) == 1


async def test_wait_command_reports_ready_stub(stub_settings) -> None:
    assert await wait_for_engine(stub_settings) == 0


async def test_reference_command_writes_the_reference_file(
    stub_settings, word_tokenizer, monkeypatch
) -> None:
    monkeypatch.setattr("engineharness.__main__.HubTokenizer", lambda _id: word_tokenizer)
    settings = stub_settings.model_copy(update={"equivalence_prompt_count": 2})

    assert await write_reference(settings) == 0
    assert len(load_reference(settings.reference_file)) == 2


async def test_sweep_command_writes_results_for_a_workload(
    stub_settings, word_tokenizer, monkeypatch
) -> None:
    monkeypatch.setattr("engineharness.__main__.HubTokenizer", lambda _id: word_tokenizer)
    settings = stub_settings.model_copy(update={"concurrency_levels": [2]})

    assert await sweep_workload(settings, "short_chat") == 0
    assert (settings.results_dir / "short_chat" / "concurrency-sweep.md").exists()


def test_cli_rejects_an_unknown_command() -> None:
    with pytest.raises(SystemExit):
        main(["dance"])


def test_settings_load_from_env_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / "phase.env"
    env_file.write_text(
        "HARNESS_TARGET_URL=http://x:1\nHARNESS_CONCURRENCY_LEVELS=[1, 16]\n"
        'HARNESS_EXPECTED_FAILURES=["concurrency"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HARNESS_ENV_FILE", str(env_file))

    settings = load_harness_settings()

    assert settings.target_url == "http://x:1"
    assert settings.concurrency_levels == [1, 16]
    assert settings.expected_failures == {"concurrency"}


async def test_wait_until_ready_gives_up_on_a_dead_url() -> None:
    from engineharness.client_factory import wait_until_ready

    async with aiohttp.ClientSession(base_url="http://127.0.0.1:9") as session:
        assert await wait_until_ready(session, timeout_seconds=0.1) is False
