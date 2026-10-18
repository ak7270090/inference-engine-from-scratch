import aiohttp
import pytest
from stub_engine import (
    STUB_FIRST_TOKEN_DELAY_SECONDS,
    STUB_INTER_TOKEN_DELAY_SECONDS,
    STUB_MODEL_ID,
    build_stub_token,
)

from engineharness.streaming_client import RequestTiming, _read_sse_line, stream_completion
from engineharness.workloads import CompletionRequest

pytestmark = pytest.mark.anyio


async def test_stream_records_every_chunk_and_usage(stub_session: aiohttp.ClientSession) -> None:
    request = CompletionRequest("r1", (1, 2, 3), max_tokens=5)

    timing = await stream_completion(stub_session, request, STUB_MODEL_ID)

    assert timing.is_successful
    assert timing.saw_done_marker
    assert timing.completion_tokens == 5
    assert len(timing.chunk_arrival_times) == 5
    assert timing.text == "".join(build_stub_token([1, 2, 3], position) for position in range(5))


async def test_stream_timings_match_the_engine_delays(stub_session: aiohttp.ClientSession) -> None:
    request = CompletionRequest("r1", (4, 5), max_tokens=6)

    timing = await stream_completion(stub_session, request, STUB_MODEL_ID)

    first_token_delay = STUB_FIRST_TOKEN_DELAY_SECONDS
    assert first_token_delay * 0.9 <= timing.ttft_seconds < first_token_delay + 0.15
    gaps = timing.inter_chunk_gaps_seconds
    assert len(gaps) == 5
    mean_gap = sum(gaps) / len(gaps)
    assert STUB_INTER_TOKEN_DELAY_SECONDS * 0.5 <= mean_gap < STUB_INTER_TOKEN_DELAY_SECONDS * 3
    assert timing.e2e_seconds >= timing.ttft_seconds
    assert timing.decode_tokens_per_second is not None


async def test_stream_reports_http_errors(stub_session: aiohttp.ClientSession) -> None:
    request = CompletionRequest("bad", (1,), max_tokens=0)

    timing = await stream_completion(stub_session, request, STUB_MODEL_ID)

    assert not timing.is_successful
    assert timing.error.startswith("HTTP 400")


async def test_stream_reports_connection_errors() -> None:
    async with aiohttp.ClientSession(base_url="http://127.0.0.1:9") as session:
        timing = await stream_completion(session, CompletionRequest("x", (1,), 3), "m")

    assert timing.error is not None
    assert not timing.is_successful


def test_a_token_with_empty_text_still_counts_as_a_chunk() -> None:
    timing = RequestTiming("special", sent_at=0.0)

    _read_sse_line('data: {"choices": [{"index": 0, "text": ""}]}', timing)
    _read_sse_line('data: {"choices": [{"index": 0, "finish_reason": "length"}]}', timing)

    assert len(timing.chunk_arrival_times) == 1
    assert timing.is_successful


def test_decode_speed_is_none_for_a_single_token() -> None:
    timing = RequestTiming("one", sent_at=0.0, chunk_arrival_times=[0.1], completion_tokens=1)

    assert timing.decode_tokens_per_second is None
