"""Send one streaming completion and timestamp every chunk as it arrives."""

import json
import time
from dataclasses import dataclass, field

import aiohttp

from engineharness.constants import COMPLETIONS_PATH, SSE_DATA_PREFIX, SSE_DONE_MARKER, STATUS_OK
from engineharness.workloads import CompletionRequest


@dataclass
class RequestTiming:
    """One streamed request as the client saw it; times come from ``time.perf_counter``."""

    request_id: str
    sent_at: float
    finished_at: float = 0.0
    chunk_arrival_times: list[float] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)
    completion_tokens: int = 0
    saw_done_marker: bool = False
    error: str | None = None

    @property
    def text(self) -> str:
        return "".join(self.chunk_texts)

    @property
    def is_successful(self) -> bool:
        return self.error is None and bool(self.chunk_arrival_times)

    @property
    def ttft_seconds(self) -> float:
        return self.chunk_arrival_times[0] - self.sent_at

    @property
    def e2e_seconds(self) -> float:
        return self.finished_at - self.sent_at

    @property
    def inter_chunk_gaps_seconds(self) -> list[float]:
        times = self.chunk_arrival_times
        return [later - earlier for earlier, later in zip(times, times[1:], strict=False)]

    @property
    def decode_tokens_per_second(self) -> float | None:
        """Tokens per second after the first one; ``None`` when there is only one token."""
        decode_seconds = self.chunk_arrival_times[-1] - self.chunk_arrival_times[0]
        if self.completion_tokens < 2 or decode_seconds <= 0:
            return None
        return (self.completion_tokens - 1) / decode_seconds


async def stream_completion(
    session: aiohttp.ClientSession, request: CompletionRequest, model_id: str
) -> RequestTiming:
    """POST ``request`` with ``stream: true`` and record when each text chunk arrives."""
    timing = RequestTiming(request_id=request.request_id, sent_at=time.perf_counter())
    payload = request.build_payload(model_id, is_streaming=True)
    try:
        async with session.post(COMPLETIONS_PATH, json=payload) as response:
            if response.status != STATUS_OK:
                body = await response.text()
                timing.error = f"HTTP {response.status}: {body[:200]}"
            else:
                async for raw_line in response.content:
                    if _read_sse_line(raw_line.decode("utf-8"), timing):
                        break
    except (aiohttp.ClientError, TimeoutError) as error:
        timing.error = f"{type(error).__name__}: {error}"
    timing.finished_at = time.perf_counter()
    if not timing.completion_tokens:
        timing.completion_tokens = len(timing.chunk_arrival_times)
    return timing


def _read_sse_line(line: str, timing: RequestTiming) -> bool:
    """Apply one SSE line to ``timing``; returns ``True`` on the ``[DONE]`` marker."""
    line = line.strip()
    if not line.startswith(SSE_DATA_PREFIX):
        return False
    data = line[len(SSE_DATA_PREFIX) :].strip()
    if data == SSE_DONE_MARKER:
        timing.saw_done_marker = True
        return True
    chunk = json.loads(data)
    usage = chunk.get("usage")
    if usage:
        timing.completion_tokens = usage.get("completion_tokens", timing.completion_tokens)
    for choice in chunk.get("choices") or []:
        text = choice.get("text")
        if text:
            timing.chunk_arrival_times.append(time.perf_counter())
            timing.chunk_texts.append(text)
    return False
