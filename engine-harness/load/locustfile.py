"""Open-ended load test: ``locust -f load/locustfile.py --host http://localhost:8000``.

Each simulated user streams short-chat completions back to back; Locust reports time to first
token as the request time and the full stream as a separate ``stream`` entry.
"""

import random
import time

from locust import HttpUser, between, events, task

from engineharness.config import load_harness_settings
from engineharness.constants import (
    COMPLETIONS_PATH,
    SSE_DATA_PREFIX,
    SSE_DONE_MARKER,
    WORKLOAD_SHORT_CHAT,
)
from engineharness.workloads import HubTokenizer, build_requests, get_workload_spec

HARNESS_SETTINGS = load_harness_settings()
SHORT_CHAT_REQUESTS = build_requests(
    get_workload_spec(HARNESS_SETTINGS.workload_scale, WORKLOAD_SHORT_CHAT),
    HubTokenizer(HARNESS_SETTINGS.tokenizer_id),
    HARNESS_SETTINGS.seed,
)


class StreamingChatUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def stream_short_chat(self) -> None:
        completion_request = random.choice(SHORT_CHAT_REQUESTS)
        payload = completion_request.build_payload(HARNESS_SETTINGS.model_id, is_streaming=True)
        started_at = time.perf_counter()
        first_token_at = None
        with self.client.post(
            COMPLETIONS_PATH, json=payload, stream=True, catch_response=True, name="first_token"
        ) as response:
            for raw_line in response.iter_lines():
                line = raw_line.decode().strip()
                if not line.startswith(SSE_DATA_PREFIX):
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                    response.success()
                if line.removeprefix(SSE_DATA_PREFIX).strip() == SSE_DONE_MARKER:
                    break
            if first_token_at is None:
                response.failure("stream ended before the first token")
                return
        events.request.fire(
            request_type="SSE",
            name="stream",
            response_time=(time.perf_counter() - started_at) * 1000,
            response_length=0,
            exception=None,
            context={},
        )
