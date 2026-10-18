import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nanoengine.config import EngineSettings, load_engine_settings
from nanoengine.constants import ENV_FILE_VARIABLE
from nanoengine.engine.engine_loop import EngineOverloadedError
from tests.conftest import TINY_MAX_MODEL_LEN, TINY_MODEL_NAME


def build_body(**overrides) -> dict:
    return {
        "model": TINY_MODEL_NAME,
        "prompt": [1, 2, 3],
        "max_tokens": 4,
        "temperature": 0,
        **overrides,
    }


def read_sse_events(response) -> list[str]:
    return [
        line.removeprefix("data: ") for line in response.iter_lines() if line.startswith("data: ")
    ]


def test_health_and_models_endpoints(api_client: TestClient) -> None:
    assert api_client.get("/healthz").json() == {"status": "ok"}
    models = api_client.get("/v1/models").json()
    assert models["data"][0]["id"] == TINY_MODEL_NAME
    assert models["data"][0]["max_model_len"] == TINY_MAX_MODEL_LEN


def test_completion_returns_text_and_usage(api_client: TestClient) -> None:
    response = api_client.post("/v1/completions", json=build_body(ignore_eos=True))

    body = response.json()
    assert response.status_code == 200
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


def test_text_prompt_is_tokenized(api_client: TestClient) -> None:
    response = api_client.post("/v1/completions", json=build_body(prompt="w1 w2"))

    assert response.json()["usage"]["prompt_tokens"] == 2


def test_greedy_streaming_matches_non_streaming(api_client: TestClient) -> None:
    full_text = api_client.post("/v1/completions", json=build_body(ignore_eos=True)).json()[
        "choices"
    ][0]["text"]
    stream_body = build_body(ignore_eos=True, stream=True, stream_options={"include_usage": True})

    with api_client.stream("POST", "/v1/completions", json=stream_body) as response:
        events = read_sse_events(response)

    chunks = [json.loads(event) for event in events[:-1]]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert events[-1] == "[DONE]"
    assert len(chunks) == 5
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks[:4]).strip() == full_text.strip()
    assert chunks[3]["choices"][0]["finish_reason"] == "length"
    assert chunks[4]["choices"] == []
    assert chunks[4]["usage"]["completion_tokens"] == 4


def test_unknown_model_is_not_found(api_client: TestClient) -> None:
    response = api_client.post("/v1/completions", json=build_body(model="other"))

    assert response.status_code == 404


def test_empty_and_too_long_prompts_are_rejected(api_client: TestClient) -> None:
    assert api_client.post("/v1/completions", json=build_body(prompt=[])).status_code == 400
    too_long = build_body(max_tokens=TINY_MAX_MODEL_LEN)
    assert api_client.post("/v1/completions", json=too_long).status_code == 400


def test_invalid_sampling_values_are_rejected(api_client: TestClient) -> None:
    assert api_client.post("/v1/completions", json=build_body(max_tokens=0)).status_code == 422
    assert api_client.post("/v1/completions", json=build_body(top_p=0)).status_code == 422


def test_metrics_count_generated_tokens(api_client: TestClient) -> None:
    api_client.post("/v1/completions", json=build_body(ignore_eos=True))

    metrics_text = api_client.get("/metrics").text

    assert "nanoengine_generated_tokens_total 4.0" in metrics_text
    assert 'nanoengine_finished_requests_total{finish_reason="length"} 1.0' in metrics_text
    assert "nanoengine_step_latency_seconds_count 4.0" in metrics_text


def reject_request(request) -> None:
    raise EngineOverloadedError("queue full")


def test_full_queue_returns_service_unavailable(
    api_client: TestClient, tiny_runtime, monkeypatch
) -> None:
    monkeypatch.setattr(tiny_runtime.engine, "submit", reject_request)

    response = api_client.post("/v1/completions", json=build_body())

    assert response.status_code == 503


def test_settings_read_the_env_file(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / "engine.env"
    env_file.write_text("NANOENGINE_DEVICE=cuda\nNANOENGINE_DTYPE=float16\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_VARIABLE, str(env_file))

    settings = load_engine_settings()

    assert (settings.device, settings.dtype) == ("cuda", "float16")
    assert EngineSettings(served_model_name="alias").public_model_name == "alias"


def test_unknown_engine_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EngineSettings(kv_cache="does-not-exist")
