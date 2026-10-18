"""The standard API: ``/v1/completions``, ``/v1/models``, ``/healthz`` and ``/metrics``."""

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from nanoengine.api.completion_responses import (
    CompletionContext,
    collect_completion,
    stream_completion_chunks,
)
from nanoengine.api.schemas import CompletionRequest, CompletionResponse, ModelCard, ModelList
from nanoengine.constants import (
    COMPLETION_ID_PREFIX,
    COMPLETIONS_PATH,
    HEALTH_PATH,
    METRICS_PATH,
    MODELS_PATH,
    RETRY_AFTER_SECONDS,
    SSE_MEDIA_TYPE,
)
from nanoengine.engine.interface import GenerationRequest, SamplingParams
from nanoengine.engine.runtime import EngineRuntime
from nanoengine.engine.scheduling.request_queue import EngineOverloadedError

router = APIRouter()


def get_engine_runtime(request: Request) -> EngineRuntime:
    return request.app.state.engine_runtime


RuntimeDependency = Annotated[EngineRuntime, Depends(get_engine_runtime)]


@router.get(HEALTH_PATH)
async def check_health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(MODELS_PATH)
async def list_models(runtime: RuntimeDependency) -> ModelList:
    return ModelList(
        data=[ModelCard(id=runtime.served_model_name, max_model_len=runtime.max_model_len)]
    )


@router.get(METRICS_PATH)
async def export_metrics(runtime: RuntimeDependency) -> Response:
    return Response(runtime.metrics.render(), media_type=CONTENT_TYPE_LATEST)


@router.post(COMPLETIONS_PATH, response_model=None)
async def create_completion(
    body: CompletionRequest, runtime: RuntimeDependency
) -> CompletionResponse | StreamingResponse:
    if body.model != runtime.served_model_name:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"model '{body.model}' is not served here")
    prompt_token_ids = build_prompt_token_ids(body, runtime)
    generation_request = GenerationRequest(
        request_id=f"{COMPLETION_ID_PREFIX}{uuid.uuid4().hex}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            max_new_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            seed=body.seed,
            ignore_eos=body.ignore_eos,
        ),
    )
    try:
        token_events = runtime.engine.submit(generation_request)
    except EngineOverloadedError as error:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            str(error),
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        ) from error
    context = CompletionContext(
        completion_id=generation_request.request_id,
        created=int(time.time()),
        model=runtime.served_model_name,
        prompt_token_count=len(prompt_token_ids),
        should_include_usage=body.should_include_usage,
    )
    if body.stream:
        return StreamingResponse(
            stream_completion_chunks(token_events, runtime.tokenizer, context),
            media_type=SSE_MEDIA_TYPE,
        )
    return await collect_completion(token_events, runtime.tokenizer, context)


def build_prompt_token_ids(body: CompletionRequest, runtime: EngineRuntime) -> list[int]:
    """Accept text or token IDs; reject empty prompts and requests longer than the model allows."""
    prompt_token_ids = (
        runtime.tokenizer.encode(body.prompt) if isinstance(body.prompt, str) else body.prompt
    )
    if not prompt_token_ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "prompt is empty")
    total_tokens = len(prompt_token_ids) + body.max_tokens
    if total_tokens > runtime.max_model_len:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"prompt + max_tokens = {total_tokens} exceeds max_model_len {runtime.max_model_len}",
        )
    return prompt_token_ids
