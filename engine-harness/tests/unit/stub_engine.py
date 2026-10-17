"""A tiny in-process OpenAI-style engine with fixed delays, for testing the harness itself."""

import asyncio
import json

from aiohttp import web

STUB_MODEL_ID = "stub-model"
STUB_FIRST_TOKEN_DELAY_SECONDS = 0.05
STUB_INTER_TOKEN_DELAY_SECONDS = 0.03


def build_stub_token(prompt: list[int], position: int) -> str:
    """Deterministic output text, so the same prompt always gives the same completion."""
    return f" t{(sum(prompt) + position) % 97}"


async def list_models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": STUB_MODEL_ID}]})


async def create_completion(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    prompt = body["prompt"]
    if body.get("max_tokens", 0) <= 0:
        return web.json_response({"error": "max_tokens must be positive"}, status=400)
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    await asyncio.sleep(STUB_FIRST_TOKEN_DELAY_SECONDS)
    for position in range(body["max_tokens"]):
        if position:
            await asyncio.sleep(STUB_INTER_TOKEN_DELAY_SECONDS)
        chunk = {"choices": [{"index": 0, "text": build_stub_token(prompt, position)}]}
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
    usage = {"choices": [], "usage": {"completion_tokens": body["max_tokens"]}}
    await response.write(f"data: {json.dumps(usage)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    return response


def create_stub_engine() -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/completions", create_completion)
    return app
