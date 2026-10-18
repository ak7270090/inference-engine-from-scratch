"""Turn an engine's token events into an SSE stream or one full completion response."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from nanoengine.api.schemas import CompletionChoice, CompletionResponse, UsageInfo
from nanoengine.constants import SSE_DATA_PREFIX, SSE_DONE_MESSAGE, SSE_EVENT_END
from nanoengine.engine.interface import TokenEvent
from nanoengine.models.text_tokenizer import IncrementalDecoder, TextTokenizer


@dataclass(frozen=True)
class CompletionContext:
    completion_id: str
    created: int
    model: str
    prompt_token_count: int
    should_include_usage: bool


def build_usage(context: CompletionContext, completion_token_count: int) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=context.prompt_token_count,
        completion_tokens=completion_token_count,
        total_tokens=context.prompt_token_count + completion_token_count,
    )


def format_sse_event(response: CompletionResponse) -> str:
    return f"{SSE_DATA_PREFIX}{response.model_dump_json(exclude_none=True)}{SSE_EVENT_END}"


async def stream_completion_chunks(
    token_events: AsyncIterator[TokenEvent], tokenizer: TextTokenizer, context: CompletionContext
) -> AsyncIterator[str]:
    """One SSE chunk per token, an optional usage chunk, then ``[DONE]``."""
    decoder = IncrementalDecoder(tokenizer)
    completion_token_count = 0
    async for event in token_events:
        completion_token_count += 1
        text = decoder.add_token(event.token_id)
        if event.is_finished:
            text += decoder.flush()
        choice = CompletionChoice(text=text, finish_reason=event.finish_reason)
        yield format_sse_event(
            CompletionResponse(
                id=context.completion_id,
                created=context.created,
                model=context.model,
                choices=[choice],
            )
        )
    if context.should_include_usage:
        yield format_sse_event(
            CompletionResponse(
                id=context.completion_id,
                created=context.created,
                model=context.model,
                choices=[],
                usage=build_usage(context, completion_token_count),
            )
        )
    yield SSE_DONE_MESSAGE


async def collect_completion(
    token_events: AsyncIterator[TokenEvent], tokenizer: TextTokenizer, context: CompletionContext
) -> CompletionResponse:
    token_ids: list[int] = []
    finish_reason = None
    async for event in token_events:
        token_ids.append(event.token_id)
        finish_reason = event.finish_reason
    return CompletionResponse(
        id=context.completion_id,
        created=context.created,
        model=context.model,
        choices=[CompletionChoice(text=tokenizer.decode(token_ids), finish_reason=finish_reason)],
        usage=build_usage(context, len(token_ids)),
    )
