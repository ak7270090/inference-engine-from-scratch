"""Request intake shared by every engine loop: a handle per request and its token stream."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from nanoengine.engine.interface import GenerationRequest, TokenEvent


class EngineOverloadedError(Exception):
    """Too many requests are already waiting."""


@dataclass
class RequestHandle:
    """The engine puts token events (or an error) here; the API reads them."""

    request: GenerationRequest
    token_events: asyncio.Queue[TokenEvent | BaseException] = field(default_factory=asyncio.Queue)
    is_cancelled: bool = False


async def stream_token_events(handle: RequestHandle) -> AsyncIterator[TokenEvent]:
    """Yield events until the last one; closing this early marks the request cancelled."""
    try:
        while True:
            event = await handle.token_events.get()
            if isinstance(event, BaseException):
                raise event
            yield event
            if event.is_finished:
                return
    finally:
        handle.is_cancelled = True
