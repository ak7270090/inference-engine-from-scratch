"""HTTP sessions for talking to the engine under test, and a readiness check.

The harness uses aiohttp: with dozens of streams in flight from one process it keeps up far
better than httpx, and a slow client would blur every per-token timing.
"""

import asyncio
import time

import aiohttp

from engineharness.config import HarnessSettings
from engineharness.constants import MODELS_PATH, STATUS_OK

READY_POLL_INTERVAL_SECONDS = 2.0


def create_http_session(settings: HarnessSettings) -> aiohttp.ClientSession:
    """Create a session whose connector keeps up to ``connection_pool_size`` connections."""
    return aiohttp.ClientSession(
        base_url=settings.target_url,
        connector=aiohttp.TCPConnector(limit=settings.connection_pool_size),
        timeout=aiohttp.ClientTimeout(total=settings.request_timeout_seconds),
    )


async def wait_until_ready(session: aiohttp.ClientSession, timeout_seconds: float) -> bool:
    """Poll ``/v1/models`` until the engine answers; ``False`` if it never does in time."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            async with session.get(MODELS_PATH) as response:
                if response.status == STATUS_OK:
                    return True
        except (aiohttp.ClientError, TimeoutError):
            pass
        await asyncio.sleep(READY_POLL_INTERVAL_SECONDS)
    return False
