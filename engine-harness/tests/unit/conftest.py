import zlib
from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp.test_utils import TestServer
from stub_engine import create_stub_engine

from engineharness.config import HarnessSettings


class WordTokenizer:
    """One token per word, IDs from a stable hash: enough to test prompt building offline."""

    def encode(self, text: str) -> list[int]:
        return [zlib.crc32(word.encode()) % 50_000 for word in text.split()]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def word_tokenizer() -> WordTokenizer:
    return WordTokenizer()


@pytest.fixture
async def stub_url() -> AsyncIterator[str]:
    server = TestServer(create_stub_engine())
    await server.start_server()
    yield str(server.make_url("")).rstrip("/")
    await server.close()


@pytest.fixture
async def stub_session(stub_url: str) -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession(base_url=stub_url) as session:
        yield session


@pytest.fixture
def stub_settings(stub_url: str, tmp_path) -> HarnessSettings:
    return HarnessSettings(
        target_url=stub_url,
        model_id="stub-model",
        results_dir=tmp_path / "results",
        reference_file=tmp_path / "reference.json",
        server_ready_timeout_seconds=2.0,
    )
