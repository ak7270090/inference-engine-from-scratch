"""FastAPI app: load the engine at startup, start its loop, stop it on shutdown."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI

from nanoengine.api.routes import router
from nanoengine.config import EngineSettings, load_engine_settings
from nanoengine.constants import ENGINE_NAME
from nanoengine.engine.runtime import EngineRuntime, create_engine_runtime


@asynccontextmanager
async def run_engine_lifespan(
    app: FastAPI, settings: EngineSettings, runtime: EngineRuntime | None
) -> AsyncIterator[None]:
    runtime = runtime or await asyncio.to_thread(create_engine_runtime, settings)
    app.state.engine_runtime = runtime
    await runtime.engine.start()
    try:
        yield
    finally:
        await runtime.engine.stop()


def create_app(
    settings: EngineSettings | None = None, runtime: EngineRuntime | None = None
) -> FastAPI:
    """Pass ``runtime`` to serve an already-built engine (tests); otherwise one is loaded."""
    settings = settings or load_engine_settings()
    app = FastAPI(
        title=ENGINE_NAME,
        lifespan=partial(run_engine_lifespan, settings=settings, runtime=runtime),
    )
    app.include_router(router)
    return app
