"""``python -m nanoengine``: serve the configured model with uvicorn."""

import logging

import uvicorn

from nanoengine.app import create_app
from nanoengine.config import load_engine_settings


def main() -> None:
    settings = load_engine_settings()
    logging.basicConfig(level=settings.log_level.upper())
    uvicorn.run(
        create_app(settings), host=settings.host, port=settings.port, log_level=settings.log_level
    )


if __name__ == "__main__":
    main()
