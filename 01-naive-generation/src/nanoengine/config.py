"""Engine settings from ``NANOENGINE_*`` variables or the ``NANOENGINE_ENV_FILE`` file."""

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from nanoengine.constants import DEFAULT_MODEL_ID, ENV_FILE_VARIABLE, ENV_PREFIX


class EngineSettings(BaseSettings):
    """Which model to serve, on which device and dtype, and how many requests may wait."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX, extra="ignore", protected_namespaces=()
    )

    model_id: str = DEFAULT_MODEL_ID
    served_model_name: str | None = None
    model_cache_dir: Path | None = None
    device: str = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    max_model_len: int = 4096
    max_waiting_requests: int = 256

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    @property
    def public_model_name(self) -> str:
        return self.served_model_name or self.model_id


def load_engine_settings() -> EngineSettings:
    """Read settings; real environment variables override the optional ``NANOENGINE_ENV_FILE``."""
    env_file = os.environ.get(ENV_FILE_VARIABLE)
    return EngineSettings(_env_file=env_file) if env_file else EngineSettings()
