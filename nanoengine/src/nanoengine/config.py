"""Engine settings from ``NANOENGINE_*`` variables or the ``NANOENGINE_ENV_FILE`` file."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from nanoengine.constants import (
    DEFAULT_MODEL_ID,
    ENV_FILE_VARIABLE,
    ENV_PREFIX,
    KV_CACHE_DTYPE_INT8,
    KV_CACHE_NONE,
    KV_CACHE_PAGED,
    SCHEDULER_CONTINUOUS,
    SCHEDULER_SERIAL,
    SPECULATIVE_NONE,
)


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

    # one value per optimization; each phase adds an option, older options stay as baselines
    kv_cache: Literal["none", "contiguous", "paged"] = "none"
    scheduler: Literal["serial", "static", "continuous"] = "serial"

    # batching and KV memory (phases 3-4)
    max_num_seqs: int = Field(default=64, ge=1)
    kv_cache_memory_mb: int = Field(default=1024, ge=1)
    kv_block_size: int = Field(default=16, ge=1)
    max_prefill_tokens: int = Field(default=4096, ge=1)
    batch_fill_timeout_ms: int = Field(default=50, ge=0)

    # prefix caching and chunked prefill (phase 5)
    prefix_cache: bool = False
    chunked_prefill: bool = False
    max_num_batched_tokens: int = Field(default=512, ge=1)

    # quantization and CUDA graphs (phase 7)
    quantization: Literal["none", "int8"] = "none"
    kv_cache_dtype: Literal["auto", "int8"] = "auto"
    cuda_graphs: bool = False
    # longer contexts decode eagerly: a graph's memory grows with batch x context
    cuda_graph_max_context: int = Field(default=1024, ge=1)
    # graphs pay off where launch overhead dominates: small batches; larger ones decode eagerly
    cuda_graph_max_batch: int = Field(default=8, ge=1)

    # speculative decoding (phase 6), serial scheduler only
    speculative: Literal["none", "ngram", "draft"] = "none"
    num_speculative_tokens: int = Field(default=4, ge=1)
    draft_model_id: str = DEFAULT_MODEL_ID
    ngram_max_size: int = Field(default=3, ge=1)

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    @property
    def public_model_name(self) -> str:
        return self.served_model_name or self.model_id

    @property
    def is_batched(self) -> bool:
        return self.scheduler != SCHEDULER_SERIAL

    @model_validator(mode="after")
    def check_batching_has_a_kv_cache(self) -> "EngineSettings":
        if self.is_batched and self.kv_cache == KV_CACHE_NONE:
            raise ValueError(f"scheduler={self.scheduler} needs kv_cache=contiguous or paged")
        if self.prefix_cache and self.kv_cache != KV_CACHE_PAGED:
            raise ValueError("prefix_cache needs kv_cache=paged (it shares blocks)")
        if self.chunked_prefill and self.scheduler != SCHEDULER_CONTINUOUS:
            raise ValueError("chunked_prefill needs scheduler=continuous")
        if self.kv_cache == KV_CACHE_NONE and (self.kv_cache_dtype == "int8" or self.cuda_graphs):
            raise ValueError("kv_cache_dtype=int8 and cuda_graphs need a KV cache")
        uses_speculation = self.speculative != SPECULATIVE_NONE
        if uses_speculation and (
            self.is_batched or self.kv_cache == KV_CACHE_NONE or self.prefix_cache
        ):
            raise ValueError(
                "speculative decoding needs scheduler=serial, a KV cache and no prefix cache"
            )
        return self

    @property
    def is_int8_kv_cache(self) -> bool:
        return self.kv_cache_dtype == KV_CACHE_DTYPE_INT8


def load_engine_settings() -> EngineSettings:
    """Read settings; real environment variables override the optional ``NANOENGINE_ENV_FILE``."""
    env_file = os.environ.get(ENV_FILE_VARIABLE)
    return EngineSettings(_env_file=env_file) if env_file else EngineSettings()
