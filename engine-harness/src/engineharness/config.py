"""Harness settings from ``HARNESS_*`` environment variables or the ``HARNESS_ENV_FILE`` file."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from engineharness.constants import ENV_FILE_VARIABLE, ENV_PREFIX, SCALE_CPU


class HarnessSettings(BaseSettings):
    """Which engine to measure, at which scale, and what this phase is expected to fail."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    target_url: str = "http://localhost:8000"
    model_id: str = "Qwen/Qwen3-0.6B"
    tokenizer_id: str = "Qwen/Qwen3-0.6B"
    workload_scale: str = SCALE_CPU
    seed: int = 1234

    expected_failures: set[str] = set()
    skipped_suites: set[str] = set()

    request_timeout_seconds: float = 600.0
    connection_pool_size: int = 128
    server_ready_timeout_seconds: float = 600.0

    concurrency_levels: list[int] = [1, 2, 4, 8]
    concurrency_min_speedup: float = 2.0

    equivalence_prompt_count: int = 10
    equivalence_output_tokens: int = 24
    min_equivalence_agreement: float = 1.0
    reference_file: Path = Path("reference/greedy-reference.json")

    prefix_min_ttft_reduction: float = 0.5
    interference_max_itl_ratio: float = 2.0
    capacity_max_error_rate: float = 0.0

    results_dir: Path = Path("results")


def load_harness_settings() -> HarnessSettings:
    """Read settings; real environment variables override the optional ``HARNESS_ENV_FILE``."""
    env_file = os.environ.get(ENV_FILE_VARIABLE)
    return HarnessSettings(_env_file=env_file) if env_file else HarnessSettings()
