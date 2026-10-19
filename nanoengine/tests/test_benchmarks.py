from pathlib import Path

import pytest
import torch

from nanoengine.benchmarks import step_timing
from nanoengine.benchmarks.step_timing import (
    CachedDecodeStepBenchmark,
    NaiveStepBenchmark,
    measure_step_timings,
    write_step_timings,
)

CPU = torch.device("cpu")


@pytest.mark.parametrize("benchmark_kind", ["naive", "cached"])
def test_step_timings_cover_every_length_and_are_written(
    tiny_model, tmp_path: Path, benchmark_kind: str
) -> None:
    if benchmark_kind == "naive":
        benchmark = NaiveStepBenchmark(tiny_model, CPU)
    else:
        benchmark = CachedDecodeStepBenchmark(tiny_model, CPU, torch.float32, 16)

    timings = measure_step_timings(benchmark, [4, 16], 2, CPU, tiny_model.config.vocab_size)
    write_step_timings(timings, "cpu-float32", benchmark.description, tmp_path)

    assert [timing.sequence_length for timing in timings] == [4, 16]
    assert all(timing.min_step_ms <= timing.median_step_ms for timing in timings)
    assert (tmp_path / "step-time-vs-length-cpu-float32.png").exists()
    assert benchmark.description in (tmp_path / "step-time-vs-length-cpu-float32.md").read_text()


@pytest.mark.parametrize(
    ("kv_cache", "expected_file"),
    [
        ("none", "step-time-vs-length-cpu-float32.json"),
        ("contiguous", "step-time-vs-length-cpu-float32-kv-cache.json"),
    ],
)
def test_main_measures_the_configured_mode(
    monkeypatch, tiny_model, tmp_path: Path, kv_cache: str, expected_file: str
) -> None:
    monkeypatch.setenv("NANOENGINE_KV_CACHE", kv_cache)
    monkeypatch.setattr(step_timing, "download_model_files", lambda model_id, cache_dir: tmp_path)
    monkeypatch.setattr(
        step_timing, "load_qwen3_model", lambda model_dir, device, dtype: tiny_model
    )

    step_timing.main(["--lengths", "4", "--repeats", "1", "--output-dir", str(tmp_path)])

    assert (tmp_path / expected_file).exists()
