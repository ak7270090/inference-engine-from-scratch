from pathlib import Path

import torch

from nanoengine import step_timing
from nanoengine.engine.naive_generator import NaiveGenerator
from nanoengine.step_timing import measure_step_timings, write_step_timings

CPU = torch.device("cpu")


def test_step_timings_cover_every_length_and_are_written(tiny_model, tmp_path: Path) -> None:
    generator = NaiveGenerator(tiny_model, frozenset(), CPU)

    timings = measure_step_timings(generator, [4, 16], repeats=2, device=CPU)
    write_step_timings(timings, "cpu-float32", tmp_path)

    assert [timing.sequence_length for timing in timings] == [4, 16]
    assert all(timing.min_step_ms <= timing.median_step_ms for timing in timings)
    assert (tmp_path / "step-time-vs-length-cpu-float32.png").exists()
    assert "| 16 |" in (tmp_path / "step-time-vs-length-cpu-float32.md").read_text()


def test_main_loads_the_configured_model(monkeypatch, tiny_model, tmp_path: Path) -> None:
    monkeypatch.setattr(step_timing, "download_model_files", lambda model_id, cache_dir: tmp_path)
    monkeypatch.setattr(
        step_timing, "load_qwen3_model", lambda model_dir, device, dtype: tiny_model
    )

    step_timing.main(["--lengths", "4", "--repeats", "1", "--output-dir", str(tmp_path)])

    assert (tmp_path / "step-time-vs-length-cpu-float32.json").exists()
