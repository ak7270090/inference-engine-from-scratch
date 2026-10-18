"""Write ``kaggle/run_phase.ipynb``: ``python kaggle/build_notebook.py``.

The notebook clones the repo, starts vLLM or nanoengine with a profile, runs the harness suites and
sweeps, and leaves the results in ``/kaggle/working``. ``run_on_kaggle.py`` fills in the
placeholders before each push.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "run_phase.ipynb"

CELLS = [
    """# Filled in by kaggle/run_on_kaggle.py before each push.
TARGET = "{{TARGET}}"
REPO_REF = "{{REPO_REF}}"
REPO_URL = "https://github.com/ak7270090/inference-engine-from-scratch.git"
MODEL_ID = "Qwen/Qwen3-0.6B"
REPO_DIR = "/tmp/repo"
WORK_DIR = "/kaggle/working"
HARNESS_WORKLOADS = ["short_chat", "long_prompt", "shared_prefix", "mixed_lengths"]""",
    """import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def run_and_print(command, save_to=None, **options):
    result = subprocess.run(command, capture_output=True, text=True, **options)
    print("$", " ".join(command), "->", result.returncode)
    print(result.stdout[-20000:], result.stderr[-5000:])
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        Path(save_to).write_text(result.stdout, encoding="utf-8")
    return result.returncode


run_and_print(["nvidia-smi"])
shutil.rmtree(REPO_DIR, ignore_errors=True)
run_and_print(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, REPO_DIR])""",
    '''pip_install = [sys.executable, "-m", "pip", "install", "-q"]
run_and_print([*pip_install, "-e", f"{REPO_DIR}/engine-harness"])
if TARGET == "vllm":
    run_and_print([*pip_install, "vllm"])
    # Kaggle's preinstalled torchaudio is built for another CUDA and breaks vLLM's imports
    run_and_print([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
    server_command = [
        "vllm", "serve", MODEL_ID, "--dtype", "half", "--port", "8000", "--max-model-len", "4096",
    ]
    harness_env_file = f"{REPO_DIR}/engine-harness/baselines/vllm-t4.env"
else:
    # TARGET is a profile such as phase-01-naive: engine modes + harness expectations
    run_and_print([*pip_install, "-e", f"{REPO_DIR}/nanoengine"])
    server_command = [sys.executable, "-m", "nanoengine"]
    os.environ.update(
        NANOENGINE_ENV_FILE=f"{REPO_DIR}/nanoengine/profiles/{TARGET}.env",
        NANOENGINE_DEVICE="cuda",
        NANOENGINE_DTYPE="float16",
    )
    harness_env_file = f"{REPO_DIR}/nanoengine/harness/{TARGET}-t4.env"''',
    """server_log = open(f"{WORK_DIR}/server.log", "w")  # noqa: SIM115 - stays open for the server
server = subprocess.Popen(
    server_command,
    stdout=server_log,
    stderr=subprocess.STDOUT,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
)""",
    """harness_env = {
    **os.environ,
    "HARNESS_ENV_FILE": harness_env_file,
    "HARNESS_RESULTS_DIR": f"{WORK_DIR}/results",
}
harness_dir = f"{REPO_DIR}/engine-harness"
wait_command = [sys.executable, "-m", "engineharness", "wait"]
if run_and_print(wait_command, cwd=harness_dir, env=harness_env) == 0:
    run_and_print(
        [sys.executable, "-m", "pytest", "tests/suites", "-rxXs"],
        save_to=f"{WORK_DIR}/results/harness-pytest.txt",
        cwd=harness_dir,
        env=harness_env,
    )
    for workload in HARNESS_WORKLOADS:
        run_and_print(
            [sys.executable, "-m", "engineharness", "sweep", workload],
            cwd=harness_dir,
            env=harness_env,
        )
    # peak concurrency, preemptions and KV usage, as the engine itself counted them
    with urllib.request.urlopen("http://localhost:8000/metrics", timeout=30) as response:
        Path(f"{WORK_DIR}/results/engine-metrics.txt").write_bytes(response.read())
if TARGET != "vllm":
    run_and_print(
        [sys.executable, "-m", "nanoengine.step_timing", "--output-dir", f"{WORK_DIR}/results"]
    )""",
    """server.terminate()
server.wait(timeout=60)
server_log.close()
print(Path(f"{WORK_DIR}/server.log").read_text()[-5000:])
report = Path(f"{WORK_DIR}/results/harness-report.md")
print(report.read_text() if report.exists() else "no harness report")""",
]


def build_cell(source: str) -> dict:
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }


def main() -> None:
    notebook = {
        "cells": [build_cell(cell) for cell in CELLS],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
