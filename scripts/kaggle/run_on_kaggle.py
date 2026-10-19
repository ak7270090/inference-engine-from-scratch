"""Run one engine and the harness on a free Kaggle T4, then download the results.

    python scripts/kaggle/run_on_kaggle.py vllm --owner <kaggle-user>
    python scripts/kaggle/run_on_kaggle.py phase-01-naive --owner <kaggle-user>

Needs the ``kaggle`` CLI and ``KAGGLE_API_TOKEN`` (read from the repo's ``.env`` if not set).
The notebook clones the public repo, so push your commits first.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_TEMPLATE = Path(__file__).resolve().parent / "run_phase.ipynb"
BUILD_DIR = Path(__file__).resolve().parent / "build"
RESULTS_ROOT = REPO_ROOT / "docs" / "results"
ENV_FILE = REPO_ROOT / ".env"

KAGGLE_ACCELERATOR = "NvidiaTeslaT4"
POLL_INTERVAL_SECONDS = 30
KAGGLE_ATTEMPTS = 3
KERNEL_STATUS_PATTERN = re.compile(r'has status "([^"]+)"')
FINISHED_STATUSES = ("complete", "error", "cancel")
STARTED_STATUSES = ("queued", "running")
# right after a push, "status" can still describe the previous version's finished run
POLLS_BEFORE_TRUSTING_A_FINISHED_STATUS = 10
# the CLI exits 0 on these, so they are matched in its output
PUSH_ERROR_MARKER = "Kernel push error"
GPU_SESSION_LIMIT_MARKER = "Maximum batch GPU session count"
GPU_SESSION_WAIT_SECONDS = 120
MAX_GPU_SESSION_WAITS = 90
NOTEBOOK_FILE_NAME = "run_phase.ipynb"


def load_env_file(env_file: Path) -> None:
    """Copy ``KEY=value`` lines into the environment without overriding what is already set."""
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def run_kaggle_with_retry(*arguments: str) -> str:
    """Kaggle's API sometimes answers with an empty body; one retry is usually enough."""
    for attempt in range(KAGGLE_ATTEMPTS):
        try:
            return run_kaggle(*arguments)
        except RuntimeError:
            if attempt + 1 == KAGGLE_ATTEMPTS:
                raise
            time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError("unreachable")


def run_kaggle(*arguments: str) -> str:
    kaggle_executable = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    result = subprocess.run(
        [kaggle_executable, *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    output = result.stdout + result.stderr
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        output = output.replace(token, "***")
    if result.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(arguments)} failed:\n{output}")
    return output


def build_kernel_dir(target: str, repo_ref: str, owner: str) -> tuple[Path, str]:
    kernel_slug = f"nanoengine-{target}".lower()
    kernel_dir = BUILD_DIR / kernel_slug
    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook = NOTEBOOK_TEMPLATE.read_text(encoding="utf-8")
    notebook = notebook.replace("{{TARGET}}", target).replace("{{REPO_REF}}", repo_ref)
    (kernel_dir / NOTEBOOK_FILE_NAME).write_text(notebook, encoding="utf-8")
    metadata = {
        "id": f"{owner}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": NOTEBOOK_FILE_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return kernel_dir, f"{owner}/{kernel_slug}"


def read_kernel_status(status_output: str) -> str:
    """Only the quoted status counts; CLI warnings around it may contain words like 'error'."""
    match = KERNEL_STATUS_PATTERN.search(status_output)
    return match.group(1).lower() if match else "unknown"


def push_kernel(kernel_dir: Path) -> str:
    """Push and start a run; wait while both free GPU sessions are busy."""
    for _ in range(MAX_GPU_SESSION_WAITS):
        output = run_kaggle_with_retry(
            "kernels", "push", "-p", str(kernel_dir), "--accelerator", KAGGLE_ACCELERATOR
        )
        if GPU_SESSION_LIMIT_MARKER in output:
            print(f"both GPU sessions busy; retrying in {GPU_SESSION_WAIT_SECONDS} s")
            time.sleep(GPU_SESSION_WAIT_SECONDS)
            continue
        if PUSH_ERROR_MARKER in output:
            raise RuntimeError(f"push failed: {output}")
        return output
    raise RuntimeError("no free GPU session")


def wait_for_kernel(kernel_id: str) -> str:
    """Wait until this push's run finishes; ignore a finished status left by the previous run."""
    has_started = False
    for poll in range(10**6):
        try:
            status = read_kernel_status(run_kaggle_with_retry("kernels", "status", kernel_id))
        except RuntimeError as error:  # the run continues on Kaggle while this machine is offline
            print(f"status check failed, retrying: {str(error)[:200]}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        print(f"{kernel_id}: {status}")
        has_started = has_started or any(started in status for started in STARTED_STATUSES)
        is_finished = any(finished in status for finished in FINISHED_STATUSES)
        if is_finished and (has_started or poll >= POLLS_BEFORE_TRUSTING_A_FINISHED_STATUS):
            return status
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="'vllm' or a nanoengine profile such as phase-01-naive")
    parser.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"), required=False)
    parser.add_argument("--repo-ref", default="main", help="branch or tag the notebook clones")
    parser.add_argument(
        "--no-push", action="store_true", help="only wait for the current run and download it"
    )
    parsed = parser.parse_args()
    load_env_file(ENV_FILE)
    owner = parsed.owner or os.environ.get("KAGGLE_USERNAME")
    if not owner:
        parser.error("pass --owner or set KAGGLE_USERNAME")

    kernel_dir, kernel_id = build_kernel_dir(parsed.target, parsed.repo_ref, owner)
    if not parsed.no_push:
        print(push_kernel(kernel_dir))
    status = wait_for_kernel(kernel_id)
    results_dir = RESULTS_ROOT / f"{parsed.target}-t4"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(run_kaggle_with_retry("kernels", "output", kernel_id, "-p", str(results_dir), "--force"))
    print(f"results in {results_dir}")
    return 0 if "complete" in status else 1


if __name__ == "__main__":
    sys.exit(main())
