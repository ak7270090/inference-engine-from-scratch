"""Run one engine and the harness on a free Kaggle T4, then download the results.

    python kaggle/run_on_kaggle.py vllm --owner <kaggle-user>
    python kaggle/run_on_kaggle.py 01-naive-generation --owner <kaggle-user>

Needs the ``kaggle`` CLI and ``KAGGLE_API_TOKEN`` (read from the repo's ``.env`` if not set).
The notebook clones the public repo, so push your commits first.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_TEMPLATE = Path(__file__).resolve().parent / "run_phase.ipynb"
BUILD_DIR = Path(__file__).resolve().parent / "build"
RESULTS_ROOT = REPO_ROOT / "docs" / "results"
ENV_FILE = REPO_ROOT / ".env"

KAGGLE_ACCELERATOR = "NvidiaTeslaT4"
POLL_INTERVAL_SECONDS = 30
FINISHED_STATUSES = ("complete", "error", "cancel")
NOTEBOOK_FILE_NAME = "run_phase.ipynb"


def load_env_file(env_file: Path) -> None:
    """Copy ``KEY=value`` lines into the environment without overriding what is already set."""
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


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


def wait_for_kernel(kernel_id: str) -> str:
    while True:
        status = run_kaggle("kernels", "status", kernel_id).lower()
        print(status.strip())
        if any(finished in status for finished in FINISHED_STATUSES):
            return status
        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="'vllm' or a phase folder such as 01-naive-generation")
    parser.add_argument("--owner", default=os.environ.get("KAGGLE_USERNAME"), required=False)
    parser.add_argument("--repo-ref", default="main", help="branch or tag the notebook clones")
    parsed = parser.parse_args()
    load_env_file(ENV_FILE)
    owner = parsed.owner or os.environ.get("KAGGLE_USERNAME")
    if not owner:
        parser.error("pass --owner or set KAGGLE_USERNAME")

    kernel_dir, kernel_id = build_kernel_dir(parsed.target, parsed.repo_ref, owner)
    print(run_kaggle("kernels", "push", "-p", str(kernel_dir), "--accelerator", KAGGLE_ACCELERATOR))
    status = wait_for_kernel(kernel_id)
    results_dir = RESULTS_ROOT / f"{parsed.target}-t4"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(run_kaggle("kernels", "output", kernel_id, "-p", str(results_dir)))
    print(f"results in {results_dir}")
    return 0 if "complete" in status else 1


if __name__ == "__main__":
    sys.exit(main())
