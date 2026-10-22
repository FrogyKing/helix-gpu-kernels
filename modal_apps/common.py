import json
import subprocess
import time
from pathlib import Path

import modal

from helix.runtime.budget import estimate, finish, reserve, write_atomic

ROOT = Path(__file__).resolve().parents[1]
GPU = "L4"
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "git")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "ninja==1.11.1.4",
        "pytest==8.3.5",
        "numpy==2.2.5",
        "transformers==4.51.3",
        "safetensors==0.5.3",
        "accelerate==1.6.0",
    )
    .env({"PYTHONPATH": "/workspace", "MAX_JOBS": "2", "TOKENIZERS_PARALLELISM": "false"})
)
for directory in ["helix", "triton_kernels", "cuda", "tests", "benchmarks/configs", "modal_apps"]:
    image = image.add_local_dir(ROOT / directory, remote_path=f"/workspace/{directory}")
image = image.add_local_file(ROOT / "pyproject.toml", remote_path="/workspace/pyproject.toml")


def function_options(timeout):
    return dict(
        image=image,
        gpu=GPU,
        cpu=2,
        memory=8192,
        timeout=timeout,
        startup_timeout=120,
        max_containers=1,
        min_containers=0,
        scaledown_window=2,
        retries=0,
        single_use_containers=True,
    )


def provenance():
    def git(*args):
        process = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return process.stdout.strip() if process.returncode == 0 else "uncommitted"

    return {
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "gpu_requested": GPU,
    }


def dispatch(function, job, timeout, execute=False, approved_usd=0.0, **kwargs):
    plan = {
        "job": job,
        "gpu": GPU,
        "timeout_seconds": timeout,
        "max_concurrency": 1,
        "reservation_usd": estimate(GPU, timeout),
    }
    print(json.dumps(plan, indent=2))
    if not execute:
        print("DRY RUN: no remote function invoked. Use --execute only after explicit approval.")
        return
    ledger = ROOT / "benchmarks/cost_ledger.json"
    record = reserve(ledger, GPU, timeout, job, approved_usd)
    started = time.perf_counter()
    status = "failed"
    try:
        result = function.remote(**kwargs)
        result.update(provenance(), cost_reservation=record)
        target = ROOT / "benchmarks/results" / f"{job}-{record['id']}.json"
        write_atomic(target, result)
        if "rows" in result:
            from helix.benchmark import export_csv

            export_csv(target.with_suffix(".csv"), result["rows"])
        print(f"Artifact: {target}")
        status = "completed" if result.get("success", True) else "validation_failed"
        if status != "completed":
            raise RuntimeError(f"Remote validation failed. See {target}")
    finally:
        finish(ledger, record["id"], time.perf_counter() - started, status)
