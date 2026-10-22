"""Conservative local reservations, never a provider-level spending cap."""

import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

GPU_SECOND = {"L4": 0.000222, "A10": 0.000306, "H100": 0.001097}
PRICE_DATE = "2026-10-22"
PRICE_SOURCE = "https://modal.com/pricing"


def estimate(gpu, timeout, jobs=1):
    if gpu not in GPU_SECOND or timeout <= 0 or jobs < 1:
        raise ValueError("Unsupported GPU or invalid execution bounds")
    # 2 physical CPU cores, 8 GiB RAM, 120s startup/tail, 25% contingency.
    rate = GPU_SECOND[gpu] + 2 * 0.0000131 + 8 * 0.00000222
    return round(jobs * (timeout + 120) * rate * 1.25, 6)


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def reserve(path, gpu, timeout, job, approved_usd, budget=30.0):
    path = Path(path)
    amount = estimate(gpu, timeout)
    if not math.isfinite(approved_usd) or approved_usd <= 0 or amount > min(approved_usd, 1.0):
        raise ValueError("Run exceeds explicit approval or automatic $1 per-run threshold")
    if not math.isfinite(budget) or not 0 < budget <= 30:
        raise ValueError("Project budget must be finite and at most $30")
    with locked(path):
        ledger = json.loads(path.read_text()) if path.exists() else {"reservations": []}
        if any(r["status"] == "reserved" for r in ledger["reservations"]):
            raise ValueError("Another GPU run is reserved; concurrency is limited to one")
        total = sum(r["reserved_usd"] for r in ledger["reservations"])
        if total + amount > min(budget, approved_usd):
            raise ValueError(
                "Cumulative conservative reservations exceed project budget or approval"
            )
        record = {
            "id": uuid.uuid4().hex,
            "created_utc": datetime.now(UTC).isoformat(),
            "job": job,
            "gpu": gpu,
            "timeout_seconds": timeout,
            "reserved_usd": amount,
            "status": "reserved",
            "pricing_date": PRICE_DATE,
        }
        ledger["reservations"].append(record)
        write_atomic(path, ledger)
    return record


def finish(path, identifier, elapsed, status):
    path = Path(path)
    with locked(path):
        ledger = json.loads(path.read_text())
        record = next(r for r in ledger["reservations"] if r["id"] == identifier)
        record.update(status=status, client_elapsed_seconds=elapsed)
        # Wall time includes builds/queueing and is explicitly NOT provider allocation time.
        write_atomic(path, ledger)
