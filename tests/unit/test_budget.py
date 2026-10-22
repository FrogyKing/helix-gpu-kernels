import json

import pytest

from helix.runtime.budget import estimate, finish, reserve


def test_reservation_is_retained_on_failure(tmp_path):
    ledger = tmp_path / "ledger.json"
    record = reserve(ledger, "L4", 120, "info", 1)
    finish(ledger, record["id"], 1, "failed")
    saved = json.loads(ledger.read_text())["reservations"][0]
    assert saved["reserved_usd"] == estimate("L4", 120)
    assert saved["status"] == "failed"
    with pytest.raises(ValueError, match="budget"):
        reserve(ledger, "L4", 120, "info", 1, budget=saved["reserved_usd"])


def test_requires_approval_and_per_run_bound(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        reserve(tmp_path / "a.json", "L4", 120, "info", 0)
    with pytest.raises(ValueError, match="threshold"):
        reserve(tmp_path / "a.json", "H100", 3600, "expensive", 10)
