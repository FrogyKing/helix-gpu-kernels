"""Portable checks for CUDA formatting, docs links and JSON configurations."""

import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = [str(p) for p in (root / "cuda").rglob("*") if p.suffix in {".cu", ".cuh", ".cpp", ".h"}]
subprocess.run(["clang-format", "--dry-run", "--Werror", *sources], check=True)
for path in (root / "benchmarks/configs").glob("*.json"):
    json.loads(path.read_text())
for path in [root / "README.md", *list((root / "docs").rglob("*.md"))]:
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" not in target and not target.startswith("#"):
            assert (path.parent / target.split("#")[0]).exists(), (path, target)
print("CUDA formatting, documentation links and configurations verified")
