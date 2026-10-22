"""Generate a Markdown table from actual artifacts, without filling missing data."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("artifact", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
artifact = json.loads(args.artifact.read_text())
if artifact.get("measured") is not True:
    raise ValueError("Only measured artifacts are eligible for performance reports")
rows = artifact.get("rows", [])
hardware = artifact["hardware"]
text = [
    "# Measured operator results",
    "",
    f"Source: `{args.artifact.name}`.",
    "",
    f"GPU: {hardware['gpu']}; CUDA {hardware['cuda_runtime']}; "
    f"PyTorch {hardware['torch']}; Triton {hardware['triton']}.",
    "",
    "Dispatch latency includes Python scheduling/output allocation device intervals. "
    "Graph replay amortizes ten captured operators. Minimum I/O is analytical.",
    "",
    "| Operation | Shape | Dtype | Causal | Backend | Median µs | p95 µs | Graph µs | Speedup | Max abs error |",
    "|---|---|---|---|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    prefix = (
        f"| {row['operation']} | {row['shape']} | {row['dtype']} | "
        f"{row['causal']} | {row['backend']} |"
    )
    if row["status"] == "passed":
        text.append(
            prefix + f" {row['median_us']:.3f} | {row['p95_us']:.3f} | "
            f"{row.get('cuda_graph_median_us', float('nan')):.3f} | "
            f"{row['speedup_vs_pytorch']:.3f}× | {row['max_abs_error']:.3g} |"
        )
    else:
        text.append(prefix + " FAILED | — | — | — | — |")
text += ["", "## Validation failures", ""]
text += [
    f"- {row['operation']} / {row['backend']} / {row['shape']}: {row.get('error')}"
    for row in rows
    if row["status"] != "passed"
] or ["None."]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text("\n".join(text) + "\n", encoding="utf-8")
print(args.output)
