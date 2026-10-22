"""Local CPU dashboard. Synthetic examples are explicitly separate from measurements."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Helix GPU Laboratory", page_icon="🧬", layout="wide")
st.title("Helix GPU Laboratory")
st.caption("Reproducible experiments · CUDA / Triton / PyTorch · forward inference")
source = st.sidebar.radio("Artifact source", ["Measured experiments", "Synthetic UI demonstration"])
if source == "Synthetic UI demonstration":
    st.warning(
        "SYNTHETIC UI DEMONSTRATION — these values are invented and are not GPU measurements."
    )
    artifacts = [json.loads((ROOT / "dashboard/examples/demo.json").read_text())]
else:
    artifacts = []
    for path in sorted((ROOT / "benchmarks/results").glob("benchmark-*.json")):
        value = json.loads(path.read_text())
        if value.get("measured") is True:
            artifacts.append(value)
    if not artifacts:
        st.info(
            "No measured artifacts yet. Run the approved Modal benchmark job to populate this view."
        )
        st.stop()
records = []
for artifact in artifacts:
    for row in artifact.get("rows", []):
        records.append(
            {
                **row,
                "gpu": artifact.get("hardware", {}).get("gpu", "unknown"),
                "run": artifact.get("cost_reservation", {}).get("id", "demo"),
                "reserved_cost_usd": artifact.get("cost_reservation", {}).get("reserved_usd", 0),
                "shape_label": " × ".join(map(str, row["shape"])),
            }
        )
df = pd.DataFrame(records)
if df.empty:
    st.info("Artifacts contain no benchmark rows.")
    st.stop()
for label, column in [
    ("Kernel", "operation"),
    ("GPU", "gpu"),
    ("Shape", "shape_label"),
    ("Precision", "dtype"),
    ("Run", "run"),
]:
    selection = st.sidebar.selectbox(label, ["All"] + sorted(df[column].unique().tolist()))
    if selection != "All":
        df = df[df[column] == selection]
if "causal" in df:
    causal = st.sidebar.selectbox("Causal masking", ["All", "Causal", "Noncausal"])
    if causal != "All":
        df = df[df.causal == (causal == "Causal")]
failed = df[df.status != "passed"]
if not failed.empty:
    st.error(f"{len(failed)} cases failed validation. They are excluded from performance charts.")
    st.dataframe(failed)
valid = df[df.status == "passed"]
if valid.empty:
    st.stop()
a, b, c = st.columns(3)
a.metric(
    "Measured comparisons" if source == "Measured experiments" else "Demo comparisons", len(valid)
)
b.metric("Lowest median (µs)", f"{valid.median_us.min():.2f}")
c.metric("Reserved run cost (USD)", f"{valid.groupby('run').reserved_cost_usd.first().sum():.4f}")
st.caption(
    "Cost is a conservative reservation, not a provider invoice. Compare matching shapes and precision."
)
left, right = st.columns(2)
left.plotly_chart(
    px.bar(
        valid,
        x="backend",
        y="median_us",
        color="dtype",
        barmode="group",
        hover_data=["operation", "shape_label", "gpu"],
        title="Median operator latency (µs)",
    ),
    width="stretch",
)
right.plotly_chart(
    px.bar(
        valid,
        x="backend",
        y="speedup_vs_pytorch",
        color="dtype",
        barmode="group",
        title="Speedup vs native PyTorch baseline",
    ),
    width="stretch",
)
samples = []
for _, row in valid.iterrows():
    for value in row.get("samples_us", []):
        samples.append(
            {
                "backend": row.backend,
                "latency_us": value,
                "dtype": row["dtype"],
                "shape": row.shape_label,
            }
        )
if samples:
    st.plotly_chart(
        px.box(
            pd.DataFrame(samples),
            x="backend",
            y="latency_us",
            color="dtype",
            points="all",
            title="Latency distribution",
        ),
        width="stretch",
    )
for metric, title in [
    ("achieved_useful_tflops", "Useful arithmetic throughput (TFLOP/s)"),
    ("effective_minimum_io_gbs", "Analytical minimum I/O / latency (GB/s)"),
    ("max_abs_error", "Maximum absolute numerical error"),
    ("peak_extra_allocated_bytes", "Measured peak incremental allocated memory"),
]:
    if metric in valid:
        st.plotly_chart(
            px.bar(valid, x="backend", y=metric, color="dtype", barmode="group", title=title),
            width="stretch",
        )
st.caption("Minimum I/O bandwidth is an analytical lower bound, not a hardware DRAM counter.")
if "cuda_graph_median_us" in valid:
    st.plotly_chart(
        px.bar(
            valid,
            x="backend",
            y="cuda_graph_median_us",
            color="dtype",
            barmode="group",
            title="CUDA Graph replay: amortized device time per operator (µs)",
        ),
        width="stretch",
    )
    st.caption(
        "Ten captured operators per replay. Python scheduling and output allocation excluded."
    )
st.dataframe(valid.drop(columns=["samples_us"], errors="ignore"), width="stretch")
with st.expander("Hardware, profiling and inference artifacts"):
    for artifact in artifacts:
        st.json({key: value for key, value in artifact.items() if key != "rows"})
