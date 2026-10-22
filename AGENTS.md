# Engineering rules

Use uv and preserve the existing virtual environment. Develop on Windows; compile CUDA
in Modal Linux using the pinned CUDA 12.6 development image. Never print credentials.

All operators are forward-only experiments, not autograd replacements. Validate shape,
device, dtype and layout at the boundary. Accumulate reductions in FP32. Do not silently
fall back to another implementation in comparative benchmarks.

No performance claims without persisted measured artifacts. GPU skips do not establish
correctness. Record failures and unsupported cases explicitly. Never relax a tolerance
just to pass. Explain precision modes and analytical memory estimates.

User approval is required before billable GPU batches. Keep concurrency at one, reserve
the timeout estimate before dispatch, retain reservations on failure, and do not treat
the local ledger as a provider spending cap. No long H100 experiments without approval.

Parallel work must have explicit file ownership. Review before pushing; run CPU tests,
lint, formatting, CUDA formatting, and all GPU checks authorized and available.
