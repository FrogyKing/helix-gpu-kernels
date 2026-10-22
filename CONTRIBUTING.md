# Contributing

## Development environment

Use uv for dependency management and preserve the existing virtual environment.
Local development supports Windows and Linux. Compile CUDA sources in the pinned
Modal development image when a compatible local toolkit is unavailable.

```powershell
uv sync --frozen --group dev
uv run pytest -m "not gpu" -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_repository.py
uv build
```

Never include credentials, authentication files, or downloaded model weights in
commits or container source uploads.

## Kernel changes

Custom operators support forward inference. Validate shape, device, dtype, and
layout at the API boundary. Accumulate reductions in FP32 and document precision
changes explicitly. Comparative benchmarks must execute the requested backend
without silently falling back to a different implementation.

Add numerical coverage for changed behavior. Investigate failed comparisons before
changing tolerances. Report unsupported cases, validation failures, and skipped
GPU tests explicitly; a skipped GPU test does not establish correctness.

## Experiments and performance claims

Publish performance claims only with measured artifacts and reproducible
configurations. Distinguish analytical memory estimates from measured traffic,
operator dispatch from device execution, and microbenchmarks from model inference.

Billable GPU batches require an approved execution and cost plan. Keep concurrency
at one, reserve the conservative timeout estimate before dispatch, and retain
reservations after failures. The local ledger is an estimate, not a provider
spending cap. H100 experiments require separate approval.

## Review and publication

Review numerical behavior, memory access bounds, synchronization, and measurement
methodology before publication. Run the CPU tests, Python lint and formatting,
CUDA formatting, documentation checks, and distribution build. Run GPU checks
appropriate to the change within the approved experiment budget.
