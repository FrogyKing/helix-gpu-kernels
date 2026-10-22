.PHONY: setup lint test gpu-test benchmark dashboard plan build
setup:
	uv sync --group dev
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run python scripts/check_repository.py
test:
	uv run pytest -m "not gpu" -q
plan:
	uv run helix plan
gpu-test:
	uv run modal run modal_apps/test_kernels.py
benchmark:
	uv run modal run modal_apps/benchmark.py
dashboard:
	uv run --extra dashboard python -m streamlit run dashboard/app.py
build:
	uv build
