import modal

from modal_apps.common import dispatch, function_options

app = modal.App("helix-benchmarks")


@app.function(**function_options(600))
def run_benchmarks(config: str = "smoke", inference: bool = False):
    import time

    from helix.benchmark import run_suite

    if config not in {"smoke", "transformer"}:
        raise ValueError("Unknown bounded configuration")
    started = time.perf_counter()
    result = run_suite(f"/workspace/benchmarks/configs/{config}.json", deadline_seconds=420)
    from helix.profiling.trace import probe_nsight_compute, profile_softmax

    result["profiling"] = profile_softmax()
    result["nsight_compute_probe"] = probe_nsight_compute()
    if inference:
        from helix.integration.smollm import run_inference

        result["inference"] = run_inference(repeats=3, new_tokens=16)
    result["remote_body_elapsed_seconds"] = time.perf_counter() - started
    return result


@app.local_entrypoint()
def main(
    execute: bool = False, approved_usd: float = 0.0, config: str = "smoke", inference: bool = False
):
    dispatch(
        run_benchmarks, "benchmark", 600, execute, approved_usd, config=config, inference=inference
    )
