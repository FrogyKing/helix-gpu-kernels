import modal

from modal_apps.common import dispatch, function_options

app = modal.App("helix-gpu-info")


@app.function(**function_options(120))
def inspect_gpu():
    import time

    from helix.runtime.metadata import hardware

    started = time.perf_counter()
    result = hardware()
    result["remote_body_elapsed_seconds"] = time.perf_counter() - started
    return result


@app.local_entrypoint()
def main(execute: bool = False, approved_usd: float = 0.0):
    dispatch(inspect_gpu, "gpu_info", 120, execute, approved_usd)
