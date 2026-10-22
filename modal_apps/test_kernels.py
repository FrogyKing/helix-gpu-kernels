import modal

from modal_apps.common import dispatch, function_options

app = modal.App("helix-kernel-tests")


@app.function(**function_options(900))
def run_tests():
    import subprocess
    import time

    from helix.runtime.metadata import hardware

    started = time.perf_counter()
    process = subprocess.run(
        ["python", "-m", "pytest", "/workspace/tests", "-m", "not integration", "-ra", "-q"],
        cwd="/workspace",
        capture_output=True,
        text=True,
        timeout=840,
    )
    return {
        "hardware": hardware(),
        "success": process.returncode == 0,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "remote_body_elapsed_seconds": time.perf_counter() - started,
    }


@app.local_entrypoint()
def main(execute: bool = False, approved_usd: float = 0.0):
    dispatch(run_tests, "test_kernels", 900, execute, approved_usd)
