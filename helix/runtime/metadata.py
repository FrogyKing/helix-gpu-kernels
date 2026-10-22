import hashlib
import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path


def command(args):
    try:
        return subprocess.check_output(
            args, text=True, timeout=15, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}"


def hardware():
    import torch

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "driver": command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "nvcc": command(["nvcc", "--version"]),
        "profiler_tools": {tool: shutil.which(tool) for tool in ["ncu", "nsys"]},
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in [
        "helix",
        "triton_kernels",
        "cuda",
        "tests",
        "benchmarks/configs",
        "modal_apps",
    ]:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".json"}:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    result["source_sha256"] = digest.hexdigest()
    try:
        import triton

        result["triton"] = triton.__version__
    except ImportError:
        result["triton"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        result.update(
            gpu=p.name,
            compute_capability=list(torch.cuda.get_device_capability()),
            total_memory_bytes=p.total_memory,
            multiprocessors=p.multi_processor_count,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,
        )
    return result
