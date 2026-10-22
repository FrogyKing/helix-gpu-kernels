"""Forward-only Triton implementations, loaded lazily for CPU/Windows tooling."""

from importlib import import_module

__all__ = ["attention", "gemm", "quantized_linear", "rmsnorm", "softmax"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    module = "quantization" if name == "quantized_linear" else name
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
