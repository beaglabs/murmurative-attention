"""murmurative CUDA kernel loader.

The CUDA extension is compiled at package-install time (pip install -e .)
using torch.utils.cpp_extension.CUDAExtension.  At runtime we simply import
the shared library, which triggers TORCH_LIBRARY registration and makes the
ops available under torch.ops.murmurative_attention.
"""

_loaded = False


def ensure_compiled() -> bool:
    """Verify that the compiled CUDA extension is loaded.

    Returns True on success.  Raises RuntimeError if the extension is not
    importable (i.e. it was never built or the build is incompatible with the
    current PyTorch / CUDA environment).
    """
    global _loaded
    if _loaded:
        return True

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "murmurative CUDA kernels require a CUDA-capable device, but none is available."
        )

    try:
        import murmurative_attention_ops  # noqa: F401 — triggers torch.library registration
    except ImportError as exc:
        raise RuntimeError(
            "murmurative CUDA extension is not built or is incompatible with this "
            "environment. Rebuild with:  pip install -e .  "
            "(requires a torch CUDA installation and nvcc on PATH)."
        ) from exc

    # Verify the ops namespace was actually registered.
    try:
        _ = torch.ops.murmurative_attention
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "murmurative CUDA extension loaded but the 'murmurative_attention' "
            "torch.ops namespace was not registered. The extension may be corrupt."
        ) from exc

    _loaded = True
    return True
