import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Extension name must match what _cuda_kernels.py imports.
_EXT_NAME = "murmurative_attention_ops"

# Embed the path to torch's shared libraries into the extension's RPATH so
# libc10.so / libc10_cuda.so / libtorch.so are found at import time without
# requiring LD_LIBRARY_PATH to be set.
_torch_lib = os.path.join(os.path.dirname(os.__file__), "..", "site-packages", "torch", "lib")
# Fallback: resolve via torch itself if the heuristic above is off.
try:
    import torch
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
except Exception:
    pass

# setuptools editable mode requires relative paths relative to setup.py.
ext_modules = [
    CUDAExtension(
        name=_EXT_NAME,
        sources=[
            "torch-ext/torch_binding.cpp",
            "torch-ext/wrappers.cpp",
            "src/murmurative/csrc/cuda/slot_select_attend.cu",
            "src/murmurative/csrc/cuda/slot_update.cu",
            "src/murmurative/csrc/cuda/slot_diffusion.cu",
            "src/murmurative/csrc/cuda/slot_update_diffuse.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3", "-std=c++17",
                # Turing (T4, RTX 2080)
                "-gencode=arch=compute_75,code=sm_75",
                # Ampere (A100, A6000, RTX 3090)
                "-gencode=arch=compute_80,code=sm_80",
                # Ampere (RTX 3070/3080, A40)
                "-gencode=arch=compute_86,code=sm_86",
                # Ada / Hopper fallback (via PTX)
                "-gencode=arch=compute_89,code=compute_89",
            ],
        },
        extra_link_args=[f"-Wl,-rpath,{_torch_lib}"],
    )
]

setup(
    name="murmurative-kernels",
    version="0.1.0",
    description="Slot-based murmurative attention CUDA kernels",
    license="Elastic-2.0",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
