from .config import configure, get_config, reset_config
from .frontend import (
    MatCoreNamespace,
    MatCoreTensorView,
    SUPPORTED_INPUT_DTYPES,
    SUPPORTED_TARGETS,
    asdtype,
    kernel,
    launch,
    load,
    add,
    relu,
    cast,
    matmul,
    mc,
    store,
    transpose,
)
from .graph import FusionGraph, graph
from .cache import cache_clear, cache_info, cache_summary
from .validation import (
    MatmulCorrectnessReport,
    benchmark_matcore_matmul,
    benchmark_numpy_matmul,
    check_matmul_correctness,
    make_reference_matmul,
    matmul_kernel,
)

MatCoreNamespace.cache_info = staticmethod(cache_info)
MatCoreNamespace.cache_clear = staticmethod(cache_clear)
MatCoreNamespace.cache_summary = staticmethod(cache_summary)

__all__ = [
    "mc",
    "kernel",
    "launch",
    "asdtype",
    "load",
    "add",
    "relu",
    "cast",
    "store",
    "matmul",
    "transpose",
    "graph",
    "FusionGraph",
    "MatCoreTensorView",
    "SUPPORTED_TARGETS",
    "SUPPORTED_INPUT_DTYPES",
    "matmul_kernel",
    "make_reference_matmul",
    "check_matmul_correctness",
    "benchmark_numpy_matmul",
    "benchmark_matcore_matmul",
    "MatmulCorrectnessReport",
    "cache_info",
    "cache_clear",
    "cache_summary",
    "configure",
    "get_config",
    "reset_config",
]
