from .config import configure, get_config, reset_config
from .device_tensor import DeviceTensor, to_device
from .frontend import (
    MatCoreNamespace,
    MatCoreTensorView,
    SUPPORTED_INPUT_DTYPES,
    SUPPORTED_TARGETS,
    asdtype,
    create_plan,
    execute_plan,
    jit,
    fused,
    kernel,
    launch,
    load,
    add,
    relu,
    cast,
    matmul,
    softmax,
    block_attn_res,
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

exp = mc.exp
log = mc.log
sqrt = mc.sqrt
tanh = mc.tanh
sigmoid = mc.sigmoid
gelu = mc.gelu
add = mc.add
relu = mc.relu
cast = mc.cast
neg = mc.neg
abs = mc.abs
sin = mc.sin
cos = mc.cos
rsqrt = mc.rsqrt
softmax = mc.softmax
block_attn_res = mc.block_attn_res
min = mc.min
max = mc.max

__all__ = [
    "mc",
    "kernel",
    "launch",
    "create_plan",
    "execute_plan",
    "jit",
    "fused",
    "to_device",
    "DeviceTensor",
    "asdtype",
    "load",
    "add",
    "relu",
    "cast",
    "store",
    "matmul",
    "softmax",
    "block_attn_res",
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
