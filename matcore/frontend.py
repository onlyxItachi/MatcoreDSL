from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap
from dataclasses import dataclass
from typing import Any

_NATIVE_MODULE_NAME = "_matcore_native"
SUPPORTED_TARGETS: tuple[str, ...] = (
    "x86-auto",
    "x86-avx2",
    "x86-avx512",
    "amd-igpu",
    "nvidia-dgpu",
    "amd-npu",
)
SUPPORTED_INPUT_DTYPES: tuple[str, ...] = (
    "float32",
    "float16",
    "bfloat16",
    "int8",
    "int32",
    "float8_e4m3fn",
)
_TARGET_ALIASES: dict[str, str] = {
    "x86": "x86-auto",
    "x86auto": "x86-auto",
    "x86-avx2": "x86-avx2",
    "x86_avx2": "x86-avx2",
    "x86-avx512": "x86-avx512",
    "x86_avx512": "x86-avx512",
    "amdgcn": "amd-igpu",
    "amd-igpu": "amd-igpu",
    "amd_igpu": "amd-igpu",
    "nvptx": "nvidia-dgpu",
    "nvidia-dgpu": "nvidia-dgpu",
    "nvidia_dgpu": "nvidia-dgpu",
    "npu": "amd-npu",
    "amd-npu": "amd-npu",
    "amd_npu": "amd-npu",
}
_NVIDIA_SM_TOKEN = re.compile(r"^(?:compute_)?sm?_?([0-9]{2,3})$")
_DTYPE_STORAGE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int32": 4,
    "float8_e4m3fn": 1,
}


def _expr_to_source(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _literal_int(node: ast.AST | None, default: int) -> int:
    if node is None:
        return default
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return default


def _indices_from_node(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, (ast.Tuple, ast.List)):
        indices: list[str] = []
        for element in node.elts:
            source = _expr_to_source(element)
            if source is not None:
                indices.append(source)
        return indices
    source = _expr_to_source(node)
    return [source] if source is not None else []


def _extract_params(args: ast.arguments) -> list[str]:
    params: list[str] = []
    for item in args.posonlyargs:
        params.append(item.arg)
    for item in args.args:
        params.append(item.arg)
    if args.vararg is not None:
        params.append(f"*{args.vararg.arg}")
    for item in args.kwonlyargs:
        params.append(item.arg)
    if args.kwarg is not None:
        params.append(f"**{args.kwarg.arg}")
    return params


def _normalize_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError("target must be a string")
    normalized = target.strip().lower()
    if not normalized:
        raise ValueError("target must not be empty")

    if match := _NVIDIA_SM_TOKEN.fullmatch(normalized):
        return f"nvidia-dgpu:sm_{match.group(1)}"

    base = normalized
    suffix = ""
    for separator in (":", "@", "/"):
        if separator in normalized:
            base, suffix = normalized.split(separator, 1)
            break

    base = _TARGET_ALIASES.get(base, base)
    if base not in SUPPORTED_TARGETS:
        raise ValueError(
            "Unsupported target "
            f"'{target}'. Supported base targets: {', '.join(SUPPORTED_TARGETS)}. "
            "NVIDIA compile profiles may be requested as nvidia-dgpu:sm_90."
        )

    if not suffix:
        return base

    if base != "nvidia-dgpu":
        raise ValueError(
            f"Target profile suffixes are currently only supported for nvidia-dgpu, got '{target}'."
        )

    match = _NVIDIA_SM_TOKEN.fullmatch(suffix.strip())
    if match is None:
        raise ValueError(
            f"Unsupported NVIDIA target profile '{target}'. Use forms like 'nvidia-dgpu:sm_90'."
        )
    return f"{base}:sm_{match.group(1)}"


def _normalize_dtype_name(dtype_name: str) -> str:
    lowered = dtype_name.strip().lower()
    aliases = {
        "float32": "float32",
        "single": "float32",
        "float16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "int8": "int8",
        "i8": "int8",
        "int32": "int32",
        "i32": "int32",
        "float8_e4m3fn": "float8_e4m3fn",
        "float8e4m3fn": "float8_e4m3fn",
        "f8e4m3fn": "float8_e4m3fn",
        "fp8_e4m3fn": "float8_e4m3fn",
        "fp8-e4m3fn": "float8_e4m3fn",
        "e4m3fn": "float8_e4m3fn",
    }
    return aliases.get(lowered, lowered)


def _tensor_symbol(params: list[str], idx: int) -> str:
    return params[idx] if idx < len(params) else f"arg{idx}"


def _logical_dtype_name(array: Any, idx: int) -> str:
    override = getattr(array, "matcore_dtype", None)
    if override is not None:
        dtype_name = _normalize_dtype_name(str(override))
    else:
        dtype_obj = getattr(array, "dtype", None)
        if dtype_obj is None:
            raise TypeError(f"Argument {idx} does not expose a NumPy-compatible dtype")
        dtype_name = _normalize_dtype_name(str(getattr(dtype_obj, "name", dtype_obj)))

    if dtype_name not in SUPPORTED_INPUT_DTYPES:
        raise TypeError(
            f"Unsupported dtype '{dtype_name}' for argument {idx}. "
            f"Supported dtypes: {', '.join(SUPPORTED_INPUT_DTYPES)}"
        )
    return dtype_name


def _collect_tensor_dtypes(arrays: tuple[Any, ...], params: list[str]) -> list[dict[str, str]]:
    tensor_dtypes: list[dict[str, str]] = []
    for idx, array in enumerate(arrays):
        symbol = _tensor_symbol(params, idx)
        dtype_name = _logical_dtype_name(array, idx)
        tensor_dtypes.append({"symbol": symbol, "dtype": dtype_name})
    return tensor_dtypes


def _parse_quant_entry(
    *,
    scale: Any | None,
    zero_point: Any | None,
    enabled: bool | None = None,
    default_enabled: bool = False,
) -> dict[str, Any]:
    has_explicit = scale is not None or zero_point is not None
    quant_enabled = bool(enabled) if enabled is not None else (default_enabled or has_explicit)
    quant_scale = 1.0 if scale is None else float(scale)
    quant_zero_point = 0 if zero_point is None else int(zero_point)
    return {
        "enabled": quant_enabled,
        "scale": quant_scale,
        "zero_point": quant_zero_point,
    }


def _collect_tensor_quantization(
    arrays: tuple[Any, ...], params: list[str], tensor_dtypes: list[dict[str, str]]
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    dtype_by_symbol = {entry["symbol"]: entry["dtype"] for entry in tensor_dtypes}
    for idx, array in enumerate(arrays):
        symbol = _tensor_symbol(params, idx)
        if dtype_by_symbol.get(symbol) != "int8":
            entries.append(
                {"symbol": symbol, "enabled": False, "scale": 1.0, "zero_point": 0}
            )
            continue
        scale = getattr(array, "matcore_scale", None)
        zero_point = getattr(array, "matcore_zero_point", None)
        enabled = getattr(array, "matcore_quant_enabled", None)
        quant_obj = getattr(array, "matcore_quantization", None)
        if quant_obj is not None and isinstance(quant_obj, dict):
            scale = quant_obj.get("scale", scale)
            zero_point = quant_obj.get("zero_point", zero_point)
            if "enabled" in quant_obj:
                enabled = bool(quant_obj["enabled"])

        entry = {"symbol": symbol}
        entry.update(
            _parse_quant_entry(
                scale=scale,
                zero_point=zero_point,
                enabled=enabled,
                default_enabled=False,
            )
        )
        entries.append(entry)
    return entries


def _build_global_quantization(
    quant: dict[str, Any] | None, tensor_dtypes: list[dict[str, str]]
) -> dict[str, Any]:
    has_int8 = any(entry["dtype"] == "int8" for entry in tensor_dtypes)
    if quant is None:
        return _parse_quant_entry(scale=None, zero_point=None, default_enabled=has_int8)
    if not isinstance(quant, dict):
        raise TypeError("quant must be a dict with optional scale/zero_point/enabled")
    return _parse_quant_entry(
        scale=quant.get("scale"),
        zero_point=quant.get("zero_point"),
        enabled=quant.get("enabled"),
        default_enabled=has_int8,
    )


def _build_runtime_ir(
    kernel_obj: MatCoreKernel, arrays: tuple[Any, ...], quant: dict[str, Any] | None
) -> dict[str, Any]:
    base_ir = kernel_obj.ir
    params = list(base_ir.get("params", []))
    runtime_ir = {
        "kernel_name": base_ir.get("kernel_name", kernel_obj.name),
        "params": params,
        "loops": [dict(loop) for loop in base_ir.get("loops", [])],
        "ops": [dict(op) for op in base_ir.get("ops", [])],
    }
    tensor_dtypes = _collect_tensor_dtypes(arrays, params)
    dtype_by_symbol = {entry["symbol"]: entry["dtype"] for entry in tensor_dtypes}
    tensor_quantization = _collect_tensor_quantization(arrays, params, tensor_dtypes)
    global_quantization = _build_global_quantization(quant, tensor_dtypes)
    if global_quantization["enabled"]:
        for entry in tensor_quantization:
            if dtype_by_symbol.get(entry["symbol"]) == "int8" and not entry["enabled"]:
                entry.update(global_quantization)

    runtime_ir["tensor_dtypes"] = tensor_dtypes
    runtime_ir["tensor_quantization"] = tensor_quantization
    runtime_ir["global_quantization"] = global_quantization
    return runtime_ir


@dataclass(frozen=True)
class _LogicalDTypeDescriptor:
    name: str
    itemsize: int


class MatCoreTensorView:
    """Logical dtype wrapper around a NumPy-like buffer for unsupported host dtypes."""

    def __init__(
        self,
        array: Any,
        *,
        dtype: str,
        scale: float | None = None,
        zero_point: int | None = None,
        quant_enabled: bool | None = None,
    ) -> None:
        logical_dtype = _normalize_dtype_name(dtype)
        if logical_dtype not in SUPPORTED_INPUT_DTYPES:
            raise TypeError(
                f"Unsupported logical dtype '{dtype}'. "
                f"Supported dtypes: {', '.join(SUPPORTED_INPUT_DTYPES)}"
            )
        if logical_dtype != "int8" and (
            scale is not None or zero_point is not None or quant_enabled is not None
        ):
            raise TypeError(
                "MatCore quantization metadata is currently only supported for int8 tensors."
            )
        self._array = array
        self.matcore_dtype = logical_dtype
        if scale is not None:
            self.matcore_scale = float(scale)
        if zero_point is not None:
            self.matcore_zero_point = int(zero_point)
        if quant_enabled is not None:
            self.matcore_quant_enabled = bool(quant_enabled)

    @property
    def dtype(self) -> _LogicalDTypeDescriptor:
        itemsize = _DTYPE_STORAGE_BYTES[self.matcore_dtype]
        return _LogicalDTypeDescriptor(name=self.matcore_dtype, itemsize=itemsize)

    @property
    def shape(self) -> Any:
        return self._array.shape

    @property
    def strides(self) -> Any:
        return self._array.strides

    @property
    def flags(self) -> Any:
        return self._array.flags

    @property
    def __array_interface__(self) -> Any:
        return self._array.__array_interface__

    @property
    def __cuda_array_interface__(self) -> Any:
        return self._array.__cuda_array_interface__

    def __getattr__(self, name: str) -> Any:
        return getattr(self._array, name)


def asdtype(
    array: Any,
    dtype: str,
    *,
    scale: float | None = None,
    zero_point: int | None = None,
    quant_enabled: bool | None = None,
) -> MatCoreTensorView:
    """Wrap a NumPy-like buffer with a logical MatCore dtype (e.g. bf16/fp8/int8)."""
    return MatCoreTensorView(
        array,
        dtype=dtype,
        scale=scale,
        zero_point=zero_point,
        quant_enabled=quant_enabled,
    )


class MatCoreASTVisitor(ast.NodeVisitor):
    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.params: list[str] = []
        self.loops: list[dict[str, Any]] = []
        self.ops: list[dict[str, Any]] = []
        self._inside_kernel = False

    def build(self, module: ast.Module) -> dict[str, Any]:
        self.visit(module)
        return {
            "kernel_name": self.kernel_name,
            "params": self.params,
            "loops": self.loops,
            "ops": self.ops,
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name != self.kernel_name:
            return
        self._inside_kernel = True
        self.params = _extract_params(node.args)
        self.generic_visit(node)
        self._inside_kernel = False

    def visit_For(self, node: ast.For) -> None:
        if not self._inside_kernel:
            return
        var = _expr_to_source(node.target)
        lower = 0
        upper = 0
        step = 1
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            if len(node.iter.args) == 1:
                upper = _literal_int(node.iter.args[0], 0)
            elif len(node.iter.args) >= 2:
                lower = _literal_int(node.iter.args[0], 0)
                upper = _literal_int(node.iter.args[1], 0)
                if len(node.iter.args) >= 3:
                    step = _literal_int(node.iter.args[2], 1)
        self.loops.append({"var": var, "lower": lower, "upper": upper, "step": step})
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self._inside_kernel:
            return
        targets = [_expr_to_source(target) for target in node.targets]
        for target in targets:
            value_op = self._value_to_op(node.value)
            value_op["output"] = target
            self.ops.append(value_op)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._inside_kernel:
            return
        value_op = self._value_to_op(node.value)
        value_op["output"] = _expr_to_source(node.target)
        self.ops.append(value_op)

    def visit_Expr(self, node: ast.Expr) -> None:
        if not self._inside_kernel:
            return
        if isinstance(node.value, ast.Call):
            op = self._call_to_op(node.value)
            if op["op"] == "store":
                self.ops.append(op)

    def _value_to_op(self, value: ast.AST | None) -> dict[str, Any]:
        if value is None:
            return {"op": "assign", "output": None, "value": None}
        if isinstance(value, ast.Call):
            call_op = self._call_to_op(value)
            if call_op["op"] in {"load", "matmul"}:
                return call_op
            return {"op": "assign", "output": None, "value": _expr_to_source(value)}
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.MatMult):
            return {
                "op": "matmul",
                "output": None,
                "lhs": _expr_to_source(value.left),
                "rhs": _expr_to_source(value.right),
            }
        return {"op": "assign", "output": None, "value": _expr_to_source(value)}

    def _call_to_op(self, call: ast.Call) -> dict[str, Any]:
        fn = _expr_to_source(call.func) or "<unknown>"
        args = list(call.args)
        kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}

        if self._is_marker(fn, "load"):
            tensor = _expr_to_source(args[0]) if args else None
            index_node = kwargs.get("index")
            if index_node is None and len(args) > 1:
                index_node = args[1]
            return {
                "op": "load",
                "output": None,
                "tensor": tensor,
                "indices": _indices_from_node(index_node),
            }

        if self._is_marker(fn, "store"):
            tensor = _expr_to_source(args[0]) if args else None
            index_node: ast.AST | None = kwargs.get("index")
            value_node: ast.AST | None = kwargs.get("tile")
            if len(args) > 2 and index_node is None:
                index_node = args[1]
                if value_node is None:
                    value_node = args[2]
            elif len(args) > 1 and value_node is None:
                value_node = args[1]
            return {
                "op": "store",
                "tensor": tensor,
                "value": _expr_to_source(value_node),
                "indices": _indices_from_node(index_node),
            }

        if self._is_marker(fn, "matmul"):
            return {
                "op": "matmul",
                "output": None,
                "lhs": _expr_to_source(args[0]) if len(args) > 0 else None,
                "rhs": _expr_to_source(args[1]) if len(args) > 1 else None,
            }

        return {"op": "assign", "output": None, "value": _expr_to_source(call)}

    @staticmethod
    def _is_marker(fn: str, marker: str) -> bool:
        return fn == marker or fn.endswith(f".{marker}")


@dataclass(frozen=True)
class MatCoreKernel:
    fn: Any
    source: str
    ir: dict[str, Any]

    def __call__(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("Kernel bodies are not executed from Python. Use mc.launch(...).")

    @property
    def name(self) -> str:
        return self.fn.__name__


def kernel(func: Any) -> MatCoreKernel:
    if not callable(func):
        raise TypeError("@mc.kernel expects a callable function.")
    try:
        source = inspect.getsource(func)
    except OSError as exc:
        raise RuntimeError("Unable to read kernel source for AST capture.") from exc

    dedented = textwrap.dedent(source)
    module = ast.parse(dedented)
    visitor = MatCoreASTVisitor(func.__name__)
    ir = visitor.build(module)
    return MatCoreKernel(fn=func, source=dedented, ir=ir)


def _marker_error(name: str) -> None:
    raise RuntimeError(
        f"mc.{name}() is a DSL marker and cannot be executed eagerly in Python. "
        "Use it only inside @mc.kernel functions."
    )


def load(*_: Any, **__: Any) -> Any:
    _marker_error("load")


def store(*_: Any, **__: Any) -> Any:
    _marker_error("store")


def matmul(*_: Any, **__: Any) -> Any:
    _marker_error("matmul")


def _get_native_module() -> Any:
    try:
        module = importlib.import_module(f"{__package__}.{_NATIVE_MODULE_NAME}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import native module '{_NATIVE_MODULE_NAME}'. Build the extension first."
        ) from exc

    compile_and_run = getattr(module, "compile_and_run", None)
    if not callable(compile_and_run):
        raise RuntimeError(
            f"Native module '{_NATIVE_MODULE_NAME}' does not expose compile_and_run(...)."
        )
    return module


def _optional_import_cupy() -> Any | None:
    try:
        return importlib.import_module("cupy")
    except Exception:
        return None


def _nvidia_target_requested(target: str) -> bool:
    return target.split(":", 1)[0] == "nvidia-dgpu"


def _stored_arg_indices(kernel_obj: MatCoreKernel) -> set[int]:
    stored_symbols = {
        str(op.get("tensor"))
        for op in kernel_obj.ir.get("ops", [])
        if isinstance(op, dict) and op.get("op") == "store" and op.get("tensor") is not None
    }
    return {
        idx
        for idx, param in enumerate(kernel_obj.ir.get("params", []))
        if param in stored_symbols
    }


def _view_metadata(array: MatCoreTensorView) -> dict[str, Any]:
    return {
        "dtype": array.matcore_dtype,
        "scale": getattr(array, "matcore_scale", None),
        "zero_point": getattr(array, "matcore_zero_point", None),
        "quant_enabled": getattr(array, "matcore_quant_enabled", None),
    }


def _stage_nvidia_arrays(
    kernel_obj: MatCoreKernel, arrays: tuple[Any, ...]
) -> tuple[tuple[Any, ...], list[tuple[Any, Any]]]:
    cp = _optional_import_cupy()
    if cp is None:
        raise RuntimeError(
            "nvidia-dgpu execution requires CuPy or another CUDA array provider "
            "when host NumPy arrays are passed."
        )

    stored_indices = _stored_arg_indices(kernel_obj)
    staged_arrays: list[Any] = []
    copybacks: list[tuple[Any, Any]] = []
    for idx, array in enumerate(arrays):
        underlying = array._array if isinstance(array, MatCoreTensorView) else array
        cuda_ready = hasattr(array, "__cuda_array_interface__") or hasattr(
            underlying, "__cuda_array_interface__"
        )
        if cuda_ready:
            staged_arrays.append(array)
            continue

        device_array = cp.asarray(underlying)
        if isinstance(array, MatCoreTensorView):
            staged_array = MatCoreTensorView(device_array, **_view_metadata(array))
        else:
            staged_array = device_array
        staged_arrays.append(staged_array)
        if idx in stored_indices:
            copybacks.append((underlying, device_array))
    return tuple(staged_arrays), copybacks


def _copy_back_staged_arrays(copybacks: list[tuple[Any, Any]]) -> None:
    if not copybacks:
        return
    cp = _optional_import_cupy()
    if cp is None:
        raise RuntimeError("CuPy became unavailable before staged output copy-back.")
    for host_array, device_array in copybacks:
        try:
            cp.asnumpy(device_array, out=host_array)
        except TypeError:
            host_array[...] = cp.asnumpy(device_array)


def launch(
    kernel_obj: MatCoreKernel,
    *arrays: Any,
    target: str = "x86-auto",
    quant: dict[str, Any] | None = None,
) -> Any:
    if not isinstance(kernel_obj, MatCoreKernel):
        raise TypeError("mc.launch expects a kernel object returned by @mc.kernel.")
    normalized_target = _normalize_target(target)
    runtime_ir = _build_runtime_ir(kernel_obj, arrays, quant)
    native = _get_native_module()
    runtime_arrays = arrays
    copybacks: list[tuple[Any, Any]] = []
    if _nvidia_target_requested(normalized_target):
        runtime_arrays, copybacks = _stage_nvidia_arrays(kernel_obj, arrays)
    native.compile_and_run(runtime_ir, normalized_target, *runtime_arrays)
    _copy_back_staged_arrays(copybacks)
    return None


class MatCoreNamespace:
    supported_targets = SUPPORTED_TARGETS
    supported_input_dtypes = SUPPORTED_INPUT_DTYPES
    kernel = staticmethod(kernel)
    launch = staticmethod(launch)
    asdtype = staticmethod(asdtype)
    load = staticmethod(load)
    store = staticmethod(store)
    matmul = staticmethod(matmul)


mc = MatCoreNamespace()
