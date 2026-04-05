from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import re
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import configure, get_config, reset_config

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
_AMD_GFX_TOKEN = re.compile(r"^gfx[0-9a-z]+$")
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


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return str(node.value)
    if isinstance(node, ast.Str):
        return str(node.s)
    return None


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
            "GPU profiles may be requested as nvidia-dgpu:sm_90 or amd-igpu:gfx90a."
        )

    if not suffix:
        return base

    normalized_suffix = suffix.strip()
    if base == "nvidia-dgpu":
        match = _NVIDIA_SM_TOKEN.fullmatch(normalized_suffix)
        if match is None:
            raise ValueError(
                f"Unsupported NVIDIA target profile '{target}'. Use forms like 'nvidia-dgpu:sm_90'."
            )
        return f"{base}:sm_{match.group(1)}"
    if base == "amd-igpu":
        if _AMD_GFX_TOKEN.fullmatch(normalized_suffix) is None:
            raise ValueError(
                f"Unsupported AMD target profile '{target}'. Use forms like 'amd-igpu:gfx90a'."
            )
        return f"{base}:{normalized_suffix}"

    raise ValueError(
        f"Target profile suffixes are not supported for '{base}', got '{target}'."
    )


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
        self._namespace_roots: set[str] = {"mc"}
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
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute) and decorator.attr == "kernel":
                root = _expr_to_source(decorator.value)
                if root is not None:
                    self._namespace_roots.add(root)
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
            elif op["op"] != "assign":
                raise RuntimeError(
                    f"Standalone mc.{op['op']}(...) is not allowed in kernel bodies; "
                    "assign the result to a variable first."
                )

    def _value_to_op(self, value: ast.AST | None) -> dict[str, Any]:
        if value is None:
            return {"op": "assign", "output": None, "value": None}
        if isinstance(value, ast.Call):
            call_op = self._call_to_op(value)
            if call_op["op"] in {"load", "matmul", "transpose", "elementwise", "cast"}:
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

        if self._is_marker(fn, "transpose"):
            if len(args) != 1 or kwargs:
                raise RuntimeError("mc.transpose(x) expects exactly 1 positional argument.")
            return {
                "op": "transpose",
                "output": None,
                "input": _expr_to_source(args[0]),
            }

        if self._is_marker(fn, "add"):
            if len(args) != 2 or kwargs:
                raise RuntimeError("mc.add(x, y) expects exactly 2 positional arguments.")
            return {
                "op": "elementwise",
                "output": None,
                "kind": "add",
                "lhs": _expr_to_source(args[0]),
                "rhs": _expr_to_source(args[1]),
            }

        if self._is_marker(fn, "relu"):
            if len(args) != 1 or kwargs:
                raise RuntimeError("mc.relu(x) expects exactly 1 positional argument.")
            return {
                "op": "elementwise",
                "output": None,
                "kind": "relu",
                "lhs": _expr_to_source(args[0]),
                "rhs": None,
            }

        if self._is_marker(fn, "cast"):
            if len(args) < 1 or len(args) > 2:
                raise RuntimeError(
                    "mc.cast(x, 'float32') expects exactly 2 arguments (dtype can be keyword)."
                )
            if len(kwargs) > 1 or (len(kwargs) == 1 and "dtype" not in kwargs):
                raise RuntimeError("mc.cast only supports the 'dtype' keyword argument.")
            dtype_node = kwargs.get("dtype")
            if dtype_node is not None and len(args) > 1:
                raise RuntimeError("mc.cast dtype must be provided once (positional or keyword).")
            if dtype_node is None and len(args) == 2:
                dtype_node = args[1]
            target_dtype = _literal_str(dtype_node)
            if target_dtype is None:
                raise RuntimeError(
                    "mc.cast(...): target dtype must be a string literal, "
                    "for example mc.cast(x, 'float32')."
                )
            return {
                "op": "cast",
                "output": None,
                "input": _expr_to_source(args[0]),
                "target_dtype": target_dtype,
            }

        return {"op": "assign", "output": None, "value": _expr_to_source(call)}

    def _is_marker(self, fn: str, marker: str) -> bool:
        if fn == marker:
            return marker in {"load", "store", "matmul"}
        if not fn.endswith(f".{marker}"):
            return False
        root = fn[: -(len(marker) + 1)]
        return root in self._namespace_roots or root.startswith("mc")


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


def transpose(*_: Any, **__: Any) -> Any:
    _marker_error("transpose")


def add(*_: Any, **__: Any) -> Any:
    _marker_error("add")


def relu(*_: Any, **__: Any) -> Any:
    _marker_error("relu")


def cast(*_: Any, **__: Any) -> Any:
    _marker_error("cast")


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


def _build_observability_options(
    kernel_obj: MatCoreKernel,
    target: str,
    debug: bool,
    trace: str,
) -> dict[str, Any]:
    """Build observability keyword arguments for the native module."""
    env_debug_dir = os.environ.get("MATCORE_DEBUG_DIR", "")
    env_session = os.environ.get("MATCORE_DEBUG_SESSION", "")

    if not debug and trace == "none":
        return {}

    if env_session:
        session_id = env_session
    else:
        hasher = hashlib.sha256()
        hasher.update(kernel_obj.source.encode("utf-8"))
        hasher.update(target.encode("utf-8"))
        hasher.update(str(time.time_ns()).encode("utf-8"))
        session_id = hasher.hexdigest()[:16]

    if env_debug_dir:
        output_dir = os.path.join(env_debug_dir, session_id)
    else:
        output_dir = os.path.join(".matcore_debug", session_id)

    return {
        "debug": debug,
        "trace": trace,
        "session_id": session_id,
        "debug_dir": output_dir,
    }


def _report_trace_output(trace: str, obs_options: dict[str, Any]) -> None:
    if trace == "none":
        return
    debug_dir = str(obs_options.get("debug_dir", ""))
    if not debug_dir:
        return
    session_id = str(obs_options.get("session_id", "session"))
    metadata_path = os.path.join(debug_dir, "session_metadata.json")
    trace_path = ""
    if trace == "json":
        trace_path = os.path.join(debug_dir, f"{session_id}_trace.json")
    elif trace == "chrome":
        trace_path = os.path.join(debug_dir, "trace.json")

    duration_ms: int | None = None
    events: int | None = None
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            duration_ms = int(metadata.get("duration_ms", 0))
            events = int(metadata.get("trace_event_count", 0))
        except Exception:
            duration_ms = None
            events = None

    if duration_ms is not None and events is not None:
        print(
            f"MatCore trace: mode={trace} duration={duration_ms}ms events={events} dir={debug_dir}",
            file=sys.stderr,
        )
    else:
        print(f"MatCore trace: mode={trace} dir={debug_dir}", file=sys.stderr)
    if trace_path:
        print(f"MatCore trace file: {trace_path}", file=sys.stderr)


def _kernel_has_matmul(kernel_ir: dict[str, Any]) -> bool:
    return any(op.get("op") == "matmul" for op in kernel_ir.get("ops", []))


def _matmul_tensors(arrays: tuple[Any, ...], kernel_ir: dict[str, Any]) -> tuple[Any, Any, Any]:
    if not _kernel_has_matmul(kernel_ir):
        raise ValueError("Kernel IR does not contain a matmul operation.")
    params = [str(param) for param in kernel_ir.get("params", [])]
    symbol_to_tensor = {
        params[idx]: array for idx, array in enumerate(arrays) if idx < len(params)
    }
    ops = kernel_ir.get("ops", [])
    load_sources: dict[str, str] = {}
    matmul_op: dict[str, Any] | None = None
    for op in ops:
        if op.get("op") == "load":
            output = op.get("output")
            tensor = op.get("tensor")
            if isinstance(output, str) and isinstance(tensor, str):
                load_sources[output] = tensor
        elif matmul_op is None and op.get("op") == "matmul":
            matmul_op = op

    if matmul_op is None:
        raise ValueError("Kernel IR does not contain a matmul operation.")

    lhs_symbol = matmul_op.get("lhs")
    rhs_symbol = matmul_op.get("rhs")
    if isinstance(lhs_symbol, str):
        lhs_symbol = load_sources.get(lhs_symbol, lhs_symbol)
    if isinstance(rhs_symbol, str):
        rhs_symbol = load_sources.get(rhs_symbol, rhs_symbol)

    out_symbol: str | None = None
    matmul_value = matmul_op.get("output")
    for op in ops:
        if op.get("op") == "store" and op.get("value") == matmul_value:
            tensor = op.get("tensor")
            if isinstance(tensor, str):
                out_symbol = tensor
                break
    if out_symbol is None and isinstance(matmul_value, str) and matmul_value in symbol_to_tensor:
        out_symbol = matmul_value

    lhs = symbol_to_tensor.get(lhs_symbol) if isinstance(lhs_symbol, str) else None
    rhs = symbol_to_tensor.get(rhs_symbol) if isinstance(rhs_symbol, str) else None
    out = symbol_to_tensor.get(out_symbol) if isinstance(out_symbol, str) else None

    if lhs is None and len(arrays) >= 1:
        lhs = arrays[0]
    if rhs is None and len(arrays) >= 2:
        rhs = arrays[1]
    if out is None and len(arrays) >= 3:
        out = arrays[2]
    if lhs is None or rhs is None or out is None:
        raise ValueError("Matmul kernels require resolvable lhs, rhs, and output tensors.")
    return lhs, rhs, out


def _shape_of_rank2(array: Any, label: str) -> tuple[int, int]:
    shape = getattr(array, "shape", None)
    if shape is None:
        raise ValueError(f"{label} tensor does not expose a shape.")
    normalized_shape = tuple(int(dim) for dim in shape)
    if len(normalized_shape) != 2:
        raise ValueError(
            f"{label} tensor must be rank-2 for matmul, got shape {normalized_shape}."
        )
    return normalized_shape[0], normalized_shape[1]


def _warn_non_contiguous(array: Any, idx: int) -> None:
    flags = getattr(array, "flags", None)
    c_contiguous = bool(getattr(flags, "c_contiguous", False)) if flags is not None else False
    if not c_contiguous:
        warnings.warn(
            f"Tensor argument {idx} is not C-contiguous; this may hurt performance.",
            stacklevel=3,
        )


def _warn_if_non_finite(array: Any, label: str) -> None:
    if hasattr(array, "matcore_dtype"):
        return
    inspected = np.asarray(array)
    if not np.issubdtype(inspected.dtype, np.floating):
        return
    if not np.isfinite(inspected).all():
        warnings.warn(f"{label} contains NaN or Inf values.", stacklevel=3)


def _validate_matmul_dtypes(
    lhs_dtype: str, rhs_dtype: str, out_dtype: str, target: str
) -> None:
    if lhs_dtype != rhs_dtype:
        raise ValueError(
            f"Matmul dtype mismatch: lhs dtype '{lhs_dtype}' != rhs dtype '{rhs_dtype}'."
        )
    if lhs_dtype == "int32":
        raise ValueError("Matmul dtype mismatch: int32 inputs are not supported for matmul.")
    if lhs_dtype == "float8_e4m3fn":
        if out_dtype != "float32":
            raise ValueError(
                "float8_e4m3fn matmul requires float32 output/accumulation for MLIR 18.1.3 FP8 WGMMA."
            )
        if not target.startswith("nvidia-dgpu"):
            raise ValueError("float8_e4m3fn matmul is currently limited to nvidia-dgpu.")
        return
    allowed_outputs: dict[str, set[str]] = {
        "float32": {"float32"},
        "float16": {"float16", "float32"},
        "bfloat16": {"bfloat16", "float32"},
        "int8": {"int32"},
    }
    if lhs_dtype in allowed_outputs and out_dtype not in allowed_outputs[lhs_dtype]:
        expected = ", ".join(sorted(allowed_outputs[lhs_dtype]))
        raise ValueError(
            f"Matmul dtype mismatch: input dtype '{lhs_dtype}' requires output dtype "
            f"to be one of [{expected}], got '{out_dtype}'."
        )
    if lhs_dtype not in allowed_outputs:
        raise ValueError(
            f"Unsupported matmul dtype combination: lhs='{lhs_dtype}', rhs='{rhs_dtype}', out='{out_dtype}'."
        )


def _validate_tensors(arrays: tuple[Any, ...], kernel_ir: dict[str, Any], target: str) -> None:
    for idx, array in enumerate(arrays):
        _warn_non_contiguous(array, idx)
        _warn_if_non_finite(array, f"Tensor argument {idx}")

    if not _kernel_has_matmul(kernel_ir):
        return

    lhs, rhs, out = _matmul_tensors(arrays, kernel_ir)
    lhs_rows, lhs_cols = _shape_of_rank2(lhs, "LHS")
    rhs_rows, rhs_cols = _shape_of_rank2(rhs, "RHS")
    out_rows, out_cols = _shape_of_rank2(out, "Output")

    if lhs_cols != rhs_rows:
        raise ValueError(
            "Matmul shape mismatch: lhs.shape[1] must equal rhs.shape[0], "
            f"got lhs.shape={getattr(lhs, 'shape', None)} and rhs.shape={getattr(rhs, 'shape', None)}."
        )
    expected_out_shape = (lhs_rows, rhs_cols)
    actual_out_shape = (out_rows, out_cols)
    if actual_out_shape != expected_out_shape:
        raise ValueError(
            "Output shape mismatch: expected output shape "
            f"{expected_out_shape} from lhs/rhs, got {actual_out_shape}."
        )

    lhs_dtype = _logical_dtype_name(lhs, 0)
    rhs_dtype = _logical_dtype_name(rhs, 1)
    out_dtype = _logical_dtype_name(out, 2)
    _validate_matmul_dtypes(lhs_dtype, rhs_dtype, out_dtype, target)


def _resolve_output_tensor(arrays: tuple[Any, ...], kernel_ir: dict[str, Any]) -> Any:
    params = [str(param) for param in kernel_ir.get("params", [])]
    symbol_to_tensor = {
        params[idx]: array for idx, array in enumerate(arrays) if idx < len(params)
    }
    for op in reversed(kernel_ir.get("ops", [])):
        if op.get("op") != "store":
            continue
        tensor_symbol = op.get("tensor")
        if isinstance(tensor_symbol, str) and tensor_symbol in symbol_to_tensor:
            return symbol_to_tensor[tensor_symbol]
    if arrays:
        return arrays[-1]
    raise ValueError("No output tensor could be resolved for validation.")


def _validate_output(result: Any, arrays: tuple[Any, ...], kernel_ir: dict[str, Any]) -> None:
    del result
    output_tensor = _resolve_output_tensor(arrays, kernel_ir)
    if hasattr(output_tensor, "matcore_dtype"):
        return
    output_arr = np.asarray(output_tensor)
    if np.issubdtype(output_arr.dtype, np.floating) and not np.isfinite(output_arr).all():
        warnings.warn("Output contains NaN or Inf values.", stacklevel=3)
    if output_arr.dtype == np.dtype(np.float32) and output_arr.size > 0:
        finite_mask = np.isfinite(output_arr)
        if finite_mask.any():
            max_abs = float(np.max(np.abs(output_arr[finite_mask])))
            if max_abs > 1e15:
                warnings.warn(
                    f"Output magnitude is unusually large for float32 (max abs={max_abs:.3e}).",
                    stacklevel=3,
                )


def _prepare_launch(
    kernel_obj: MatCoreKernel,
    arrays: tuple[Any, ...],
    *,
    target: str | None,
    quant: dict[str, Any] | None,
    debug: bool | None,
    trace: str | None,
    validate: bool | None,
) -> tuple[dict[str, Any], str, dict[str, Any], bool]:
    from .config import resolve_launch_options

    resolved = resolve_launch_options(
        target=target, debug=debug, trace=trace, validate=validate
    )
    normalized_target = _normalize_target(resolved["target"])
    runtime_ir = _build_runtime_ir(kernel_obj, arrays, quant)
    obs_options = _build_observability_options(
        kernel_obj, normalized_target, resolved["debug"], resolved["trace"]
    )
    return runtime_ir, normalized_target, obs_options, bool(resolved["validate"])


def _launch_immediate(
    runtime_ir: dict[str, Any],
    normalized_target: str,
    arrays: tuple[Any, ...],
    obs_options: dict[str, Any],
) -> Any:
    native = _get_native_module()
    try:
        return native.compile_and_run(runtime_ir, normalized_target, *arrays, **obs_options)
    finally:
        if obs_options:
            _report_trace_output(str(obs_options.get("trace", "none")), obs_options)


def launch(
    kernel_obj: MatCoreKernel,
    *arrays: Any,
    target: str | None = None,
    quant: dict[str, Any] | None = None,
    debug: bool | None = None,
    trace: str | None = None,
    validate: bool | None = None,
) -> Any:
    if not isinstance(kernel_obj, MatCoreKernel):
        raise TypeError("mc.launch expects a kernel object returned by @mc.kernel.")
    runtime_ir, normalized_target, obs_options, validation_enabled = _prepare_launch(
        kernel_obj,
        arrays,
        target=target,
        quant=quant,
        debug=debug,
        trace=trace,
        validate=validate,
    )
    if validation_enabled:
        _validate_tensors(arrays, runtime_ir, normalized_target)
    from .graph import get_active_graph

    active_graph = get_active_graph()
    if active_graph is not None:
        active_graph.add_node(
            runtime_ir,
            arrays,
            target=normalized_target,
            obs_options=obs_options,
            validation_enabled=validation_enabled,
        )
        return None
    launch_result = _launch_immediate(runtime_ir, normalized_target, arrays, obs_options)
    if validation_enabled:
        _validate_output(launch_result, arrays, runtime_ir)
    return launch_result


def _graph_context_manager() -> Any:
    from .graph import graph as graph_context

    return graph_context()


class MatCoreNamespace:
    supported_targets = SUPPORTED_TARGETS
    supported_input_dtypes = SUPPORTED_INPUT_DTYPES
    kernel = staticmethod(kernel)
    launch = staticmethod(launch)
    asdtype = staticmethod(asdtype)
    load = staticmethod(load)
    store = staticmethod(store)
    matmul = staticmethod(matmul)
    transpose = staticmethod(transpose)
    add = staticmethod(add)
    relu = staticmethod(relu)
    cast = staticmethod(cast)
    graph = staticmethod(_graph_context_manager)
    config = staticmethod(configure)
    get_config = staticmethod(get_config)
    reset_config = staticmethod(reset_config)


mc = MatCoreNamespace()
