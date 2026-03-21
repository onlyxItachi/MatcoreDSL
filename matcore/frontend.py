from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any

_NATIVE_MODULE_NAME = "_matcore_native"


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


def launch(kernel_obj: MatCoreKernel, *arrays: Any, target: str = "x86-auto") -> Any:
    if not isinstance(kernel_obj, MatCoreKernel):
        raise TypeError("mc.launch expects a kernel object returned by @mc.kernel.")
    native = _get_native_module()
    return native.compile_and_run(kernel_obj.ir, target, *arrays)


class MatCoreNamespace:
    kernel = staticmethod(kernel)
    launch = staticmethod(launch)
    load = staticmethod(load)
    store = staticmethod(store)
    matmul = staticmethod(matmul)


mc = MatCoreNamespace()
