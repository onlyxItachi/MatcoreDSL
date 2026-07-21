from __future__ import annotations

import ast
import re
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any

import numpy as np

_GRAPH_STATE = threading.local()


def _graph_stack() -> list["FusionGraph"]:
    stack = getattr(_GRAPH_STATE, "stack", None)
    if stack is None:
        stack = []
        _GRAPH_STATE.stack = stack
    return stack


def get_active_graph() -> "FusionGraph | None":
    stack = _graph_stack()
    return stack[-1] if stack else None


def _push_graph(graph_obj: "FusionGraph") -> None:
    _graph_stack().append(graph_obj)


def _pop_graph(expected: "FusionGraph") -> None:
    stack = _graph_stack()
    if not stack or stack[-1] is not expected:
        raise RuntimeError("Invalid graph context stack state.")
    stack.pop()


class FusionGraph:
    """Captures a sequence of kernel launches for fusion analysis."""

    def __init__(self):
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.analysis_report: dict[str, Any] | None = None
        self.execution_results: list[Any] = []

    def add_node(self, kernel_ir: dict[str, Any], arrays: tuple[Any, ...], **kwargs: Any) -> int:
        node_id = len(self.nodes)
        self.nodes.append(
            {
                "id": node_id,
                "kernel_ir": kernel_ir,
                "arrays": tuple(arrays),
                "kwargs": dict(kwargs),
            }
        )
        return node_id

    def _resolve_array_owner(self, value: Any) -> Any:
        current = getattr(value, "_array", value)
        seen: set[int] = set()
        while True:
            parent = getattr(current, "base", None)
            if parent is None:
                break
            parent_id = id(parent)
            if parent_id in seen:
                break
            seen.add(parent_id)
            current = parent
        return current

    def _array_identity(self, value: Any) -> tuple[str, int]:
        owner = self._resolve_array_owner(value)
        array_interface = getattr(owner, "__array_interface__", None)
        if isinstance(array_interface, dict):
            data_field = array_interface.get("data")
            if isinstance(data_field, tuple) and data_field:
                return ("ptr", int(data_field[0]))
        return ("obj", id(owner))

    def _arrays_may_alias(self, lhs: Any, rhs: Any) -> bool:
        try:
            lhs_arr = np.asarray(lhs)
            rhs_arr = np.asarray(rhs)
        except Exception:
            return False
        try:
            return bool(np.may_share_memory(lhs_arr, rhs_arr))
        except Exception:
            return False

    def _has_alias_overlap(
        self,
        lhs_ids: set[tuple[str, int]],
        rhs_ids: set[tuple[str, int]],
        lhs_arrays: list[Any],
        rhs_arrays: list[Any],
    ) -> bool:
        if lhs_ids & rhs_ids:
            return True
        for lhs in lhs_arrays:
            for rhs in rhs_arrays:
                if self._arrays_may_alias(lhs, rhs):
                    return True
        return False

    def _symbols_from_expression(
        self,
        expression: str,
        param_index: dict[str, int],
    ) -> set[str]:
        symbols: set[str] = set()
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError:
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression):
                if token in param_index:
                    symbols.add(token)
            return symbols

        for node in ast.walk(parsed):
            if isinstance(node, ast.Name) and node.id in param_index:
                symbols.add(node.id)
        return symbols

    def _infer_read_write_sets(
        self, node: dict[str, Any]
    ) -> tuple[set[tuple[str, int]], set[tuple[str, int]], list[Any], list[Any]]:
        kernel_ir = node["kernel_ir"]
        params = [str(param) for param in kernel_ir.get("params", [])]
        if not params:
            return set(), set(), [], []
        param_index = {symbol: idx for idx, symbol in enumerate(params)}
        arrays = node["arrays"]

        read_symbols: set[str] = set()
        write_symbols: set[str] = set()
        for op in kernel_ir.get("ops", []):
            if not isinstance(op, dict):
                continue
            op_name = op.get("op")
            tensor = op.get("tensor")
            if op_name == "store" and isinstance(tensor, str) and tensor in param_index:
                write_symbols.add(tensor)
            for value in op.values():
                if isinstance(value, str) and value in param_index:
                    if not (op_name == "store" and value == tensor):
                        read_symbols.add(value)
                elif isinstance(value, str):
                    read_symbols.update(self._symbols_from_expression(value, param_index))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item in param_index:
                            read_symbols.add(item)

        read_arrays: set[tuple[str, int]] = set()
        write_arrays: set[tuple[str, int]] = set()
        read_objects: list[Any] = []
        write_objects: list[Any] = []
        for symbol in read_symbols:
            idx = param_index[symbol]
            if idx < len(arrays):
                owner = self._resolve_array_owner(arrays[idx])
                read_arrays.add(self._array_identity(owner))
                read_objects.append(owner)
        for symbol in write_symbols:
            idx = param_index[symbol]
            if idx < len(arrays):
                owner = self._resolve_array_owner(arrays[idx])
                write_arrays.add(self._array_identity(owner))
                write_objects.append(owner)
        return read_arrays, write_arrays, read_objects, write_objects

    def _primary_op_kind(self, node: dict[str, Any]) -> str:
        kernel_ir = node["kernel_ir"]
        for op in kernel_ir.get("ops", []):
            if not isinstance(op, dict):
                continue
            op_name = op.get("op")
            if op_name in {"matmul", "elementwise", "transpose", "cast"}:
                return str(op_name)
        return "unknown"

    def _stable_topological_order(self, edges: list[dict[str, Any]]) -> list[int]:
        node_count = len(self.nodes)
        outgoing: dict[int, list[int]] = defaultdict(list)
        indegree = [0] * node_count
        for edge in edges:
            src = int(edge["from"])
            dst = int(edge["to"])
            outgoing[src].append(dst)
            indegree[dst] += 1
        for src in outgoing:
            outgoing[src].sort()

        ready = deque(idx for idx in range(node_count) if indegree[idx] == 0)
        order: list[int] = []
        while ready:
            current = ready.popleft()
            order.append(current)
            for neighbor in outgoing.get(current, []):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    ready.append(neighbor)

        if len(order) != node_count:
            raise RuntimeError("FusionGraph dependency cycle detected; cannot execute graph.")
        return order

    def _parallelizable_levels(self, edges: list[dict[str, Any]]) -> list[list[int]]:
        node_count = len(self.nodes)
        outgoing: dict[int, list[int]] = defaultdict(list)
        indegree = [0] * node_count
        for edge in edges:
            src = int(edge["from"])
            dst = int(edge["to"])
            outgoing[src].append(dst)
            indegree[dst] += 1
        for src in outgoing:
            outgoing[src].sort()

        frontier = [idx for idx in range(node_count) if indegree[idx] == 0]
        levels: list[list[int]] = []
        while frontier:
            level = sorted(frontier)
            levels.append(level)
            next_frontier: list[int] = []
            for src in level:
                for dst in outgoing.get(src, []):
                    indegree[dst] -= 1
                    if indegree[dst] == 0:
                        next_frontier.append(dst)
            frontier = next_frontier
        return levels

    def analyze_fusion_opportunities(self) -> dict[str, Any]:
        metadata: list[dict[str, Any]] = []
        for node in self.nodes:
            reads, writes, read_objects, write_objects = self._infer_read_write_sets(node)
            metadata.append(
                {
                    "reads": reads,
                    "writes": writes,
                    "read_objects": read_objects,
                    "write_objects": write_objects,
                    "kind": self._primary_op_kind(node),
                }
            )

        edges: list[dict[str, Any]] = []
        edge_lookup: set[tuple[int, int]] = set()
        node_count = len(self.nodes)
        for producer_idx in range(node_count):
            for consumer_idx in range(producer_idx + 1, node_count):
                producer = metadata[producer_idx]
                consumer = metadata[consumer_idx]
                hazards: list[str] = []
                if self._has_alias_overlap(
                    producer["writes"],
                    consumer["reads"],
                    producer["write_objects"],
                    consumer["read_objects"],
                ):
                    hazards.append("RAW")
                if self._has_alias_overlap(
                    producer["writes"],
                    consumer["writes"],
                    producer["write_objects"],
                    consumer["write_objects"],
                ):
                    hazards.append("WAW")
                if self._has_alias_overlap(
                    producer["reads"],
                    consumer["writes"],
                    producer["read_objects"],
                    consumer["write_objects"],
                ):
                    hazards.append("WAR")
                if not hazards:
                    continue
                key = (producer_idx, consumer_idx)
                if key in edge_lookup:
                    continue
                edge_lookup.add(key)
                edges.append({"from": producer_idx, "to": consumer_idx, "hazards": hazards})

        execution_order = self._stable_topological_order(edges)
        levels = self._parallelizable_levels(edges)
        in_degree = [0] * node_count
        out_degree = [0] * node_count
        for edge in edges:
            in_degree[int(edge["to"])] += 1
            out_degree[int(edge["from"])] += 1

        independent_nodes = [
            idx for idx in range(node_count) if in_degree[idx] == 0 and out_degree[idx] == 0
        ]
        fusible_pairs: list[dict[str, Any]] = []
        for edge in edges:
            if "RAW" not in edge["hazards"]:
                continue
            producer_idx = int(edge["from"])
            consumer_idx = int(edge["to"])
            producer_kind = metadata[producer_idx]["kind"]
            consumer_kind = metadata[consumer_idx]["kind"]
            pattern: str | None = None
            if producer_kind == "matmul" and consumer_kind == "elementwise":
                pattern = "matmul->elementwise"
            elif producer_kind == "matmul" and consumer_kind == "transpose":
                pattern = "matmul->transpose"
            elif producer_kind == "elementwise" and consumer_kind == "elementwise":
                pattern = "elementwise->elementwise"
            if pattern is None:
                continue
            fusible_pairs.append(
                {"producer": producer_idx, "consumer": consumer_idx, "pattern": pattern}
            )

        report = {
            "node_count": node_count,
            "edges": edges,
            "execution_order": execution_order,
            "parallelizable_levels": levels,
            "independent_nodes": independent_nodes,
            "fusible_pairs": fusible_pairs,
        }
        self.edges = edges
        self.analysis_report = report
        return report

    def execute(self) -> list[Any]:
        if self.analysis_report is None:
            self.analyze_fusion_opportunities()
        assert self.analysis_report is not None
        order = list(self.analysis_report.get("execution_order", []))
        from .frontend import (
            _launch_immediate,
            _validate_tensors,
            _validate_output,
        )

        self.execution_results = [None] * len(self.nodes)
        for node_index in order:
            node = self.nodes[node_index]
            kwargs = node.get("kwargs", {})
            target = kwargs.get("target")
            if not isinstance(target, str):
                raise RuntimeError("Captured graph node is missing normalized target metadata.")
            obs_options = kwargs.get("obs_options", {})
            if not isinstance(obs_options, dict):
                obs_options = {}
            if kwargs.get("validation_enabled"):
                _validate_tensors(node["arrays"], node["kernel_ir"], target)
            launch_result = _launch_immediate(
                node["kernel_ir"],
                target,
                node["arrays"],
                obs_options,
            )
            if kwargs.get("validation_enabled"):
                _validate_output(launch_result, node["arrays"], node["kernel_ir"])
            self.execution_results[node_index] = launch_result
        return [self.execution_results[idx] for idx in order]


@contextmanager
def graph():
    """mc.graph() context — captures launches for fusion."""
    if get_active_graph() is not None:
        raise RuntimeError("mc.graph() contexts cannot be nested")
    fusion_graph = FusionGraph()
    had_error = False
    _push_graph(fusion_graph)
    try:
        yield fusion_graph
    except BaseException:
        had_error = True
        raise
    finally:
        _pop_graph(fusion_graph)
    if not had_error:
        fusion_graph.analyze_fusion_opportunities()
        fusion_graph.execute()
