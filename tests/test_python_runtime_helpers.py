from __future__ import annotations

import ctypes
import importlib
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cache_module = importlib.import_module("matcore.cache")
config_module = importlib.import_module("matcore.config")
device_tensor_module = importlib.import_module("matcore.device_tensor")
frontend_module = importlib.import_module("matcore.frontend")
graph_module = importlib.import_module("matcore.graph")


class CacheModuleTests(unittest.TestCase):
    def test_cache_info_summary_and_clear_handle_multiple_cache_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = pathlib.Path(tmpdir)
            exact_cache = base / ".matcore_cache"
            prefixed_cache = base / ".matcore_cache_review"
            exact_cache.mkdir()
            prefixed_cache.mkdir()

            artifact_a = exact_cache / "artifact-a"
            artifact_a.mkdir()
            (artifact_a / "kernel.bin").write_bytes(b"abc")
            (artifact_a / "metadata.json").write_text('{"target": "x86-auto"}')

            artifact_b = prefixed_cache / "artifact-b"
            artifact_b.mkdir()
            (artifact_b / "artifact_payload.txt").write_text("payload")
            (artifact_b / "metadata.json").write_text("{invalid json")

            info = cache_module.cache_info(tmpdir, include_prefixed=True)
            self.assertEqual(len(info), 2)
            self.assertEqual({entry["artifact_count"] for entry in info}, {1})

            metadata_by_dir = {
                pathlib.Path(entry["cache_dir"]).name: entry["artifact_dirs"][0]["metadata"]
                for entry in info
            }
            self.assertEqual(metadata_by_dir[".matcore_cache"], {"target": "x86-auto"})
            self.assertIsNone(metadata_by_dir[".matcore_cache_review"])

            summary = cache_module.cache_summary(tmpdir, include_prefixed=True)
            self.assertIn("2 cache dir(s)", summary)
            self.assertIn(".matcore_cache: 1 artifact(s)", summary)
            self.assertIn(".matcore_cache_review: 1 artifact(s)", summary)

            cleared = cache_module.cache_clear(tmpdir, include_prefixed=True)
            self.assertEqual(cleared, 2)
            self.assertEqual(cache_module.cache_info(tmpdir, include_prefixed=True), [])

    def test_cache_uses_env_override_for_default_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_cache = pathlib.Path(tmpdir) / ".matcore_cache"
            env_cache.mkdir()
            artifact = env_cache / "artifact-a"
            artifact.mkdir()
            (artifact / "metadata.json").write_text('{"via": "env"}')
            with mock.patch.dict(os.environ, {"MATCORE_CACHE_DIR": str(env_cache)}, clear=False):
                info = cache_module.cache_info()
            self.assertEqual(len(info), 1)
            self.assertEqual(info[0]["cache_dir"], str(env_cache))
            self.assertEqual(info[0]["artifact_dirs"][0]["metadata"], {"via": "env"})

    def test_cache_summary_reports_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = cache_module.cache_summary(tmpdir)
        self.assertEqual(summary, "No MatCore cache directories found.")


class ConfigModuleTests(unittest.TestCase):
    def tearDown(self) -> None:
        config_module.reset_config()

    def test_env_bool_parses_common_values(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "BOOL_TRUE": " yes ",
                "BOOL_FALSE": "OFF",
                "BOOL_INVALID": "sometimes",
            },
            clear=False,
        ):
            self.assertTrue(config_module._env_bool("BOOL_TRUE"))
            self.assertFalse(config_module._env_bool("BOOL_FALSE"))
            self.assertIsNone(config_module._env_bool("BOOL_INVALID"))
            self.assertIsNone(config_module._env_bool("BOOL_MISSING"))

    def test_configure_get_and_reset_round_trip(self) -> None:
        config_module.configure(default_target="x86-avx512", debug=True, trace="ir")
        snapshot = config_module.get_config()
        snapshot.trace = "mutated"
        self.assertEqual(config_module.get_config().trace, "ir")

        config_module.reset_config()
        reset_cfg = config_module.get_config()
        self.assertEqual(reset_cfg.default_target, "x86-auto")
        self.assertFalse(reset_cfg.debug)

    def test_configure_rejects_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown config key 'bogus'"):
            config_module.configure(bogus=True)

    def test_resolve_launch_options_honors_env_overrides(self) -> None:
        config_module.configure(
            default_target="x86-avx2",
            debug=False,
            trace="graph",
            validate=False,
        )
        with mock.patch.dict(
            os.environ,
            {
                "MATCORE_TARGET": "nvidia-dgpu:sm_90",
                "MATCORE_DEBUG": "true",
                "MATCORE_TRACE": "IR",
                "MATCORE_VALIDATE": "1",
            },
            clear=False,
        ):
            options = config_module.resolve_launch_options(
                target="x86-auto",
                debug=False,
                trace="none",
                validate=False,
            )

        self.assertEqual(
            options,
            {
                "target": "nvidia-dgpu:sm_90",
                "debug": True,
                "trace": "ir",
                "validate": True,
            },
        )


class FusionGraphTests(unittest.TestCase):
    def test_analyze_fusion_opportunities_tracks_dependencies_and_parallelism(self) -> None:
        graph = graph_module.FusionGraph()
        lhs = np.arange(6, dtype=np.float32).reshape(2, 3)
        rhs = np.arange(6, dtype=np.float32).reshape(3, 2)
        tmp = np.zeros((2, 2), dtype=np.float32)
        out = np.zeros((2, 2), dtype=np.float32)
        side_in = np.arange(4, dtype=np.float32).reshape(2, 2)
        side_out = np.zeros((2, 2), dtype=np.float32)

        graph.add_node(
            {
                "params": ["lhs", "rhs", "tmp"],
                "ops": [
                    {"op": "matmul", "lhs": "lhs", "rhs": "rhs", "output": "acc"},
                    {"op": "store", "tensor": "tmp", "value": "acc"},
                ],
            },
            (lhs, rhs, tmp),
            target="x86-auto",
        )
        graph.add_node(
            {
                "params": ["inp", "out"],
                "ops": [
                    {"op": "elementwise", "inputs": ["inp"], "fn": "relu", "output": "val"},
                    {"op": "store", "tensor": "out", "value": "val"},
                ],
            },
            (tmp, out),
            target="x86-auto",
        )
        graph.add_node(
            {
                "params": ["inp", "out"],
                "ops": [
                    {"op": "elementwise", "inputs": ["inp"], "fn": "exp", "output": "val"},
                    {"op": "store", "tensor": "out", "value": "val"},
                ],
            },
            (side_in, side_out),
            target="x86-auto",
        )

        report = graph.analyze_fusion_opportunities()

        self.assertEqual(report["node_count"], 3)
        self.assertEqual(report["execution_order"], [0, 2, 1])
        self.assertEqual(report["parallelizable_levels"], [[0, 2], [1]])
        self.assertEqual(report["independent_nodes"], [2])
        self.assertEqual(
            report["edges"],
            [{"from": 0, "to": 1, "hazards": ["RAW"]}],
        )
        self.assertEqual(
            report["fusible_pairs"],
            [{"producer": 0, "consumer": 1, "pattern": "matmul->elementwise"}],
        )

    def test_execute_uses_frontend_hooks_and_validation_flags(self) -> None:
        graph = graph_module.FusionGraph()
        arrays = (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
        kernel_ir = {
            "params": ["inp", "out"],
            "ops": [
                {"op": "elementwise", "inputs": ["inp"], "fn": "relu", "output": "val"},
                {"op": "store", "tensor": "out", "value": "val"},
            ],
        }
        graph.add_node(
            kernel_ir,
            arrays,
            target="x86-auto",
            obs_options={"trace": "none"},
            validation_enabled=True,
        )
        graph.analysis_report = {"execution_order": [0]}

        with (
            mock.patch.object(frontend_module, "_launch_immediate", return_value="launch-result") as launch,
            mock.patch.object(frontend_module, "_validate_tensors") as validate_tensors,
            mock.patch.object(frontend_module, "_validate_output") as validate_output,
        ):
            result = graph.execute()

        self.assertEqual(result, ["launch-result"])
        validate_tensors.assert_called_once_with(arrays, kernel_ir, "x86-auto")
        launch.assert_called_once_with(kernel_ir, "x86-auto", arrays, {"trace": "none"})
        validate_output.assert_called_once_with("launch-result", arrays, kernel_ir)

    def test_execute_requires_normalized_target_metadata(self) -> None:
        graph = graph_module.FusionGraph()
        graph.add_node({"params": [], "ops": []}, tuple(), validation_enabled=False)
        graph.analysis_report = {"execution_order": [0]}

        with self.assertRaisesRegex(RuntimeError, "missing normalized target metadata"):
            graph.execute()

    def test_graph_context_executes_on_success_and_cleans_up_on_failure(self) -> None:
        with (
            mock.patch.object(graph_module.FusionGraph, "analyze_fusion_opportunities") as analyze,
            mock.patch.object(graph_module.FusionGraph, "execute") as execute,
        ):
            with graph_module.graph() as active_graph:
                self.assertIs(graph_module.get_active_graph(), active_graph)

            analyze.assert_called_once()
            execute.assert_called_once()
            self.assertIsNone(graph_module.get_active_graph())

        with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
            with graph_module.graph():
                with graph_module.graph():
                    pass
        self.assertIsNone(graph_module.get_active_graph())


class _FakeNativeModule:
    def __init__(self) -> None:
        self.alloc_sizes: list[int] = []
        self.uploads: list[tuple[object, int]] = []
        self.zeroed: list[object] = []
        self.freed: list[object] = []
        self._buffers: dict[object, bytes] = {}

    def matcore_device_alloc(self, size_bytes: int) -> object:
        handle = f"handle-{len(self.alloc_sizes)}"
        self.alloc_sizes.append(size_bytes)
        return handle

    def matcore_device_upload(self, handle: object, host_ptr: int, size_bytes: int) -> None:
        self.uploads.append((handle, size_bytes))
        self._buffers[handle] = ctypes.string_at(host_ptr, size_bytes)

    def matcore_device_download(self, host_ptr: int, handle: object, size_bytes: int) -> None:
        ctypes.memmove(host_ptr, self._buffers[handle], size_bytes)

    def matcore_device_zero(self, handle: object) -> None:
        self.zeroed.append(handle)
        self._buffers[handle] = b"\x00" * len(self._buffers[handle])

    def matcore_device_free(self, handle: object) -> None:
        self.freed.append(handle)


class DeviceTensorTests(unittest.TestCase):
    def test_to_device_to_host_zero_and_free_use_native_runtime(self) -> None:
        fake_native = _FakeNativeModule()
        host = np.arange(6, dtype=np.float16).reshape(2, 3)

        with mock.patch.object(device_tensor_module, "_get_native_module", return_value=fake_native):
            tensor = device_tensor_module.to_device(host)
            self.assertEqual(tensor.shape, (2, 3))
            self.assertEqual(tensor.dtype, "float16")
            self.assertEqual(tensor.strides, (3, 1))
            self.assertEqual(tensor.size_bytes, host.nbytes)
            self.assertEqual(tensor.device_ptr, "handle-0")
            np.testing.assert_array_equal(tensor.to_host(), host)

            tensor.zero_()
            np.testing.assert_array_equal(tensor.to_host(), np.zeros_like(host))
            tensor.free()
            tensor.free()

        self.assertEqual(fake_native.alloc_sizes, [host.nbytes])
        self.assertEqual(fake_native.uploads, [("handle-0", host.nbytes)])
        self.assertEqual(fake_native.zeroed, ["handle-0"])
        self.assertEqual(fake_native.freed, ["handle-0"])

        with self.assertRaisesRegex(RuntimeError, "already been freed"):
            tensor.zero_()
        with self.assertRaisesRegex(RuntimeError, "already been freed"):
            _ = tensor.device_ptr

    def test_public_apis_raise_for_unsupported_dtypes(self) -> None:
        fake_native = _FakeNativeModule()
        with mock.patch.object(device_tensor_module, "_get_native_module", return_value=fake_native):
            with self.assertRaisesRegex(ValueError, "Unsupported numpy dtype"):
                device_tensor_module.to_device(np.array([1 + 2j], dtype=np.complex64))

            tensor = device_tensor_module.DeviceTensor("raw-handle", (1,), "complex64", (1,), 8)
            with self.assertRaisesRegex(ValueError, "Unsupported MatCore dtype"):
                tensor.to_host()


if __name__ == "__main__":
    unittest.main()
