from __future__ import annotations

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import matcore as mc
from matcore import frontend
from matcore.device_tensor import DeviceTensor


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_namespace_qualified_kernel_helpers() -> None:
    alias = mc

    @alias.kernel
    def helper_kernel(A, B, C):
        lhs = alias.load(A)
        rhs = alias.load(B)
        summed = alias.add(lhs, rhs)
        casted = alias.cast(summed, dtype="float16")
        activated = alias.softmax(casted)
        alias.store(C, activated)

    ops = helper_kernel.ir["ops"]
    assert [op["op"] for op in ops] == [
        "load",
        "load",
        "elementwise",
        "cast",
        "elementwise",
        "store",
    ]
    assert ops[2]["kind"] == "add"
    assert ops[3]["target_dtype"] == "float16"
    assert ops[4]["kind"] == "softmax"


def test_bare_kernel_helper_rejected() -> None:
    try:

        @mc.kernel
        def bad_kernel(A, B):
            tmp = relu(A)
            mc.store(B, tmp)

    except RuntimeError as exc:
        assert "namespace-qualified" in str(exc)
    else:
        raise AssertionError("Expected bare kernel helper use to raise RuntimeError")


def test_bare_kernel_load_rejected() -> None:
    try:

        @mc.kernel
        def bad_load(A, B):
            tmp = load(A)
            mc.store(B, tmp)

    except RuntimeError as exc:
        assert "namespace-qualified" in str(exc)
    else:
        raise AssertionError("Expected bare load() to raise RuntimeError")


def test_standalone_bare_kernel_helper_not_silent() -> None:
    try:

        @mc.kernel
        def bad_expr(A, B):
            relu(A)
            mc.store(B, A)

    except RuntimeError as exc:
        assert "namespace-qualified" in str(exc)
    else:
        raise AssertionError("Expected standalone bare helper to raise RuntimeError")


def test_get_config_applies_env_overrides() -> None:
    snapshot = {
        "MATCORE_TARGET": os.environ.get("MATCORE_TARGET"),
        "MATCORE_DEBUG": os.environ.get("MATCORE_DEBUG"),
    }
    mc.reset_config()
    os.environ["MATCORE_TARGET"] = "nvidia-dgpu:sm_90"
    os.environ["MATCORE_DEBUG"] = "true"
    try:
        cfg = mc.get_config()
        assert cfg.default_target == "nvidia-dgpu:sm_90"
        assert cfg.debug is True
    finally:
        _restore_env(snapshot)
        mc.reset_config()


def test_fused_uses_effective_target_from_env() -> None:
    snapshot = {"MATCORE_TARGET": os.environ.get("MATCORE_TARGET")}
    native_calls: dict[str, str] = {}

    class DummyNative:
        @staticmethod
        def compile_and_run(kernel_ir, target, *args, **kwargs):
            native_calls["target"] = target
            native_calls["kernel_name"] = kernel_ir["kernel_name"]
            return None

    original_get_native_module = frontend._get_native_module
    frontend._get_native_module = lambda: DummyNative()
    os.environ["MATCORE_TARGET"] = "x86-avx2"
    try:

        @mc.fused
        def gemm_relu(A, B):
            return mc.relu(A @ B)

        out = gemm_relu(
            np.zeros((4, 4), dtype=np.float32),
            np.zeros((4, 4), dtype=np.float32),
        )
        assert native_calls["target"] == "x86-avx2"
        assert out.shape == (4, 4)
    finally:
        frontend._get_native_module = original_get_native_module
        _restore_env(snapshot)


def test_fused_rejects_effective_non_nvidia_target_for_device_tensors() -> None:
    snapshot = {"MATCORE_TARGET": os.environ.get("MATCORE_TARGET")}
    os.environ["MATCORE_TARGET"] = "x86-auto"
    try:

        @mc.fused
        def gemm_relu(A, B):
            return mc.relu(A @ B)

        dA = DeviceTensor(object(), (4, 4), "float16", (4, 1), 32)
        dB = DeviceTensor(object(), (4, 4), "float16", (4, 1), 32)
        try:
            gemm_relu(dA, dB)
        except ValueError as exc:
            assert "DeviceTensors are only supported" in str(exc)
        else:
            raise AssertionError("Expected non-NVIDIA effective target rejection")
    finally:
        _restore_env(snapshot)


def test_to_device_rejects_non_nvidia_target_before_runtime_import() -> None:
    try:
        mc.to_device(np.zeros((2, 2), dtype=np.float32), target="x86-auto")
    except ValueError as exc:
        assert "nvidia-dgpu" in str(exc)
    else:
        raise AssertionError("Expected mc.to_device to reject non-NVIDIA targets")


def test_to_device_rejects_quantized_int8_wrappers() -> None:
    quantized = mc.asdtype(
        np.zeros((2, 2), dtype=np.int8),
        "int8",
        scale=0.5,
        zero_point=2,
    )
    try:
        mc.to_device(quantized)
    except ValueError as exc:
        assert "does not yet support quantized int8" in str(exc)
    else:
        raise AssertionError("Expected mc.to_device to reject quantized int8 wrappers")


def test_cuda_array_interface_counts_as_device_resident() -> None:
    class CudaArray:
        shape = (2, 2)
        strides = (8, 4)
        flags = type("Flags", (), {"c_contiguous": True})()
        __cuda_array_interface__ = {
            "shape": shape,
            "strides": strides,
            "typestr": "<f4",
            "data": (1, False),
            "version": 3,
        }

    device = CudaArray()
    wrapped = mc.asdtype(device, "float32")
    assert frontend._analyze_tensor_residency((device, wrapped)) == (True, False)
    assert frontend._analyze_tensor_residency((device, np.zeros((2, 2)))) == (
        True,
        True,
    )


def test_invalid_cuda_array_interface_does_not_spoof_residency() -> None:
    class InvalidCudaArray:
        __cuda_array_interface__ = {"data": "not-a-pointer-tuple", "version": 3}

    assert frontend._analyze_tensor_residency((InvalidCudaArray(),)) == (
        False,
        True,
    )


def test_device_validation_does_not_force_numpy_conversion() -> None:
    class CudaArray:
        __cuda_array_interface__ = {
            "shape": (2, 2),
            "strides": (8, 4),
            "typestr": "<f4",
            "data": (1, False),
            "version": 3,
        }

        def __array__(self):
            raise AssertionError("device validation must not copy through NumPy")

    device = CudaArray()
    frontend._warn_if_non_finite(device, "device")
    frontend._validate_output(None, (device,), {"params": ["out"], "ops": []})


@mc.kernel
def _int8_kernel(A, B, C):
    lhs = mc.load(A)
    rhs = mc.load(B)
    out = mc.matmul(lhs, rhs)
    mc.store(C, out)


def test_create_plan_rejects_quantized_device_tensors_before_native_dispatch() -> None:
    dA = DeviceTensor(object(), (2, 2), "int8", (2, 1), 4)
    dB = DeviceTensor(object(), (2, 2), "int8", (2, 1), 4)
    dC = DeviceTensor(object(), (2, 2), "int32", (2, 1), 16)
    dA.matcore_quant_enabled = True
    dA.matcore_scale = 0.5
    dA.matcore_zero_point = 2
    try:
        mc.create_plan(_int8_kernel, dA, dB, dC, target="nvidia-dgpu:sm_89")
    except ValueError as exc:
        assert "Quantized DeviceTensors are not yet supported" in str(exc)
    else:
        raise AssertionError("Expected mc.create_plan to reject quantized DeviceTensor input")


def test_native_create_plan_rejects_quantized_device_tensors() -> None:
    try:
        dA = mc.to_device(np.zeros((2, 2), dtype=np.int8))
        dB = mc.to_device(np.zeros((2, 2), dtype=np.int8))
        dC = mc.to_device(np.zeros((2, 2), dtype=np.int32))
    except Exception:
        return

    try:
        dA.matcore_quant_enabled = True
        dA.matcore_scale = 0.5
        dA.matcore_zero_point = 2
        native = frontend._get_native_module()
        try:
            native.create_plan(_int8_kernel.ir, "nvidia-dgpu:sm_89", dA, dB, dC, graph_mode=False)
        except RuntimeError as exc:
            assert "Quantized DeviceTensor inputs are not yet supported" in str(exc)
        else:
            raise AssertionError("Expected native.create_plan to reject quantized DeviceTensor input")
    finally:
        dA.free()
        dB.free()
        dC.free()


def main() -> None:
    test_namespace_qualified_kernel_helpers()
    test_bare_kernel_helper_rejected()
    test_bare_kernel_load_rejected()
    test_standalone_bare_kernel_helper_not_silent()
    test_get_config_applies_env_overrides()
    test_fused_uses_effective_target_from_env()
    test_fused_rejects_effective_non_nvidia_target_for_device_tensors()
    test_to_device_rejects_non_nvidia_target_before_runtime_import()
    test_to_device_rejects_quantized_int8_wrappers()
    test_cuda_array_interface_counts_as_device_resident()
    test_invalid_cuda_array_interface_does_not_spoof_residency()
    test_device_validation_does_not_force_numpy_conversion()
    test_create_plan_rejects_quantized_device_tensors_before_native_dispatch()
    test_native_create_plan_rejects_quantized_device_tensors()
    print("Frontend contract tests passed")


if __name__ == "__main__":
    main()
