from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import matcore as mc
from matcore.validation import probe_backend_availability


@mc.kernel
def matmul(A, B, C):
    lhs = mc.load(A)
    rhs = mc.load(B)
    out = mc.matmul(lhs, rhs)
    mc.store(C, out)


def _expected(lhs: np.ndarray, rhs: np.ndarray, zero_point: int) -> np.ndarray:
    lhs_i32 = lhs.astype(np.int32) - zero_point
    rhs_i32 = rhs.astype(np.int32) - zero_point
    return lhs_i32 @ rhs_i32


def test_quantized_plan_round_trip() -> None:
    ok, _ = probe_backend_availability("x86-avx512")
    if not ok:
        return

    lhs_raw = np.array([[4, -3, 2], [1, 0, -2]], dtype=np.int8)
    rhs_raw = np.array([[2, -1], [3, 4], [-2, 1]], dtype=np.int8)
    lhs = mc.asdtype(lhs_raw, "int8", scale=0.5, zero_point=2)
    rhs = mc.asdtype(rhs_raw, "int8", scale=0.5, zero_point=2)
    out = np.zeros((2, 2), dtype=np.int32)

    plan = mc.create_plan(matmul, lhs, rhs, out, target="x86-avx512")
    mc.execute_plan(plan, lhs, rhs, out)

    np.testing.assert_array_equal(out, _expected(lhs_raw, rhs_raw, zero_point=2))


def test_quantized_plan_rejects_mismatched_metadata() -> None:
    ok, _ = probe_backend_availability("x86-avx512")
    if not ok:
        return

    lhs_raw = np.array([[4, -3, 2], [1, 0, -2]], dtype=np.int8)
    rhs_raw = np.array([[2, -1], [3, 4], [-2, 1]], dtype=np.int8)
    lhs = mc.asdtype(lhs_raw, "int8", scale=0.5, zero_point=2)
    rhs = mc.asdtype(rhs_raw, "int8", scale=0.5, zero_point=2)
    out = np.zeros((2, 2), dtype=np.int32)

    plan = mc.create_plan(matmul, lhs, rhs, out, target="x86-avx512")
    mismatched_lhs = mc.asdtype(lhs_raw.copy(), "int8", scale=0.25, zero_point=1)

    try:
        mc.execute_plan(plan, mismatched_lhs, rhs, out)
    except RuntimeError as exc:
        assert "quantization mismatch" in str(exc)
    else:
        raise AssertionError("Expected execute_plan() to reject mismatched quantization metadata")


def main() -> None:
    test_quantized_plan_round_trip()
    test_quantized_plan_rejects_mismatched_metadata()
    print("Plan quantization tests passed")


if __name__ == "__main__":
    main()
