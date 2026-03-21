from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from matcore import mc


@mc.kernel
def matmul_kernel(a, b, c):
    lhs = mc.load(a)
    rhs = mc.load(b)
    product = mc.matmul(lhs, rhs)
    mc.store(c, product)


def main() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    c = np.zeros((2, 2), dtype=np.float32)

    c_ptr_before = c.__array_interface__["data"][0]
    mc.launch(matmul_kernel, a, b, c, target="x86-auto")
    c_ptr_after = c.__array_interface__["data"][0]

    expected = a @ b
    np.testing.assert_allclose(c, expected)
    assert c_ptr_before == c_ptr_after
    assert c.flags.c_contiguous


if __name__ == "__main__":
    main()
