import pathlib, sys
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from matcore import matmul_kernel, check_matmul_correctness

lhs, rhs = np.random.default_rng(42).standard_normal((16,16), dtype=np.float32).astype(np.float16), \
           np.random.default_rng(43).standard_normal((16,8), dtype=np.float32).astype(np.float16)
out = np.zeros((16,8), dtype=np.float16)
try:
    report = check_matmul_correctness(
        matmul_kernel, lhs, rhs, target="nvidia-dgpu:sm_89", out=out, atol=1e-2, rtol=1e-1
    )
    print(f"PASS: max_abs={report.max_abs_error:.6e} elapsed={report.elapsed_ms:.3f}ms")
except Exception as e:
    print(f"ERROR: {e}")
