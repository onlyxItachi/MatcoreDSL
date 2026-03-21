from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matcore import mc

BATCH = 4
M = 1024
K = 1024
N = 1024
STEPS = 8
TARGET_TIMEOUT_SECONDS = 45.0

TARGETS = [
    "x86-avx2",
    "x86-avx512",
    "amd-igpu",
    "nvidia-dgpu",
    "amd-npu",
]


@mc.kernel
def matmul_kernel(a, b, c):
    lhs = mc.load(a)
    rhs = mc.load(b)
    out = mc.matmul(lhs, rhs)
    mc.store(c, out)


def benchmark_target(target: str, a_seed: np.ndarray, b_seed: np.ndarray) -> None:
    current = a_seed.copy()
    scratch = np.empty((BATCH, M, N), dtype=np.float16)

    start = time.perf_counter()
    try:
        for _ in range(STEPS):
            for batch_idx in range(BATCH):
                mc.launch(
                    matmul_kernel,
                    current[batch_idx],
                    b_seed[batch_idx],
                    scratch[batch_idx],
                    target=target,
                )
            current, scratch = scratch, current

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        checksum = float(np.sum(current, dtype=np.float64))
        print(f"{target}: {elapsed_ms:.3f} ms (ok, checksum={checksum:.6e})")
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(f"{target}: FAILED after {elapsed_ms:.3f} ms ({type(exc).__name__}: {exc})")


def benchmark_target_isolated(target: str) -> None:
    start = time.perf_counter()
    command = [sys.executable, __file__, "--worker-target", target]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=TARGET_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"{target}: FAILED after {elapsed_ms:.3f} ms "
            f"(TimeoutExpired: exceeded {TARGET_TIMEOUT_SECONDS:.0f}s)"
        )
        return

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode == 0:
        if stdout:
            print(stdout)
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(
                f"{target}: FAILED after {elapsed_ms:.3f} ms "
                "(RuntimeError: backend worker exited without status output)"
            )
        return

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    detail = stderr.splitlines()[-1] if stderr else "worker exited without stderr"
    print(
        f"{target}: FAILED after {elapsed_ms:.3f} ms "
        f"(WorkerExit: returncode={result.returncode}, detail={detail})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-target", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(20260321)
    a_seed = rng.standard_normal((BATCH, M, K), dtype=np.float32).astype(np.float16)
    b_seed = rng.standard_normal((BATCH, K, N), dtype=np.float32).astype(np.float16)

    if args.worker_target is not None:
        benchmark_target(args.worker_target, a_seed, b_seed)
        return

    print(
        f"MatCore massive benchmark: batch={BATCH}, shape={M}x{K} @ {K}x{N}, "
        f"dtype=float16, steps={STEPS}"
    )
    for target in TARGETS:
        benchmark_target_isolated(target)


if __name__ == "__main__":
    main()
