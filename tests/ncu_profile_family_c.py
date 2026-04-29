"""Minimal Family C attention run for Nsight Compute profiling.

This script keeps Q/K/V on device, warms up JIT/cache first, then runs a small
NVTX-marked profiling region. It intentionally avoids PyTorch comparison work
inside the profiled path.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matcore as mc


@mc.fused
def family_c_attention(Q, K, V):
    return mc.softmax(Q @ K.T) @ V


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(123)
    q = (rng.standard_normal((args.m, args.d)).astype(np.float32) * 0.1)
    k = (rng.standard_normal((args.n, args.d)).astype(np.float32) * 0.1)
    v = (rng.standard_normal((args.n, args.d)).astype(np.float32) * 0.1)

    dq = mc.to_device(q)
    dk = mc.to_device(k)
    dv = mc.to_device(v)
    try:
        for _ in range(args.warmup):
            out = family_c_attention(dq, dk, dv)
            out.free()
        torch.cuda.synchronize()

        print(
            f"MATCORE_FAMILY_C_PROFILE_BEGIN shape={args.m}x{args.n}x{args.d} "
            f"profile_iters={args.profile_iters}"
        )
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("MATCORE_FAMILY_C_PROFILE")
        for _ in range(args.profile_iters):
            out = family_c_attention(dq, dk, dv)
            out.free()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()
        print("MATCORE_FAMILY_C_PROFILE_END")
    finally:
        dq.free()
        dk.free()
        dv.free()


if __name__ == "__main__":
    main()
