"""Minimal script for ncu profiling — 4096² fused activation."""
import sys, numpy as np
sys.path.insert(0, "/home/hamza-usta/MatcoreDSL")
import matcore as mc
import torch

N = 4096

# ── PyTorch (5 separate kernels) ──
tA = torch.randn(N, N, dtype=torch.float16, device="cuda")
tB = torch.randn(N, N, dtype=torch.float16, device="cuda")

def pytorch_unfused():
    x = torch.mm(tA, tB)
    x = torch.relu(x)
    x = torch.nn.functional.gelu(x)
    x = torch.tanh(x)
    x = torch.sigmoid(x)
    return x

# warmup pytorch
for _ in range(3):
    pytorch_unfused()
torch.cuda.synchronize()

# ── MatcoreDSL (fused) ──
@mc.fused
def chaos_fused(A, B):
    x = A @ B
    return mc.sigmoid(mc.tanh(mc.gelu(mc.relu(x))))

npA = tA.cpu().numpy()
npB = tB.cpu().numpy()
dA = mc.to_device(npA)
dB = mc.to_device(npB)

# JIT warmup
_ = chaos_fused(dA, dB)
torch.cuda.synchronize()

# ── Profiled runs ──
print("=== PYTORCH RUN ===")
torch.cuda.synchronize()
pytorch_unfused()
torch.cuda.synchronize()

print("=== MATCORE RUN ===")
torch.cuda.synchronize()
chaos_fused(dA, dB)
torch.cuda.synchronize()

print("=== DONE ===")
