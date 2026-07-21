"""Dump the IR at the NVVM failure point for multi-warp debugging."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matcore as mc

@mc.kernel
def matmul(A, B, C):
    a = mc.load(A)
    b = mc.load(B)
    c = mc.matmul(a, b)
    mc.store(c, C)

N = 384
A = np.random.randn(N, N).astype(np.float16)
B = np.random.randn(N, N).astype(np.float16)
C = np.zeros((N, N), dtype=np.float16)

try:
    mc.launch(matmul, A, B, C, target='nvidia-dgpu:sm_89')
    print("SUCCESS - no error!")
except Exception as e:
    err = str(e)
    # Extract the stage info
    lines = err.split('\n')
    for i, line in enumerate(lines):
        if 'Failed at stage' in line or 'Diagnostic' in line:
            print(line.strip())
    
    # Find IR dump and look for specific ops
    if 'IR at failure:' in err:
        ir_part = err.split('IR at failure:')[1]
        # Look for unrealized casts and their context
        ir_lines = ir_part.split('\n')
        for i, line in enumerate(ir_lines):
            if 'unrealized_conversion_cast' in line and i < 50:
                # Print surrounding context
                start = max(0, i-2)
                end = min(len(ir_lines), i+3)
                for j in range(start, end):
                    print(f"  [{j}] {ir_lines[j].strip()}")
                print("  ---")
        
        # Check for residual non-LLVM ops
        print("\n=== RESIDUAL NON-LLVM OPS ===")
        for line in ir_lines:
            stripped = line.strip()
            # Look for ops that should have been lowered
            for op in ['vector.', 'memref.', 'arith.', 'scf.', 'gpu.thread_id', 'gpu.block_id', 'affine.']:
                if op in stripped and 'llvm.' not in stripped:
                    print(f"  RESIDUAL: {stripped[:120]}")
                    break
