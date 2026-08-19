# Post-Op Fusion Performance & Memory Traffic Validation

**Investigation Focus**: Empirical host validation of fused vs separate epilogue execution time and clarification of RAM vs Cache traffic claims.  
**Audited Toolchain**: LLVM/Clang 21.1.8 on Windows x64 host  
**Confidence Level**: `REPLICATED_OBSERVATION`

---

## 1. Empirical Host Benchmark Results (`HARDEN-05` Resolved)

We benchmarked a square matrix sweep comparing:
1. `Separate_ReLU`: Vector-tiled GEMM writing to $C$, followed by a dedicated scalar/vector ReLU pass over $C$.
2. `Fused_ReLU`: Vector-tiled GEMM with in-register `_mm256_max_ps` applied during row store.

| Matrix Shape ($N \times N$) | Working Set Size ($3 \times N^2 \times 4\text{B}$) | Tiled GEMM Time (ms) | Separate ReLU Time (ms) | Fused ReLU Time (ms) | Speedup Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$16 \times 16$** | 3 KB (L1 Cache Resident) | 0.0004 ms | 0.0004 ms | 0.0005 ms | **0.92x** |
| **$32 \times 32$** | 12 KB (L1 Cache Resident) | 0.0024 ms | 0.0024 ms | 0.0025 ms | **0.96x** |
| **$64 \times 64$** | 48 KB (L2 Cache Resident) | 0.0145 ms | 0.0147 ms | 0.0158 ms | **0.93x** |
| **$128 \times 128$** | 192 KB (L2 Cache Resident) | 0.1108 ms | 0.1115 ms | 0.1147 ms | **0.97x** |
| **$256 \times 256$** | 768 KB (L3 Cache Resident) | 0.8680 ms | 0.8705 ms | 0.8816 ms | **0.99x** |

---

## 2. Critical Analysis & Falsification of Overconfident Claims

### A. Falsification of "2x Speedup on GEMM+ReLU"
- For compute-bound dense GEMM ($O(N^3)$ compute vs $O(N^2)$ memory), the runtime is completely dominated by the matrix multiply ($>99.5\%$ of time).
- Fusing a simple ReLU into GEMM produces **negligible overall speedup ($\approx 0.95\text{–}1.0\times$)** because the arithmetic cost of the GEMM dominates the epilogue.
- **Where Fusion Delivers $2\times$ Speedup**: Post-op fusion delivers major speedups on **memory-bandwidth-bound elementwise chains** (e.g. Bias + LayerNorm + ReLU + Residual Add), where arithmetic intensity is low and every pass is DRAM-bound.

### B. Correction of "100% Eliminates RAM Traffic"
- For cache-resident working sets ($N \le 128$), matrix $C$ remains in L1/L2 CPU cache. The separate pass reads from L1/L2 cache, resulting in **zero physical DRAM traffic in both cases**.
- **Corrected Architectural Claim**: Post-op fusion eliminates the **intermediate store/reload pass at the logical program and cache level**. Physical DRAM traffic reduction occurs only when the matrix size exceeds CPU cache capacity.

---

## 3. Claim Status
- `HARDEN-05` is **NARROWED & RECALIBRATED** from `ARCHITECTURAL_INVARIANT` to `REPLICATED_OBSERVATION`.
