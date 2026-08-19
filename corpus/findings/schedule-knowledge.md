# Schedule Knowledge Base & Optimization Findings

## 1. High-Performance Schedule Parameters (Empirical Evidence)

Based on compiler archaeology across high-performance BLAS libraries (OpenBLAS, BLIS, CUTLASS, rocWMMA) and compiler pass analysis, this document codifies verified hardware schedules and maps them to upstream MLIR transformation concepts.

---

## 2. CPU Architecture Schedule Taxonomy

### AVX2 / FMA Row-Major F32 GEMM:
* **Microkernel Geometry**: $M_R \times N_R = 6 \times 16$ (utilizing 12 vector registers for accumulation, 2 for $A$ broadcast, 2 for $B$ stream; total 16 YMM registers fully utilized).
* **L1 Cache Blocking**: $M_C = 64, K_C = 256$.
* **L2 / L3 Cache Staging**: $N_C = 2048$, prepacked $B$ panel ($K_C \times N_C$) stored in contiguous transposed or packed micro-panels ($K \times 16$) to eliminate TLB misses and stride penalties.
* **MLIR Transform Mapping**:
  ```text
  linalg.matmul
    -> linalg.tile_to_forall_op (num_threads = num_workers, tile_sizes = [64, 256])
    -> linalg.tile (tile_sizes = [6, 16, 4])
    -> vector.contract
    -> vector.unroll [6, 16, 4]
  ```

### AVX-512 Row-Major F32 GEMM:
* **Microkernel Geometry**: $M_R \times N_R = 12 \times 32$ or $14 \times 32$ (utilizing 24–28 ZMM registers for accumulation out of 32 ZMM registers).
* **L1 / L2 Cache Blocking**: $M_C = 128, K_C = 512, N_C = 4096$.

---

## 3. NVIDIA Tensor Core Schedule Taxonomy (Ada / `sm_89`)

* **CTA Task Geometry**: $128 \times 128 \times 32$ or $128 \times 256 \times 32$.
* **Warp Level Allocation**: 4 warps ($2 \times 2$ warp grid), each executing $64 \times 64 \times 16$.
* **Hardware MMA Primitive**: `m16n8k16` FP16 input $\rightarrow$ FP32 accumulation (`nvgpu.mma.sync`).
* **Memory Staging Primitive**: `cp.async` vectorized 16-byte global-to-shared transactions.
* **Shared Memory Layout**: $128 \times 32$ tiles with 8-byte skew/swizzle padding to guarantee zero bank conflicts during `ldmatrix.x4.b16` fragment loading.
* **Pipelining**: 3-stage or 4-stage async pipeline overlapping global copy of slice $k+1$ with MMA compute of slice $k$.

---

## 4. AMD Matrix Core Schedule Taxonomy (CDNA2 / CDNA3 / RDNA3)

* **Workgroup Geometry**: $128 \times 128 \times 32$.
* **Wavefront Allocation**: 4 waves (Wave64 on CDNA, Wave32 on RDNA3), executing $64 \times 64 \times 16$ tile segments.
* **Hardware Matrix Primitive**: `amdgpu.mfma` $32 \times 32 \times 8$ or $16 \times 16 \times 16$.
* **LDS Staging**: Double-buffered ping-pong LDS allocation with direct `ds_read_b128` fragment loads into VGPR pairs.
