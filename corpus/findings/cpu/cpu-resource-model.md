# Modern CPU Compute Resource Model

**Compiler Baseline**: LLVM/Clang 21.1.8 (Audited MDSLC Baseline)  
**Target Coverage**: 7 Architectures spanning Fixed SIMD, Scalable Vector, and 2D Matrix/Tile Hardware  
**Target Execution Status**: Host is Windows x64; non-host targets are `statically_compiled_only`  
**Confidence Level**: `STRONGLY_SUPPORTED` (Probed and verified via Clang 21.1.8 compiler backends)

---

## 1. Multi-ISA Architecture Resource Taxonomy

| ISA Family | Hardware Register Budget | Vector / Matrix Semantics | Primary Arithmetic Instructions | Predicate / Masking Resources | Primary Hardware Bottlenecks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **x86-64 SSE4.2** | 16 $\times$ 128-bit `XMM` | Fixed-length (128-bit) | `mulps`, `addps` (2-operand non-destructive or 3-operand AVX VEX) | None (scalar loop remainder peeling) | High register pressure; lack of FMA instructions; 4 floats/vector. |
| **x86-64 AVX2 + FMA** | 16 $\times$ 256-bit `YMM` | Fixed-length (256-bit) | `vfmadd213ps`, `vfmadd231ps`, `vbroadcastss` | Vector blend instructions (`vblendvps`); remainder peeling | 16-register pressure cliff ($16 \times 6$ saturation); dual FMA execution ports. |
| **x86-64 AVX-512** | 32 $\times$ 512-bit `ZMM` | Fixed-length (512-bit) | `vfmadd213ps %zmm`, `vbroadcastss %zmm` | 8 dedicated opmask registers ($k_0..k_7$) | Thermal downclocking on older Intel; large register file absorbs $16 \times 14$ tiles. |
| **x86-64 AMX** | 8 $\times$ 2D Tile Regs (`tmm0`..`tmm7`), 1KB/tile | 2D Matrix Tile ($16 \times 64$ bytes) | `tdpbf16ps`, `tdpbusd`, `tdpfp16ps`, `tileloadd`, `tilestored` | Explicit tile configuration (`ldtilecfg` / `sttilecfg`) | Coarse granularity (matrix tile shapes); fixed BF16/INT8/FP16 precision; host config latency. |
| **AArch64 NEON** | 32 $\times$ 128-bit `V` (`v0`..`v31`) | Fixed-length (128-bit) | `fmla v0.4s, v1.4s, v2.s[0]`, `fmla v0.4s, v1.4s, v2.4s` | None (scalar remainder loop) | 128-bit vector width requires 24 accumulators ($8 \times 12$ tile) to saturate FMA pipelines. |
| **AArch64 SVE / SVE2** | 32 $\times$ Scalable `Z` (`z0`..`z31`) | Scalable / Length-Agnostic (128–2048 bits) | `fmla z0.s, p0/m, z1.s, z2.s`, `ld1w {z0.s}, p0/z, [x0]` | 16 predicate registers ($p_0..p_{15}$) | Dynamic vector length requires predicate-driven loop tail masking without scalar peeling. |
| **RISC-V RVV** | 32 $\times$ Scalable `V` (`v0`..`v31`) + `vlenb` | Dynamic Vector Length ($VLEN \ge 128$) | `vfmacc.vv`, `vfmacc.vf`, `vsetvli` | Active vector mask register ($v_0$) | Compiler must emit `vsetvli` configuration before vector loops; register grouping (`LMUL`). |

---

## 2. Structural Classification of Execution Topologies

The 7 target ISAs partition into **Three Distinct Structural Classes**:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. FIXED-WIDTH SIMD (SSE4.2, AVX2, AVX-512, NEON)                                                │
│    - Vector length is known at compile time (4, 8, 16 floats).                                   │
│    - Schedulers unroll loops into static unroll factors ($VF \times UF$).                         │
│    - Tail handling requires scalar peeling or vector mask blending.                              │
│    - Register allocation is tightly coupled to static architectural register file size (16 or 32).│
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. SCALABLE / VECTOR-LENGTH-AGNOSTIC SIMD (AArch64 SVE, RISC-V RVV)                              │
│    - Vector length is determined at CPU runtime (not fixed at compile time).                     │
│    - Loop control is driven by hardware predicate / length-setting instructions (`vsetvli`, `p0`).│
│    - Loop tail peeling is completely eliminated: hardware predicate masks out excess elements.   │
│    - Schedulers unroll abstract vector registers without hardcoding physical element counts.     │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. 2D MATRIX / TILE ACCELERATION (Intel AMX, ARM SME)                                            │
│    - Computation moves from 1D vector lanes to 2D outer-product matrix tiles ($16 \times 16$).    │
│    - Schedulers do not manage individual vector lanes; they manage 2D tile registers (`tmm0..7`). │
│    - Epilogues and transformations must be staged through memory or vector register tile stores. │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
