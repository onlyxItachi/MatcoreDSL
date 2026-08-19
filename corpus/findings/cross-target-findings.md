# Cross-Target Compiler Lowering Findings

**Investigation Scope**: Multi-architecture compilation across x86-64 (AVX2, AVX-512), AArch64 (NEON, SVE)  
**Execution Status**: Statically compiled & verified via Clang/LLVM 21.1.8 (`statically_compiled_only` for non-host AArch64 targets)  
**Confidence Level**: `STRONGLY_SUPPORTED`

---

## 1. Cross-Target Lowering Matrix

| Architecture & ISA | Target Triple & Flags | Register Class Queried | Vector Representation | Instructions Emitted | Zero-Spill Max Tile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **x86-64 AVX2 / FMA** | `x86_64-pc-windows-msvc`<br>`-mavx2 -mfma` | 16 $\times$ 256-bit YMM | Fixed `<8 x float>` | `vfmadd213ps %ymm`, `vbroadcastss` | $M_R \times N_R = 16 \times 6$ (12 YMM acc) |
| **x86-64 AVX-512** | `x86_64-pc-windows-msvc`<br>`-mavx512f -mavx512vl` | 32 $\times$ 512-bit ZMM | Fixed `<16 x float>` | `vfmadd213ps %zmm`, `vbroadcastss` | $M_R \times N_R = 32 \times 6$ or $16 \times 14$ (28 ZMM acc) |
| **AArch64 NEON** | `aarch64-unknown-linux-gnu`<br>`-march=armv8-a` | 32 $\times$ 128-bit V | Fixed `<4 x float>` | `fmla v0.4s, v1.4s, v2.s[0]` | $M_R \times N_R = 8 \times 12$ (24 V acc) |
| **AArch64 SVE** | `aarch64-unknown-linux-gnu`<br>`-march=armv8.4-a+sve` | 32 $\times$ Scalable Z, 8 $\times$ P | Scalable `vector<[4]xf32>` | `fmla z0.s, p0/m, z1.s, z2.s` | Predicate-driven scalable unroll |

---

## 2. Universal Architectural Generalizations

1. **TTI Register Class Interface Generalization**:
   - The LLVM middle-end vectorizer does not hardcode x86 assumptions. It queries `TTI::getRegisterBitWidth()` and `TTI::getNumberOfRegisters()`. The exact same loop vectorization logic that selects $VF=8$ on AVX2 selects $VF=16$ on AVX-512, $VF=4$ on AArch64 NEON, and scalable vectorization on AArch64 SVE.
2. **Register File Budget Controls Tile Saturation**:
   - AArch64 NEON has 32 architectural vector registers (vs 16 on x86 AVX2). Consequently, OpenBLAS and BLIS choose larger $N_R$ register tile dimensions on ARM ($M_R \times N_R = 8 \times 12$) compared to Haswell ($M_R \times N_R = 16 \times 6$ or $8 \times 4$).
3. **The Necessity of Explicit Target Flags**:
   - On every architecture tested, omitting target flags causes Clang to fall back to the lowest common denominator (128-bit SSE on x86, baseline scalar/vector on ARM). Target awareness is universally mandatory at the frontend optimization layer.
