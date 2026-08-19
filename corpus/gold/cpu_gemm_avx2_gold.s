# MDSLC Gold Fixture: CPU AVX2 GEMM Inner Kernel Snippet
# Architecture: x86-64 / AVX2 + FMA
# Represents: Optimal FMA unrolling and scalar broadcast across 8 YMM accumulators
.text
.globl gemm_f32_microkernel_avx2_6x16
gemm_f32_microkernel_avx2_6x16:
    vbroadcastss    (%rdx), %ymm0
    vbroadcastss    4(%rdx), %ymm1
    vmovaps         (%r8), %ymm2
    vmovaps         32(%r8), %ymm3
    vfmadd231ps     %ymm2, %ymm0, %ymm4
    vfmadd231ps     %ymm3, %ymm0, %ymm5
    vfmadd231ps     %ymm2, %ymm1, %ymm6
    vfmadd231ps     %ymm3, %ymm1, %ymm7
    retq
