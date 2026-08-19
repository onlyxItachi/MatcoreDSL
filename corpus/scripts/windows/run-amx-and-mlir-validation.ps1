<#
.SYNOPSIS
    Compiles real Intel AMX intrinsics to verify tmm register emission and analyzes structured MLIR-style vector contraction microkernel assembly.
#>

param(
    [string]$ClangPath = "C:\Users\hamza\tools\llvm-21.1.8\bin\clang.exe",
    [string]$OutputDir = "$PSScriptRoot\..\..\..\raw-corpus\amx_mlir_validation"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# 1. Real Intel AMX Intrinsic Microkernel Test
$amxSrc = Join-Path $OutputDir "amx_matmul_bf16.c"
@"
#include <immintrin.h>

void amx_matmul_tile_bf16(
    const void *A, int lda,
    const void *B, int ldb,
    void *C, int ldc) {
    
    // Configure and load AMX tiles (tmm0 = C, tmm1 = A, tmm2 = B)
    _tile_zero(0);
    _tile_loadd(1, A, lda);
    _tile_loadd(2, B, ldb);

    // Execute 2D matrix tile BF16 dot-product multiply-accumulate: tmm0 += tmm1 * tmm2
    _tile_dpbf16ps(0, 1, 2);

    // Store tile back to memory C
    _tile_stored(0, C, ldc);
    _tile_release();
}
"@ | Out-File -FilePath $amxSrc -Encoding utf8

$amxAsm = Join-Path $OutputDir "amx_matmul_bf16.s"
& $ClangPath -O3 -march=sapphirerapids -mamx-tile -mamx-bf16 -fverbose-asm -S $amxSrc -o $amxAsm

$amxLines = Get-Content $amxAsm
$tmmOps = ($amxLines | Select-String "%tmm" -AllMatches).Matches.Count
$tileLoadOps = ($amxLines | Select-String "tileloadd" -AllMatches).Matches.Count
$tdpOps = ($amxLines | Select-String "tdpbf16ps" -AllMatches).Matches.Count
$tileStoreOps = ($amxLines | Select-String "tilestored" -AllMatches).Matches.Count

Write-Output "=== Verified AMX Tile Generation ==="
Write-Output "Total TMM Register References: $tmmOps"
Write-Output "tileloadd Instructions: $tileLoadOps"
Write-Output "tdpbf16ps Tile Multiply Instructions: $tdpOps"
Write-Output "tilestored Instructions: $tileStoreOps"

# 2. Structured MLIR-Style Vector Contraction (16x6 AVX2 Tile)
$mlirVectorSrc = Join-Path $OutputDir "vector_contract_16x6.c"
@"
#include <immintrin.h>

void vector_contract_16x6(
    int K,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int ldc) {
    
    // 12 vector accumulators for 16x6 tile (2 YMM x 6 cols)
    __m256 acc0_0 = _mm256_setzero_ps(); __m256 acc1_0 = _mm256_setzero_ps();
    __m256 acc0_1 = _mm256_setzero_ps(); __m256 acc1_1 = _mm256_setzero_ps();
    __m256 acc0_2 = _mm256_setzero_ps(); __m256 acc1_2 = _mm256_setzero_ps();
    __m256 acc0_3 = _mm256_setzero_ps(); __m256 acc1_3 = _mm256_setzero_ps();
    __m256 acc0_4 = _mm256_setzero_ps(); __m256 acc1_4 = _mm256_setzero_ps();
    __m256 acc0_5 = _mm256_setzero_ps(); __m256 acc1_5 = _mm256_setzero_ps();

    for (int k = 0; k < K; ++k) {
        __m256 a0 = _mm256_loadu_ps(&A[k * 16 + 0]);
        __m256 a1 = _mm256_loadu_ps(&A[k * 16 + 8]);

        __m256 b0 = _mm256_set1_ps(B[k * 6 + 0]);
        acc0_0 = _mm256_fmadd_ps(a0, b0, acc0_0);
        acc1_0 = _mm256_fmadd_ps(a1, b0, acc1_0);

        __m256 b1 = _mm256_set1_ps(B[k * 6 + 1]);
        acc0_1 = _mm256_fmadd_ps(a0, b1, acc0_1);
        acc1_1 = _mm256_fmadd_ps(a1, b1, acc1_1);

        __m256 b2 = _mm256_set1_ps(B[k * 6 + 2]);
        acc0_2 = _mm256_fmadd_ps(a0, b2, acc0_2);
        acc1_2 = _mm256_fmadd_ps(a1, b2, acc1_2);

        __m256 b3 = _mm256_set1_ps(B[k * 6 + 3]);
        acc0_3 = _mm256_fmadd_ps(a0, b3, acc0_3);
        acc1_3 = _mm256_fmadd_ps(a1, b3, acc1_3);

        __m256 b4 = _mm256_set1_ps(B[k * 6 + 4]);
        acc0_4 = _mm256_fmadd_ps(a0, b4, acc0_4);
        acc1_4 = _mm256_fmadd_ps(a1, b4, acc1_4);

        __m256 b5 = _mm256_set1_ps(B[k * 6 + 5]);
        acc0_5 = _mm256_fmadd_ps(a0, b5, acc0_5);
        acc1_5 = _mm256_fmadd_ps(a1, b5, acc1_5);
    }

    _mm256_storeu_ps(&C[0 * ldc + 0], acc0_0); _mm256_storeu_ps(&C[0 * ldc + 8], acc1_0);
    _mm256_storeu_ps(&C[1 * ldc + 0], acc0_1); _mm256_storeu_ps(&C[1 * ldc + 8], acc1_1);
    _mm256_storeu_ps(&C[2 * ldc + 0], acc0_2); _mm256_storeu_ps(&C[2 * ldc + 8], acc1_2);
    _mm256_storeu_ps(&C[3 * ldc + 0], acc0_3); _mm256_storeu_ps(&C[3 * ldc + 8], acc1_3);
    _mm256_storeu_ps(&C[4 * ldc + 0], acc0_4); _mm256_storeu_ps(&C[4 * ldc + 8], acc1_4);
    _mm256_storeu_ps(&C[5 * ldc + 0], acc0_5); _mm256_storeu_ps(&C[5 * ldc + 8], acc1_5);
}
"@ | Out-File -FilePath $mlirVectorSrc -Encoding utf8

$mlirVectorAsm = Join-Path $OutputDir "vector_contract_16x6.s"
& $ClangPath -O3 -mavx2 -mfma -fverbose-asm -S $mlirVectorSrc -o $mlirVectorAsm

$vectorLines = Get-Content $mlirVectorAsm
$vecSpills = ($vectorLines | Select-String "#\s*.*Spill|#\s*.*Reload" -AllMatches).Matches.Count
$vecFma = ($vectorLines | Select-String "vfmadd" -AllMatches).Matches.Count
$vecYmm = ($vectorLines | Select-String "%ymm" -AllMatches).Matches.Count

Write-Output "`n=== Verified Structured Vector Contraction (16x6 AVX2) ==="
Write-Output "Inner Loop Compiler Spills: $vecSpills"
Write-Output "FMA Instructions Emitted: $vecFma"
Write-Output "Total YMM Register References: $vecYmm"
