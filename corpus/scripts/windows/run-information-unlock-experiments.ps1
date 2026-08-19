<#
.SYNOPSIS
    Runs controlled information-unlock experiments changing ONE compiler-visible fact at a time.
.DESCRIPTION
    Evaluates what compiler optimizations are unlocked by: Restrict, Alignment, Reassociation Pragma, Loop Order (IKJ), Constant Dimensions, Tiling, and Pre-Packing.
#>

param(
    [string]$ClangPath = "C:\Users\hamza\tools\llvm-21.1.8\bin\clang.exe",
    [string]$OutputDir = "$PSScriptRoot\..\..\..\raw-corpus\information_unlock_experiments"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$variants = @(
    @{
        Name = "01_naive"
        Description = "Baseline dynamic C99 GEMM (IJK)"
        Code = @"
void gemm_01_naive(int M, int N, int K, const float *A, const float *B, float *C, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}
"@
    },
    @{
        Name = "02_restrict"
        Description = "Adds __restrict__ (Disjoint memory spaces)"
        Code = @"
void gemm_02_restrict(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}
"@
    },
    @{
        Name = "03_aligned"
        Description = "Adds 32-byte alignment proof"
        Code = @"
#include <stddef.h>
void gemm_03_aligned(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int lda, int ldb, int ldc) {
    const float * __restrict__ a = (const float *)__builtin_assume_aligned(A, 32);
    const float * __restrict__ b = (const float *)__builtin_assume_aligned(B, 32);
    float * __restrict__ c = (float *)__builtin_assume_aligned(C, 32);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * lda + k] * b[k * ldb + j];
            }
            c[i * ldc + j] = sum;
        }
    }
}
"@
    },
    @{
        Name = "04_reassoc_pragma"
        Description = "Adds reduction reassociation permission"
        Code = @"
void gemm_04_reassoc_pragma(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            #pragma clang loop vectorize(enable)
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}
"@
    },
    @{
        Name = "05_loop_order_ikj"
        Description = "Loop interchange to IKJ (Accumulation in inner loop)"
        Code = @"
void gemm_05_loop_order_ikj(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_val = A[i * lda + k];
            #pragma clang loop vectorize(enable)
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += a_val * B[k * ldb + j];
            }
        }
    }
}
"@
    },
    @{
        Name = "06_constant_dims"
        Description = "Known constant dimensions (M=64, N=64, K=64)"
        Code = @"
#define M 64
#define N 64
#define K 64
void gemm_06_constant_dims(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_val = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}
"@
    },
    @{
        Name = "07_tiled"
        Description = "32x32x64 Cache-blocked loop nest"
        Code = @"
#define TM 32
#define TN 32
#define TK 64
void gemm_07_tiled(int M_dim, int N_dim, int K_dim, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    for (int bi = 0; bi < M_dim; bi += TM) {
        for (int bj = 0; bj < N_dim; bj += TN) {
            for (int bk = 0; bk < K_dim; bk += TK) {
                for (int i = 0; i < TM; ++i) {
                    for (int k = 0; k < TK; ++k) {
                        float a_val = A[(bi + i) * K_dim + (bk + k)];
                        for (int j = 0; j < TN; ++j) {
                            C[(bi + i) * N_dim + (bj + j)] += a_val * B[(bk + k) * N_dim + (bj + j)];
                        }
                    }
                }
            }
        }
    }
}
"@
    }
)

$results = @()

foreach ($v in $variants) {
    $src = Join-Path $OutputDir "$($v.Name).c"
    $asm = Join-Path $OutputDir "$($v.Name)_avx2.s"
    $remarks_file = Join-Path $OutputDir "$($v.Name)_remarks.txt"

    $v.Code | Out-File -FilePath $src -Encoding utf8

    # Compile with AVX2/FMA and capture remarks
    $remarks = & $ClangPath -O3 -mavx2 -mfma -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -S $src -o $asm 2>&1
    $remarks | Out-File -FilePath $remarks_file -Encoding utf8

    $asmContent = Get-Content $asm
    $ymm = ($asmContent | Select-String "%ymm" -AllMatches).Matches.Count
    $vmovaps = ($asmContent | Select-String "vmovaps" -AllMatches).Matches.Count
    $vmovups = ($asmContent | Select-String "vmovups" -AllMatches).Matches.Count
    $vfmadd = ($asmContent | Select-String "vfmadd" -AllMatches).Matches.Count
    $aliasCheck = ($asmContent | Select-String "jae|jbe" -AllMatches).Matches.Count

    $remStr = $remarks -join " "
    $vecStatus = if ($remStr -match "vectorized loop \(vectorization width: (\d+)") { "VF=$($Matches[1])" } else { "Scalar" }

    $results += [PSCustomObject]@{
        Variant = $v.Name
        Fact_Introduced = $v.Description
        Vector_Status = $vecStatus
        YMM_Count = $ymm
        FMA_Count = $vfmadd
        Aligned_Moves = $vmovaps
        Unaligned_Moves = $vmovups
        Branch_Checks = $aliasCheck
    }
}

$results | Format-Table -AutoSize
