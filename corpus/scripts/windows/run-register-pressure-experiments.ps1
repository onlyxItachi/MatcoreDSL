<#
.SYNOPSIS
    Compiles controlled microkernel variants with varying MR x NR register tile geometries to observe register pressure and spill thresholds.
.DESCRIPTION
    Tests geometries: 8x2, 8x4, 16x4, 16x6, 16x8 on AVX2 (16 YMM budget) and AVX-512 (32 ZMM budget).
#>

param(
    [string]$ClangPath = "C:\Users\hamza\tools\llvm-21.1.8\bin\clang.exe",
    [string]$OutputDir = "$PSScriptRoot\..\..\..\raw-corpus\register_pressure_experiments"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# C Template for unrolled microkernel
function Generate-MicrokernelSource([int]$MR, [int]$NR, [int]$K_UNROLL, [string]$Path) {
    $accumCount = ($MR / 8) * $NR  # Number of 256-bit YMM vectors for accumulation on AVX2
    if ($accumCount -lt 1) { $accumCount = 1 }

    $code = @"
#include <immintrin.h>

void microkernel_MR${MR}_NR${NR}_K${K_UNROLL}(
    int K,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int ldc) {
    
    // Accumulator definitions
"@
    for ($i = 0; $i -lt $MR; $i += 8) {
        for ($j = 0; $j -lt $NR; ++$j) {
            $code += "`n    __m256 acc_${i}_${j} = _mm256_setzero_ps();"
        }
    }

    $code += @"

    for (int k = 0; k < K; k += $K_UNROLL) {
"@
    for ($u = 0; $u -lt $K_UNROLL; ++$u) {
        for ($i = 0; $i -lt $MR; $i += 8) {
            $code += "`n        __m256 a_${i}_u${u} = _mm256_loadu_ps(&A[(k + $u) * $MR + $i]);"
        }
        for ($j = 0; $j -lt $NR; ++$j) {
            $code += "`n        __m256 b_${j}_u${u} = _mm256_set1_ps(B[(k + $u) * $NR + $j]);"
            for ($i = 0; $i -lt $MR; $i += 8) {
                $code += "`n        acc_${i}_${j} = _mm256_fmadd_ps(a_${i}_u${u}, b_${j}_u${u}, acc_${i}_${j});"
            }
        }
    }

    $code += @"

    }

    // Store back to C
"@
    for ($i = 0; $i -lt $MR; $i += 8) {
        for ($j = 0; $j -lt $NR; ++$j) {
            $code += "`n    _mm256_storeu_ps(&C[${j} * ldc + ${i}], acc_${i}_${j});"
        }
    }

    $code += "`n}`n"

    $code | Out-File -FilePath $Path -Encoding utf8
}

$geometries = @(
    @{ MR = 8; NR = 2; K = 4 },
    @{ MR = 8; NR = 4; K = 4 },
    @{ MR = 16; NR = 4; K = 4 },
    @{ MR = 16; NR = 6; K = 4 },
    @{ MR = 16; NR = 8; K = 4 },   # 16 accumulators -> Should trigger spills on AVX2 (16 regs)
    @{ MR = 16; NR = 10; K = 4 }   # 20 accumulators -> Heavy spilling on AVX2
)

$results = @()

foreach ($g in $geometries) {
    $mr = $g.MR
    $nr = $g.NR
    $k = $g.K
    $src = Join-Path $OutputDir "microkernel_MR${mr}_NR${nr}.c"
    $asm_avx2 = Join-Path $OutputDir "microkernel_MR${mr}_NR${nr}_avx2.s"
    $asm_avx512 = Join-Path $OutputDir "microkernel_MR${mr}_NR${nr}_avx512.s"

    Generate-MicrokernelSource -MR $mr -NR $nr -K_UNROLL $k -Path $src

    # Compile AVX2
    & $ClangPath -O3 -mavx2 -mfma -S $src -o $asm_avx2
    # Compile AVX-512
    & $ClangPath -O3 -mavx512f -mavx512dq -mavx512vl -S $src -o $asm_avx512

    # Analyze Spills & Registers
    $content_avx2 = Get-Content $asm_avx2
    $spills_avx2 = ($content_avx2 | Select-String "Spill|(%rsp)" -AllMatches).Matches.Count
    $ymm_avx2 = ($content_avx2 | Select-String "%ymm" -AllMatches).Matches.Count

    $content_avx512 = Get-Content $asm_avx512
    $spills_avx512 = ($content_avx512 | Select-String "Spill|(%rsp)" -AllMatches).Matches.Count
    $zmm_avx512 = ($content_avx512 | Select-String "%zmm|%ymm" -AllMatches).Matches.Count

    $results += [PSCustomObject]@{
        Tile_Geometry = "${mr}x${nr}"
        Accumulator_YMM_Count = ($mr / 8) * $nr
        AVX2_YMM_Ops = $ymm_avx2
        AVX2_Stack_Spill_Refs = $spills_avx2
        AVX2_Spill_State = if ($spills_avx2 -gt 2) { "SPILLING_DETECTED" } else { "ZERO_INNER_SPILLS" }
        AVX512_Spill_State = if ($spills_avx512 -gt 2) { "SPILLING_DETECTED" } else { "ZERO_INNER_SPILLS" }
    }
}

$results | Format-Table -AutoSize
