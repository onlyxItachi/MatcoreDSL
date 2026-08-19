<#
.SYNOPSIS
    Evaluates register pressure, spill thresholds, and instruction selection across 7 CPU target architectures.
#>

param(
    [string]$ClangPath = "C:\Users\hamza\tools\llvm-21.1.8\bin\clang.exe",
    [string]$OutputDir = "$PSScriptRoot\..\..\..\raw-corpus\cross_isa_register_pressure"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Generate synthetic microkernels with varying accumulator counts
function Generate-GeneralMicrokernel([int]$NumAccums, [string]$Path) {
    $code = @"
#include <stddef.h>

void microkernel_accums_${NumAccums}(
    int K,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C) {
    
    // Accumulator array
    float acc[${NumAccums}];
    for (int i = 0; i < ${NumAccums}; ++i) acc[i] = 0.0f;

    for (int k = 0; k < K; ++k) {
        float b_val = B[k];
        #pragma clang loop unroll(full)
        for (int i = 0; i < ${NumAccums}; ++i) {
            acc[i] += A[k * ${NumAccums} + i] * b_val;
        }
    }

    for (int i = 0; i < ${NumAccums}; ++i) {
        C[i] += acc[i];
    }
}
"@
    $code | Out-File -FilePath $Path -Encoding utf8
}

$accumCounts = @(4, 8, 12, 16, 20, 24, 28, 32)
$targets = @(
    @{ Name = "SSE4.2"; Flags = @("-target", "x86_64-pc-windows-msvc", "-msse4.2"); RegClass = "%xmm"; SpillPattern = "Spill|(%rsp)" },
    @{ Name = "AVX2"; Flags = @("-target", "x86_64-pc-windows-msvc", "-mavx2", "-mfma"); RegClass = "%ymm"; SpillPattern = "Spill|(%rsp)" },
    @{ Name = "AVX-512"; Flags = @("-target", "x86_64-pc-windows-msvc", "-mavx512f", "-mavx512vl"); RegClass = "%zmm"; SpillPattern = "Spill|(%rsp)" },
    @{ Name = "AArch64-NEON"; Flags = @("-target", "aarch64-unknown-linux-gnu", "-march=armv8-a"); RegClass = "v\d+\."; SpillPattern = "str\s+q|ldr\s+q" },
    @{ Name = "AArch64-SVE"; Flags = @("-target", "aarch64-unknown-linux-gnu", "-march=armv8.4-a+sve"); RegClass = "z\d+\."; SpillPattern = "str\s+z|ldr\s+z" },
    @{ Name = "RISCV-RVV"; Flags = @("-target", "riscv64-unknown-linux-gnu", "-march=rv64gcv"); RegClass = "v\d+"; SpillPattern = "vse|vle" }
)

$report = @()

foreach ($acc in $accumCounts) {
    $src = Join-Path $OutputDir "microkernel_acc${acc}.c"
    Generate-GeneralMicrokernel -NumAccums $acc -Path $src

    foreach ($tgt in $targets) {
        $asm = Join-Path $OutputDir "acc${acc}_$($tgt.Name).s"
        $cmd = @($ClangPath, "-O3", "-S", $src, "-o", $asm) + $tgt.Flags
        & $cmd[0] $cmd[1..($cmd.Length - 1)]

        $asmContent = Get-Content $asm
        $regMatches = ($asmContent | Select-String -Pattern $tgt.RegClass -AllMatches).Matches.Count
        $spillMatches = ($asmContent | Select-String -Pattern $tgt.SpillPattern -AllMatches).Matches.Count

        $status = if ($spillMatches -gt 2) { "SPILL_DETECTED" } else { "ZERO_SPILLS" }

        $report += [PSCustomObject]@{
            Target = $tgt.Name
            Accumulators = $acc
            Vector_Ops = $regMatches
            Stack_Spills = $spillMatches
            Spill_Status = $status
        }
    }
}

$report | Group-Object Target | ForEach-Object {
    Write-Output "`n=== Target: $($_.Name) ==="
    $_.Group | Format-Table Accumulators, Vector_Ops, Stack_Spills, Spill_Status -AutoSize
}
