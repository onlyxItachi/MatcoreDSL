<#
.SYNOPSIS
    Hardened register pressure and spill detection benchmark across 6 CPU architectures using -fverbose-asm.
#>

param(
    [string]$ClangPath = "C:\Users\hamza\tools\llvm-21.1.8\bin\clang.exe",
    [string]$OutputDir = "$PSScriptRoot\..\..\..\raw-corpus\hardened_register_pressure"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Generate-HardenedMicrokernel([int]$NumAccums, [string]$Path) {
    $code = @"
#include <stddef.h>

void microkernel_hardened_acc_${NumAccums}(
    int K,
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C) {
    
    float acc[${NumAccums}];
    #pragma clang loop unroll(full)
    for (int i = 0; i < ${NumAccums}; ++i) acc[i] = 0.0f;

    for (int k = 0; k < K; ++k) {
        float b_val = B[k];
        #pragma clang loop unroll(full)
        for (int i = 0; i < ${NumAccums}; ++i) {
            acc[i] += A[k * ${NumAccums} + i] * b_val;
        }
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < ${NumAccums}; ++i) {
        C[i] += acc[i];
    }
}
"@
    $code | Out-File -FilePath $Path -Encoding utf8
}

$accumCounts = @(4, 8, 12, 16, 20, 24, 28, 32)
$targets = @(
    @{ Name = "SSE4.2"; Flags = @("-target", "x86_64-pc-windows-msvc", "-msse4.2", "-fverbose-asm") },
    @{ Name = "AVX2"; Flags = @("-target", "x86_64-pc-windows-msvc", "-mavx2", "-mfma", "-fverbose-asm") },
    @{ Name = "AVX-512"; Flags = @("-target", "x86_64-pc-windows-msvc", "-mavx512f", "-mavx512vl", "-fverbose-asm") },
    @{ Name = "AArch64-NEON"; Flags = @("-target", "aarch64-unknown-linux-gnu", "-march=armv8-a", "-fverbose-asm") },
    @{ Name = "AArch64-SVE"; Flags = @("-target", "aarch64-unknown-linux-gnu", "-march=armv8.4-a+sve", "-fverbose-asm") },
    @{ Name = "RISCV-RVV"; Flags = @("-target", "riscv64-unknown-linux-gnu", "-march=rv64gcv", "-fverbose-asm") }
)

$results = @()

foreach ($acc in $accumCounts) {
    $src = Join-Path $OutputDir "acc_${acc}.c"
    Generate-HardenedMicrokernel -NumAccums $acc -Path $src

    foreach ($tgt in $targets) {
        $asm = Join-Path $OutputDir "acc_${acc}_$($tgt.Name).s"
        $cmd = @($ClangPath, "-O3", "-S", $src, "-o", $asm) + $tgt.Flags
        & $cmd[0] $cmd[1..($cmd.Length - 1)]

        $asmLines = Get-Content $asm
        # Match explicit spill and reload annotations emitted by Clang's code generator
        $spills = ($asmLines | Select-String "#\s*.*Spill|#\s*.*Reload" -AllMatches).Matches.Count

        $results += [PSCustomObject]@{
            Target = $tgt.Name
            Accumulators = $acc
            Compiler_Spills = $spills
            Status = if ($spills -gt 0) { "SPILL_PRESENT ($spills)" } else { "ZERO_SPILLS" }
        }
    }
}

$results | Group-Object Target | ForEach-Object {
    Write-Output "`n=== Hardened Target: $($_.Name) ==="
    $_.Group | Format-Table Accumulators, Compiler_Spills, Status -AutoSize
}
