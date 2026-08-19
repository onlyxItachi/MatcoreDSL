<#
.SYNOPSIS
    Cross-ISA executable proof of floating-point reassociation and loop interchange vectorization across 6 targets.
#>

param(
    [string]$ClangPath = "C:\Users\hamza\tools\llvm-21.1.8\bin\clang.exe",
    [string]$ProofDir = $PSScriptRoot
)

$ErrorActionPreference = "Continue"

$targets = @(
    @{ Name = "SSE4.2"; Flags = @("-target", "x86_64-pc-windows-msvc", "-msse4.2") },
    @{ Name = "AVX2"; Flags = @("-target", "x86_64-pc-windows-msvc", "-mavx2", "-mfma") },
    @{ Name = "AVX-512"; Flags = @("-target", "x86_64-pc-windows-msvc", "-mavx512f", "-mavx512vl") },
    @{ Name = "AArch64-NEON"; Flags = @("-target", "aarch64-unknown-linux-gnu", "-march=armv8-a") },
    @{ Name = "AArch64-SVE"; Flags = @("-target", "aarch64-unknown-linux-gnu", "-march=armv8.4-a+sve") },
    @{ Name = "RISCV-RVV"; Flags = @("-target", "riscv64-unknown-linux-gnu", "-march=rv64gcv") }
)

$experiments = @(
    @{ Kind = "strict_fp"; File = "strict_fp.c"; ExtraFlags = @("-O3") },
    @{ Kind = "reassoc_enabled"; File = "reassoc_enabled.c"; ExtraFlags = @("-O3", "-fassociative-math", "-freciprocal-math") },
    @{ Kind = "loop_interchanged"; File = "loop_interchanged_control.c"; ExtraFlags = @("-O3") }
)

$results = @()

foreach ($tgt in $targets) {
    foreach ($exp in $experiments) {
        $srcPath = Join-Path $ProofDir $exp.File
        $outAsm = Join-Path $ProofDir "$($exp.Kind)_$($tgt.Name).s"
        $remarkLog = Join-Path $ProofDir "$($exp.Kind)_$($tgt.Name)_remarks.log"

        $cmd = @($ClangPath, "-S", $srcPath, "-o", $outAsm, "-Rpass=loop-vectorize", "-Rpass-missed=loop-vectorize", "-Rpass-analysis=loop-vectorize") + $exp.ExtraFlags + $tgt.Flags
        
        # Execute and capture stderr/stdout remarks
        $proc = Start-Process -FilePath $cmd[0] -ArgumentList $cmd[1..($cmd.Length - 1)] -NoNewWindow -PassThru -RedirectStandardError $remarkLog -Wait
        $exitCode = $proc.ExitCode
        $remarks = if (Test-Path $remarkLog) { Get-Content $remarkLog -Raw } else { "" }

        $vectorized = $false
        $missedReason = "None"
        $vectorWidth = "None"

        if ($remarks -match "vectorized loop \(vector width:? (\d+|scalable\[\d+\])") {
            $vectorized = $true
            $vectorWidth = $Matches[1]
        } elseif ($remarks -match "loop not vectorized: (.*)") {
            $vectorized = $false
            $missedReason = $Matches[1].Trim()
        } elseif ($remarks -match "cannot vectorize (.*)") {
            $vectorized = $false
            $missedReason = $Matches[1].Trim()
        }

        # Check assembly directly as secondary confirmation
        $asmContent = if (Test-Path $outAsm) { Get-Content $outAsm -Raw } else { "" }
        if (-not $vectorized) {
            if ($tgt.Name -like "*x86*" -or $tgt.Name -eq "AVX2" -or $tgt.Name -eq "AVX-512" -or $tgt.Name -eq "SSE4.2") {
                if ($asmContent -match "vaddps|addps") {
                    $vectorized = $true
                    $missedReason = "None"
                }
            } elseif ($tgt.Name -eq "AArch64-NEON" -and $asmContent -match "fadd\s+v\d+\.") {
                $vectorized = $true
                $missedReason = "None"
            } elseif ($tgt.Name -eq "AArch64-SVE" -and $asmContent -match "fadd\s+z\d+\.") {
                $vectorized = $true
                $missedReason = "None"
            } elseif ($tgt.Name -eq "RISCV-RVV" -and $asmContent -match "vfadd") {
                $vectorized = $true
                $missedReason = "None"
            }
        }

        $results += [PSCustomObject]@{
            Target = $tgt.Name
            Experiment = $exp.Kind
            Vectorized = $vectorized
            VectorWidth = $vectorWidth
            MissedReason = if ($vectorized) { "N/A (Successfully Vectorized)" } else { $missedReason }
            FPFlags = ($exp.ExtraFlags -join " ")
            TargetFlags = ($tgt.Flags -join " ")
            ExitCode = $exitCode
        }
    }
}

$results | Format-Table Target, Experiment, Vectorized, VectorWidth, MissedReason -AutoSize

$results | ConvertTo-Json -Depth 4 | Out-File -FilePath (Join-Path $ProofDir "RESULT.json") -Encoding utf8
Write-Output "Reassociation Proof Completed: RESULT.json written."
