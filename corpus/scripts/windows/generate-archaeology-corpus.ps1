<#
.SYNOPSIS
    Generates fine-grained optimization pass checkpoints, optimization remarks, and true target-aware lowering lanes.
.DESCRIPTION
    Builds the expanded evidence layer across LLVM 20.1.8, 21.1.8, and 22.1.8 on Windows x64.
#>

param(
    [string]$ToolsRoot = "C:\Users\hamza\tools",
    [string]$CorpusDataRoot = "C:\Users\hamza\MDSLC-Corpus\windows-x64",
    [string]$InputsDir = "$PSScriptRoot\..\..\inputs",
    [string]$ManifestPath = "$PSScriptRoot\..\..\manifests\windows-corpus-v2-archaeology.json"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $CorpusDataRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $ManifestPath) | Out-Null

$versions = @("20.1.8", "21.1.8", "22.1.8")
$cases = @()

foreach ($ver in $versions) {
    $llvmDir = Join-Path $ToolsRoot "llvm-$ver"
    $clang = Join-Path $llvmDir "bin\clang.exe"
    $opt = Join-Path $llvmDir "bin\opt.exe"
    $llc = Join-Path $llvmDir "bin\llc.exe"

    if (-not (Test-Path $clang)) {
        Write-Host "Skipping LLVM $ver (not installed)" -ForegroundColor Yellow
        continue
    }

    Write-Host "`n=== Generating Archaeology Corpus for LLVM $ver ===" -ForegroundColor Cyan

    $cpuInputs = Get-ChildItem (Join-Path $InputsDir "cpu") -File
    foreach ($file in $cpuInputs) {
        $caseBase = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
        $caseId = "archaeology_${caseBase}_llvm${ver}"
        $caseOutDir = Join-Path $CorpusDataRoot "$ver\$caseBase\archaeology"
        New-Item -ItemType Directory -Force -Path $caseOutDir | Out-Null

        Write-Host "  -> Processing $caseId" -ForegroundColor Green

        # 1. Pass Checkpoints
        $raw_ll = Join-Path $caseOutDir "01_frontend_raw.ll"
        $sroa_ll = Join-Path $caseOutDir "02_sroa_normalized.ll"
        $opt_ll = Join-Path $caseOutDir "03_final_optimized.ll"
        $remarks_txt = Join-Path $caseOutDir "vectorization_remarks.txt"

        # Emit raw frontend IR without optnone
        & $clang -S -emit-llvm -Xclang -disable-O0-optnone -O0 $file.FullName -o $raw_ll
        # Run SROA / mem2reg normalization
        if (Test-Path $opt) {
            & $opt -S -passes="mem2reg,sroa" $raw_ll -o $sroa_ll
        }
        # Emit generic O3 optimized IR
        & $clang -S -emit-llvm -O3 $file.FullName -o $opt_ll

        # Capture Vectorization Remarks
        $remarks = & $clang -O3 -mavx2 -mfma -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -c $file.FullName -o "$env:TEMP\remark_temp.o" 2>&1
        $remarks | Out-File -FilePath $remarks_txt -Encoding utf8

        # 2. True Target-Aware AVX2 / FMA Lane
        $avx2Dir = Join-Path $caseOutDir "target_aware_avx2"
        New-Item -ItemType Directory -Force -Path $avx2Dir | Out-Null
        $avx2_ll = Join-Path $avx2Dir "O3_target_aware_avx2.ll"
        $avx2_s = Join-Path $avx2Dir "O3_target_aware_avx2.s"

        & $clang -S -emit-llvm -O3 -mavx2 -mfma $file.FullName -o $avx2_ll
        & $clang -S -O3 -mavx2 -mfma $file.FullName -o $avx2_s

        $avx2_content = Get-Content $avx2_s
        $avx2_ymm = ($avx2_content | Select-String "%ymm" -AllMatches).Matches.Count
        $avx2_xmm = ($avx2_content | Select-String "%xmm" -AllMatches).Matches.Count

        # 3. True Target-Aware AVX-512 Lane
        $avx512Dir = Join-Path $caseOutDir "target_aware_avx512"
        New-Item -ItemType Directory -Force -Path $avx512Dir | Out-Null
        $avx512_ll = Join-Path $avx512Dir "O3_target_aware_avx512.ll"
        $avx512_s = Join-Path $avx512Dir "O3_target_aware_avx512.s"

        & $clang -S -emit-llvm -O3 -mavx512f -mavx512dq -mavx512vl $file.FullName -o $avx512_ll
        & $clang -S -O3 -mavx512f -mavx512dq -mavx512vl $file.FullName -o $avx512_s

        $avx512_content = Get-Content $avx512_s
        $avx512_zmm = ($avx512_content | Select-String "%zmm" -AllMatches).Matches.Count
        $avx512_ymm = ($avx512_content | Select-String "%ymm" -AllMatches).Matches.Count
        $avx512_xmm = ($avx512_content | Select-String "%xmm" -AllMatches).Matches.Count

        # Register Artifacts
        $artifacts = @(
            @{ stage = "frontend_raw"; relative_path = "$ver/$caseBase/archaeology/01_frontend_raw.ll"; artifact_type = "llvm_ir"; sha256 = (Get-FileHash -Algorithm SHA256 $raw_ll).Hash.ToLowerInvariant(); size_bytes = (Get-Item $raw_ll).Length },
            @{ stage = "sroa_normalized"; relative_path = "$ver/$caseBase/archaeology/02_sroa_normalized.ll"; artifact_type = "llvm_ir"; sha256 = (Get-FileHash -Algorithm SHA256 $sroa_ll).Hash.ToLowerInvariant(); size_bytes = (Get-Item $sroa_ll).Length },
            @{ stage = "final_optimized"; relative_path = "$ver/$caseBase/archaeology/03_final_optimized.ll"; artifact_type = "llvm_ir"; sha256 = (Get-FileHash -Algorithm SHA256 $opt_ll).Hash.ToLowerInvariant(); size_bytes = (Get-Item $opt_ll).Length },
            @{ stage = "vectorization_remarks"; relative_path = "$ver/$caseBase/archaeology/vectorization_remarks.txt"; artifact_type = "diagnostics"; sha256 = (Get-FileHash -Algorithm SHA256 $remarks_txt).Hash.ToLowerInvariant(); size_bytes = (Get-Item $remarks_txt).Length },
            @{ stage = "target_aware_avx2_ir"; relative_path = "$ver/$caseBase/archaeology/target_aware_avx2/O3_target_aware_avx2.ll"; artifact_type = "llvm_ir"; sha256 = (Get-FileHash -Algorithm SHA256 $avx2_ll).Hash.ToLowerInvariant(); size_bytes = (Get-Item $avx2_ll).Length },
            @{ stage = "target_aware_avx2_asm"; relative_path = "$ver/$caseBase/archaeology/target_aware_avx2/O3_target_aware_avx2.s"; artifact_type = "x86_assembly"; sha256 = (Get-FileHash -Algorithm SHA256 $avx2_s).Hash.ToLowerInvariant(); size_bytes = (Get-Item $avx2_s).Length },
            @{ stage = "target_aware_avx512_ir"; relative_path = "$ver/$caseBase/archaeology/target_aware_avx512/O3_target_aware_avx512.ll"; artifact_type = "llvm_ir"; sha256 = (Get-FileHash -Algorithm SHA256 $avx512_ll).Hash.ToLowerInvariant(); size_bytes = (Get-Item $avx512_ll).Length },
            @{ stage = "target_aware_avx512_asm"; relative_path = "$ver/$caseBase/archaeology/target_aware_avx512/O3_target_aware_avx512.s"; artifact_type = "x86_assembly"; sha256 = (Get-FileHash -Algorithm SHA256 $avx512_s).Hash.ToLowerInvariant(); size_bytes = (Get-Item $avx512_s).Length }
        )

        $caseEntry = [ordered]@{
            schema_version = "1.0.0"
            case_id = $caseId
            lane = "cpu_native"
            host = @{ os = "Windows 11"; architecture = "x86_64"; windows_build = "26100" }
            toolchain = @{ name = "LLVM"; version = $ver; compiler_path = $clang; llvm_config_version = $ver }
            target = @{ triple = "x86_64-pc-windows-msvc"; cpu_or_gpu = "x86-64"; features = @("avx2", "fma", "avx512") }
            input_kernel = @{ path = "inputs/cpu/$($file.Name)"; language = "C/C++"; sha256 = (Get-FileHash -Algorithm SHA256 $file.FullName).Hash.ToLowerInvariant() }
            artifacts = $artifacts
            semantic_summary = @{
                operation = "gemm"
                dtype = "f32"
                accumulation_dtype = "f32"
                shape_contract = "M_N_K"
                memory_space = "host_global"
                aliasing_contract = if ($file.Name -match "restrict") { "noalias" } else { "may_alias" }
                vectorization_width = if ($file.Name -match "tiled") { "256bit_avx2_512bit_avx512" } else { "256bit_outer_j" }
                fma_formation = "supported"
            }
            status = "available"
            notes = "Archaeology pass checkpoints + remarks + target-aware AVX2/AVX512 lowering"
        }

        $caseDescPath = Join-Path $caseOutDir "descriptor.json"
        $caseEntry | ConvertTo-Json -Depth 6 | Out-File -FilePath $caseDescPath -Encoding utf8
        $cases += $caseEntry
    }
}

# Write Manifest
$manifest = [ordered]@{
    manifest_version = "2.0.0"
    generated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")
    corpus_root = $CorpusDataRoot
    total_cases = $cases.Count
    cases = $cases
}

$manifest | ConvertTo-Json -Depth 7 | Out-File -FilePath $ManifestPath -Encoding utf8
Write-Host "`nArchaeology Manifest written: $ManifestPath ($($cases.Count) cases registered)" -ForegroundColor Green
