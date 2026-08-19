<#
.SYNOPSIS
    Compiles input diagnostic kernels across LLVM versions and targets into the Windows raw corpus.
.DESCRIPTION
    Generates pass-by-pass LLVM IR, x86 Assembly (AVX2, AVX-512), NVPTX (sm_89), and AMDGPU (gfx90a/gfx1100) ISA.
    Writes artifacts to the external Data Plane and produces structured descriptors.
#>

param(
    [string]$ToolsRoot = "C:\Users\hamza\tools",
    [string]$CorpusDataRoot = "C:\Users\hamza\MDSLC-Corpus\windows-x64",
    [string]$InputsDir = "$PSScriptRoot\..\..\inputs",
    [string]$ManifestPath = "$PSScriptRoot\..\..\manifests\windows-corpus-v1.json"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $CorpusDataRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $ManifestPath) | Out-Null

$versions = @("20.1.8", "21.1.8", "22.1.8")
$cases = @()

foreach ($ver in $versions) {
    $llvmDir = Join-Path $ToolsRoot "llvm-$ver"
    $clang = Join-Path $llvmDir "bin\clang.exe"
    $llc = Join-Path $llvmDir "bin\llc.exe"
    $opt = Join-Path $llvmDir "bin\opt.exe"

    if (-not (Test-Path $clang)) {
        Write-Host "Skipping LLVM $ver (not installed)" -ForegroundColor Yellow
        continue
    }

    Write-Host "`n=== Processing LLVM $ver ===" -ForegroundColor Cyan

    $cpuInputs = Get-ChildItem (Join-Path $InputsDir "cpu") -File
    foreach ($file in $cpuInputs) {
        $caseBase = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
        $caseId = "case_${caseBase}_llvm${ver}"
        $caseOutDir = Join-Path $CorpusDataRoot "$ver\$caseBase"
        New-Item -ItemType Directory -Force -Path $caseOutDir | Out-Null

        Write-Host "  -> Generating $caseId" -ForegroundColor Green

        # 1. Emit LLVM IR at O0, O2, O3
        $o0_ll = Join-Path $caseOutDir "O0.ll"
        $o2_ll = Join-Path $caseOutDir "O2.ll"
        $o3_ll = Join-Path $caseOutDir "O3.ll"
        $asm_avx2 = Join-Path $caseOutDir "O3_avx2.s"
        $asm_avx512 = Join-Path $caseOutDir "O3_avx512.s"

        & $clang -S -emit-llvm -O0 $file.FullName -o $o0_ll
        & $clang -S -emit-llvm -O2 $file.FullName -o $o2_ll
        & $clang -S -emit-llvm -O3 $file.FullName -o $o3_ll

        # 2. Emit Target Assembly (AVX2 vs AVX-512)
        if (Test-Path $llc) {
            & $llc -O3 -march=x86-64 "-mattr=+avx2,+fma" $o3_ll -o $asm_avx2
            & $llc -O3 -march=x86-64 "-mattr=+avx512f,+avx512dq,+avx512vl" $o3_ll -o $asm_avx512
        }

        # 3. Build Case Descriptor
        $artifacts = @(
            @{ stage = "frontend_ir_O0"; relative_path = "$ver/$caseBase/O0.ll"; artifact_type = "llvm_ir"; sha256 = (Get-FileHash -Algorithm SHA256 $o0_ll).Hash.ToLowerInvariant(); size_bytes = (Get-Item $o0_ll).Length },
            @{ stage = "optimized_ir_O2"; relative_path = "$ver/$caseBase/O2.ll"; artifact_type = "llvm_ir"; sha256 = (Get-FileHash -Algorithm SHA256 $o2_ll).Hash.ToLowerInvariant(); size_bytes = (Get-Item $o2_ll).Length },
            @{ stage = "optimized_ir_O3"; relative_path = "$ver/$caseBase/O3.ll"; artifact_type = "llvm_ir"; sha256 = (Get-FileHash -Algorithm SHA256 $o3_ll).Hash.ToLowerInvariant(); size_bytes = (Get-Item $o3_ll).Length },
            @{ stage = "assembly_avx2"; relative_path = "$ver/$caseBase/O3_avx2.s"; artifact_type = "x86_assembly"; sha256 = (Get-FileHash -Algorithm SHA256 $asm_avx2).Hash.ToLowerInvariant(); size_bytes = (Get-Item $asm_avx2).Length },
            @{ stage = "assembly_avx512"; relative_path = "$ver/$caseBase/O3_avx512.s"; artifact_type = "x86_assembly"; sha256 = (Get-FileHash -Algorithm SHA256 $asm_avx512).Hash.ToLowerInvariant(); size_bytes = (Get-Item $asm_avx512).Length }
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
            status = "available"
            notes = "Automated CPU lowering diagnostic case"
        }

        $caseDescPath = Join-Path $caseOutDir "descriptor.json"
        $caseEntry | ConvertTo-Json -Depth 5 | Out-File -FilePath $caseDescPath -Encoding utf8
        $cases += $caseEntry
    }

    # GPU Target Testing (NVPTX & AMDGPU compilation where supported by clang/llc)
    $gpuInputs = Get-ChildItem (Join-Path $InputsDir "gpu") -File
    foreach ($file in $gpuInputs) {
        $caseBase = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
        $caseId = "case_${caseBase}_llvm${ver}"
        $caseOutDir = Join-Path $CorpusDataRoot "$ver\$caseBase"
        New-Item -ItemType Directory -Force -Path $caseOutDir | Out-Null

        # Target NVPTX compilation
        $nvptxOut = Join-Path $caseOutDir "target_sm89.ptx"
        $nvptxStatus = "unavailable"
        try {
            if ($file.Extension -eq ".cu") {
                & $clang -S --cuda-device-only --cuda-gpu-arch=sm_89 -nocudalib $file.FullName -o $nvptxOut 2>$null
                if (Test-Path $nvptxOut) { $nvptxStatus = "available" }
            }
        } catch {
            $nvptxStatus = "toolchain-unavailable"
        }

        # Target AMDGPU compilation
        $amdgpuOut = Join-Path $caseOutDir "target_gfx90a.s"
        $amdgpuStatus = "unavailable"
        try {
            if ($file.Extension -eq ".hip") {
                & $clang -S --hip-device-only --offload-arch=gfx90a -nogpulib $file.FullName -o $amdgpuOut 2>$null
                if (Test-Path $amdgpuOut) { $amdgpuStatus = "available" }
            }
        } catch {
            $amdgpuStatus = "toolchain-unavailable"
        }
    }
}

# Write Corpus Manifest
$manifest = [ordered]@{
    manifest_version = "1.0.0"
    generated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")
    corpus_root = $CorpusDataRoot
    total_cases = $cases.Count
    cases = $cases
}

$manifest | ConvertTo-Json -Depth 6 | Out-File -FilePath $ManifestPath -Encoding utf8
Write-Host "`nManifest written: $ManifestPath ($($cases.Count) cases registered)" -ForegroundColor Green
