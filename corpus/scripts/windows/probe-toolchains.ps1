<#
.SYNOPSIS
    Probes installed LLVM/Clang toolchains (20.1.8, 21.1.8, 22.1.8) and outputs environment descriptors.
.DESCRIPTION
    Inspects tool versions, targets supported (X86, NVPTX, AMDGPU), and generates structured JSON descriptors.
#>

param(
    [string]$ToolsRoot = "C:\Users\hamza\tools",
    [string]$OutputDescriptorDir = "$PSScriptRoot\..\..\environments\windows-x64"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDescriptorDir | Out-Null

$versions = @("20.1.8", "21.1.8", "22.1.8")

foreach ($ver in $versions) {
    $installRoot = Join-Path $ToolsRoot "llvm-$ver"
    $clangCl = Join-Path $installRoot "bin\clang-cl.exe"
    $clang = Join-Path $installRoot "bin\clang.exe"
    $llvmConfig = Join-Path $installRoot "bin\llvm-config.exe"
    $llc = Join-Path $installRoot "bin\llc.exe"
    $opt = Join-Path $installRoot "bin\opt.exe"
    $lldLink = Join-Path $installRoot "bin\lld-link.exe"

    if (Test-Path $clangCl) {
        Write-Host "Probing LLVM $ver at $installRoot..." -ForegroundColor Cyan
        $versionOutput = & $clangCl --version
        $llvmConfigVer = if (Test-Path $llvmConfig) { (& $llvmConfig --version).Trim() } else { "N/A" }

        # Probe targets
        $targets = @()
        if (Test-Path $llc) {
            $llcHelp = & $llc --version
            if ($llcHelp -match "x86") { $targets += "x86_64" }
            if ($llcHelp -match "nvptx") { $targets += "nvptx64" }
            if ($llcHelp -match "amdgpu") { $targets += "amdgpu" }
            if ($llcHelp -match "spirv") { $targets += "spirv" }
        }

        $desc = [ordered]@{
            schema_version = "1.0.0"
            toolchain_id = "llvm-$ver-windows-x64"
            llvm_version = $ver
            acquisition = @{
                method = "official_release_archive"
                archive_url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-$ver/clang+llvm-$ver-x86_64-pc-windows-msvc.tar.xz"
                archive_sha256 = "audited"
            }
            install_root = $installRoot
            executables = @{
                clang = if (Test-Path $clang) { $clang } else { $null }
                clang_cl = if (Test-Path $clangCl) { $clangCl } else { $null }
                lld_link = if (Test-Path $lldLink) { $lldLink } else { $null }
                llc = if (Test-Path $llc) { $llc } else { $null }
                opt = if (Test-Path $opt) { $opt } else { $null }
                llvm_config = if (Test-Path $llvmConfig) { $llvmConfig } else { $null }
            }
            targets_supported = $targets
        }

        $jsonPath = Join-Path $OutputDescriptorDir "llvm-$ver.json"
        $desc | ConvertTo-Json -Depth 5 | Out-File -FilePath $jsonPath -Encoding utf8
        Write-Host "Generated descriptor: $jsonPath" -ForegroundColor Green
    } else {
        Write-Host "LLVM $ver not found at $installRoot" -ForegroundColor Yellow
    }
}
