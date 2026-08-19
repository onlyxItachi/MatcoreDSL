<#
.SYNOPSIS
    Exports compact fingerprint files for representative artifacts in the Windows lowering corpus.
.DESCRIPTION
    Scans corpus manifests and generates structured fingerprint JSONs in corpus/fingerprints/windows-x64/.
#>

param(
    [string]$ManifestV1Path = "$PSScriptRoot\..\..\manifests\windows-corpus-v1.json",
    [string]$ManifestV2Path = "$PSScriptRoot\..\..\manifests\windows-corpus-v2-archaeology.json",
    [string]$FingerprintDir = "$PSScriptRoot\..\..\fingerprints\windows-x64"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $FingerprintDir | Out-Null

$manifests = @($ManifestV1Path, $ManifestV2Path)
$fingerprints = @{}

foreach ($manPath in $manifests) {
    if (-not (Test-Path $manPath)) { continue }
    $manifest = Get-Content -Raw $manPath | ConvertFrom-Json

    foreach ($case in $manifest.cases) {
        $ver = $case.toolchain.version
        if (-not $fingerprints.ContainsKey($ver)) {
            $fingerprints[$ver] = @()
        }

        foreach ($art in $case.artifacts) {
            $entry = [ordered]@{
                case_id = $case.case_id
                stage = $art.stage
                artifact_type = $art.artifact_type
                relative_path = $art.relative_path
                sha256 = $art.sha256
                size_bytes = $art.size_bytes
            }
            $fingerprints[$ver] += $entry
        }
    }
}

foreach ($ver in $fingerprints.Keys) {
    $verDir = Join-Path $FingerprintDir "llvm-$ver"
    New-Item -ItemType Directory -Force -Path $verDir | Out-Null
    $fpPath = Join-Path $verDir "fingerprints.json"
    $fingerprints[$ver] | ConvertTo-Json -Depth 5 | Out-File -FilePath $fpPath -Encoding utf8
    Write-Host "Exported fingerprints for LLVM $ver to $fpPath ($($fingerprints[$ver].Count) entries)" -ForegroundColor Green
}
