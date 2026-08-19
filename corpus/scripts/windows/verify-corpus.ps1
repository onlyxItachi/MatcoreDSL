<#
.SYNOPSIS
    Verifies the integrity, SHA-256 hashes, and JSON schema compliance of all artifacts registered in the Windows corpus.
.DESCRIPTION
    Validates manifests against Draft 2020-12 schemas via validate-schemas.py, checks physical file existence, and verifies exact SHA-256 hashes.
#>

param(
    [string]$ManifestV1Path = "$PSScriptRoot\..\..\manifests\windows-corpus-v1.json",
    [string]$ManifestV2Path = "$PSScriptRoot\..\..\manifests\windows-corpus-v2-archaeology.json"
)

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Executing Strict JSON Schema Validation ===" -ForegroundColor Cyan
& python "$PSScriptRoot\validate-schemas.py"
if ($LASTEXITCODE -ne 0) {
    throw "JSON Schema Validation failed."
}

$manifests = @($ManifestV1Path, $ManifestV2Path)
$totalPassed = 0
$totalFailed = 0

Write-Host "`n=== 2. Executing Physical Artifact SHA-256 Integrity Audit ===" -ForegroundColor Cyan

foreach ($manPath in $manifests) {
    if (-not (Test-Path $manPath)) {
        Write-Host "Skipping manifest (not found): $manPath" -ForegroundColor Yellow
        continue
    }

    $manifest = Get-Content -Raw $manPath | ConvertFrom-Json
    $corpusRoot = $manifest.corpus_root
    Write-Host "Verifying $($manPath) at $corpusRoot ($($manifest.total_cases) cases)..." -ForegroundColor Yellow

    foreach ($case in $manifest.cases) {
        foreach ($art in $case.artifacts) {
            $fullPath = Join-Path $corpusRoot $art.relative_path
            if (-not (Test-Path $fullPath)) {
                Write-Host "  [FAIL] Missing artifact: $fullPath" -ForegroundColor Red
                $totalFailed++
                continue
            }
            $actualHash = (Get-FileHash -Algorithm SHA256 $fullPath).Hash.ToLowerInvariant()
            if ($actualHash -ne $art.sha256) {
                Write-Host "  [FAIL] Hash mismatch for $fullPath" -ForegroundColor Red
                $totalFailed++
            } else {
                $totalPassed++
            }
        }
    }
}

Write-Host "`n=== Final Verification Summary ===" -ForegroundColor Cyan
Write-Host "Artifacts Verified (SHA-256 Match): $totalPassed" -ForegroundColor Green
Write-Host "Artifacts Failed: $totalFailed" -ForegroundColor $(if ($totalFailed -gt 0) { "Red" } else { "Green" })

if ($totalFailed -gt 0) {
    exit 1
} else {
    Write-Host "Corpus Integrity & Schema Compliance: 100% VERIFIED." -ForegroundColor Green
}
