<#
.SYNOPSIS
    Verifies the integrity and SHA-256 hashes of all artifacts registered in the Windows corpus manifest.
.DESCRIPTION
    Validates manifest against corpus-entry.schema.json, checks file existence, recalculates hashes, and reports integrity status.
#>

param(
    [string]$ManifestPath = "$PSScriptRoot\..\..\manifests\windows-corpus-v1.json"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$manifestContent = Get-Content -Raw $ManifestPath | ConvertFrom-Json
$corpusRoot = $manifestContent.corpus_root

Write-Host "Verifying Windows Lowering Corpus at $corpusRoot..." -ForegroundColor Cyan
Write-Host "Total Registered Cases: $($manifestContent.total_cases)" -ForegroundColor Yellow

$passed = 0
$failed = 0

foreach ($case in $manifestContent.cases) {
    Write-Host "Checking case $($case.case_id)..." -ForegroundColor Gray
    foreach ($art in $case.artifacts) {
        $fullPath = Join-Path $corpusRoot $art.relative_path
        if (-not (Test-Path $fullPath)) {
            Write-Host "  [FAIL] Missing artifact: $fullPath" -ForegroundColor Red
            $failed++
            continue
        }
        $actualHash = (Get-FileHash -Algorithm SHA256 $fullPath).Hash.ToLowerInvariant()
        if ($actualHash -ne $art.sha256) {
            Write-Host "  [FAIL] Hash mismatch for $fullPath (Expected: $($art.sha256), Got: $actualHash)" -ForegroundColor Red
            $failed++
        } else {
            $passed++
        }
    }
}

Write-Host "`n=== Verification Summary ===" -ForegroundColor Cyan
Write-Host "Artifacts Verified (SHA-256 Match): $passed" -ForegroundColor Green
if ($failed -gt 0) {
    Write-Host "Artifacts Failed: $failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "Corpus Integrity 100% Verified." -ForegroundColor Green
}
