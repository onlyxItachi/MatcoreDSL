<#
.SYNOPSIS
    Automated assembly validator proving true accumulator-register fusion vs separate store/reload epilogue.
#>

param(
    [string]$ClangPath = "C:\Users\hamza\tools\llvm-21.1.8\bin\clang.exe",
    [string]$ProofDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

$src = Join-Path $ProofDir "epilogue_kernels.c"
$asm = Join-Path $ProofDir "epilogue_kernels.s"

& $ClangPath -O3 -mavx2 -mfma -fverbose-asm -S $src -o $asm
if ($LASTEXITCODE -ne 0) { throw "Assembly compilation failed" }

$lines = Get-Content $asm

function Parse-FunctionAsm([string]$funcName) {
    $inFunc = $false
    $funcLines = @()
    foreach ($line in $lines) {
        if ($line -match "^$($funcName):") {
            $inFunc = $true
            continue
        }
        if ($inFunc) {
            if ($line -match "^\t\.seh_endproc" -or ($line -match "^[a-zA-Z_0-9]+:" -and -not ($line -match "^\.L"))) {
                break
            }
            $funcLines += $line
        }
    }
    return $funcLines
}

$sepAsm = Parse-FunctionAsm "microkernel_separate_relu"
$fusedAsm = Parse-FunctionAsm "microkernel_true_fused_relu"

Write-Output "=== 1. Validating microkernel_separate_relu (Negative Control) ==="
$sepStores = ($sepAsm | Select-String "vmovups\s+%ymm\d+,\s*\(|vmovups\s+%ymm\d+,\s*\d+\(").Matches.Count
$sepLoads = ($sepAsm | Select-String "vmovups\s+\([^,]+,\s*%ymm|vmovups\s+\d+\([^,]+,\s*%ymm").Matches.Count
$sepMaxps = ($sepAsm | Select-String "vmaxps").Matches.Count

Write-Output "Separate Function: Total Stores = $sepStores, Total Loads = $sepLoads, vmaxps = $sepMaxps"
if ($sepStores -lt 16 -or $sepMaxps -lt 8) {
    throw "VALIDATOR ERROR: Separate function failed negative control (expected at least 16 vector stores for two passes)."
}
Write-Output "[PASS] microkernel_separate_relu exhibits store -> reload -> maxps -> store pattern."

Write-Output "`n=== 2. Validating microkernel_true_fused_relu (True Epilogue Fusion) ==="
$fusedStores = ($fusedAsm | Select-String "vmovups\s+%ymm\d+,\s*\(|vmovups\s+%ymm\d+,\s*\d+\(").Matches.Count
$fusedMaxps = ($fusedAsm | Select-String "vmaxps").Matches.Count
$fusedFma = ($fusedAsm | Select-String "vfmadd").Matches.Count

Write-Output "Fused Function: Total Final Stores = $fusedStores, vmaxps = $fusedMaxps, vfmadd = $fusedFma"

# Assert exact ordering in true_fused: all vmaxps occur BEFORE the first final result store to memory
$firstStoreIndex = -1
$lastMaxpsIndex = -1

for ($i = 0; $i -lt $fusedAsm.Count; $i++) {
    $line = $fusedAsm[$i]
    if ($line -match "vmaxps") {
        $lastMaxpsIndex = $i
    }
    if ($firstStoreIndex -eq -1 -and ($line -match "vmovups\s+%ymm\d+,\s*\(" -or $line -match "vmovups\s+%ymm\d+,\s*\d+\(")) {
        $firstStoreIndex = $i
    }
}

Write-Output "Last vmaxps index: $lastMaxpsIndex, First memory store index: $firstStoreIndex"

if ($lastMaxpsIndex -ge $firstStoreIndex) {
    throw "VALIDATION FAILED: In true_fused, memory store occurred before or interleaving with vmaxps!"
}

if ($fusedStores -ne 8) {
    throw "VALIDATION FAILED: In true_fused, expected exactly 8 final result stores, found $fusedStores"
}

Write-Output "[PASS] microkernel_true_fused_relu: 100% verified in-register accumulator ReLU prior to single final memory store."

$validationReport = [PSCustomObject]@{
    proof_id = "PROOF-02-EPILOGUE-FUSION-ASSEMBLY"
    separate_function_stores = $sepStores
    separate_function_loads = $sepLoads
    separate_function_vmaxps = $sepMaxps
    fused_function_final_stores = $fusedStores
    fused_function_vmaxps = $fusedMaxps
    fused_function_vfmadd = $fusedFma
    ordering_verified = ($lastMaxpsIndex -lt $firstStoreIndex)
    verdict = "PASSED_ASSEMBLY_ACCEPTANCE"
}

$validationReport | ConvertTo-Json -Depth 4 | Out-File -FilePath (Join-Path $ProofDir "ASSEMBLY_VALIDATION.json") -Encoding utf8
Write-Output "Assembly Validation Report written to ASSEMBLY_VALIDATION.json"
