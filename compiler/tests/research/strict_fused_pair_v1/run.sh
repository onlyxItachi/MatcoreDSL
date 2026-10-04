#!/usr/bin/env bash
# Standalone research reproduction. No MDSLC build or production issuer use.
set -euo pipefail
fused_source_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fused_repo_dir=$(CDPATH= cd -- "$fused_source_dir/../../../.." && pwd)
fused_prefix=${1:-/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21}
fused_build=${2:-$fused_repo_dir/build-research-fused-pair}
fused_clang=/usr/bin/clang++-21
fused_opt=$fused_prefix/bin/mlir-opt
fused_translate=$fused_prefix/bin/mlir-translate
"$fused_opt" --version | rg -q 'version 21\.1\.8'
"$fused_translate" --version | rg -q 'version 21\.1\.8'
"$fused_clang" --version | rg -q 'version 21\.1\.8'
mkdir -p -- "$fused_build"

# One small compile/link at a time; the MLIR tool links the shared DSO.
"$fused_clang" -std=c++17 -O0 "$fused_source_dir/workspace_bufferize.cpp" \
  -I"$fused_prefix/include" -I/usr/lib/llvm-21/include \
  "$fused_prefix/lib/libMLIR.so.21.1" -L/usr/lib/llvm-21/lib -lLLVM-21 \
  -Wl,-rpath,"$fused_prefix/lib" -o "$fused_build/workspace-bufferize"
"$fused_opt" "$fused_source_dir/pair.mlir" --transform-interpreter \
  --test-transform-dialect-erase-schedule --canonicalize -o "$fused_build/scheduled.mlir"
"$fused_build/workspace-bufferize" "$fused_build/scheduled.mlir" > "$fused_build/workspace-raw.mlir"
"$fused_opt" "$fused_build/workspace-raw.mlir" --canonicalize --cse \
  --canonicalize --drop-equivalent-buffer-results -o "$fused_build/workspace.mlir"
if rg -q 'memref\.(alloc|alloca|realloc|dealloc|copy)|bufferization\.' "$fused_build/workspace.mlir"; then
  printf 'unexpected storage operation after workspace mapping\n' >&2
  exit 1
fi
"$fused_opt" "$fused_build/workspace.mlir" --convert-linalg-to-loops \
  --expand-strided-metadata --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
  --convert-cf-to-llvm --reconcile-unrealized-casts -o "$fused_build/workspace-llvm.mlir"
"$fused_translate" --mlir-to-llvmir "$fused_build/workspace-llvm.mlir" -o "$fused_build/workspace.ll"
[[ $(rg -c ' = fmul float ' "$fused_build/workspace.ll") -eq 2 ]]
[[ $(rg -c ' = fadd float ' "$fused_build/workspace.ll") -eq 2 ]]
if rg -q 'llvm\.fma| (fast|contract|reassoc|nnan|ninf|nsz|arcp|afn) |noalias' "$fused_build/workspace.ll"; then
  printf 'unexpected arithmetic/alias permission\n' >&2
  exit 1
fi
if rg '^[[:space:]]+.*call |^declare ' "$fused_build/workspace.ll" | \
    rg -v '@llvm.smin.i64|@research_strict_fused_pair'; then
  printf 'unexpected LLVM call or declaration\n' >&2
  exit 1
fi
"$fused_clang" -x ir -O2 -ffp-contract=off -c "$fused_build/workspace.ll" -o "$fused_build/workspace.o"
"$fused_clang" -std=c++20 -O2 -ffp-contract=off -fno-fast-math \
  "$fused_source_dir/execute.cpp" "$fused_build/workspace.o" -o "$fused_build/execute"
"$fused_build/execute"

# Default One-Shot allocation is intentionally not accepted for production.
"$fused_opt" "$fused_build/scheduled.mlir" \
  --one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
  --canonicalize --cse --canonicalize --drop-equivalent-buffer-results \
  --buffer-deallocation-pipeline --canonicalize -o "$fused_build/default-allocation.mlir"
rg -q 'memref.alloc' "$fused_build/default-allocation.mlir"
rg -q 'memref.dealloc' "$fused_build/default-allocation.mlir"
for fused_negative in __no_empty_fold __column_recompute; do
  "$fused_opt" "$fused_source_dir/pair.mlir" \
    --transform-interpreter="entry-point=$fused_negative" \
    --test-transform-dialect-erase-schedule --canonicalize -o "$fused_build/$fused_negative.mlir"
  if "$fused_build/workspace-bufferize" "$fused_build/$fused_negative.mlir" \
      > "$fused_build/$fused_negative-rejected.mlir" 2> "$fused_build/$fused_negative-rejection.log"; then
    printf 'negative schedule unexpectedly mapped: %s\n' "$fused_negative" >&2
    exit 1
  fi
  rg -q 'rejected request shape/origin' "$fused_build/$fused_negative-rejection.log"
  printf 'rejected %s\n' "$fused_negative"
done
printf 'Object undefined symbols (O2 may lower fills to memset):\n'
nm -u "$fused_build/workspace.o"
sha256sum "$fused_build/scheduled.mlir" "$fused_build/workspace.mlir" \
  "$fused_build/workspace.ll" "$fused_build/workspace.o"
