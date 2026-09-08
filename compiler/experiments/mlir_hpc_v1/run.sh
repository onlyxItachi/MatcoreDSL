#!/usr/bin/env bash
# Research-only CLI reproduction. Generated files stay in a private temp dir or
# the explicitly supplied build directory. No product target/issuer is changed.
set -euo pipefail
# Symbolization must not block on an external debuginfod service.
export DEBUGINFOD_URLS=
experiment_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mlir_bin=${MDSLC_RESEARCH_MLIR_BIN:-/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/bin}
clangxx=${MDSLC_RESEARCH_CLANGXX:-/usr/bin/clang++-21}
clang=${MDSLC_RESEARCH_CLANG:-/usr/bin/clang-21}
opt=${MDSLC_RESEARCH_OPT:-/usr/bin/opt-21}
out=${1:-$(mktemp -d /tmp/mdslc-mlir-hpc-reproduce.XXXXXX)}
mkdir -p -- "$out"
"$mlir_bin/mlir-opt" --version
"$clang" --version
lower=(--canonicalize --lower-vector-multi-reduction --lower-vector-mask
       --convert-vector-to-scf --convert-linalg-to-loops
       --expand-strided-metadata --lower-affine --convert-scf-to-cf
       --convert-vector-to-llvm=vector-contract-lowering=parallelarith
       --convert-arith-to-llvm --convert-ub-to-llvm --finalize-memref-to-llvm
       --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts)
strict=(-ffp-contract=off -frounding-math -ftrapping-math)
for variant in tiled_vector tiled_interchange_vector tiled_mkn; do
  mkdir -p -- "$out/$variant"
  "$mlir_bin/mlir-opt" "$experiment_dir/$variant.mlir" \
    --transform-interpreter --test-transform-dialect-erase-schedule \
    -o "$out/$variant/scheduled.mlir"
  "$mlir_bin/mlir-opt" "$out/$variant/scheduled.mlir" "${lower[@]}" \
    -o "$out/$variant/lowered.mlir"
  "$mlir_bin/mlir-translate" --mlir-to-llvmir "$out/$variant/lowered.mlir" \
    -o "$out/$variant/leaf.ll"
  if grep -Eq 'llvm\.(fma|fmuladd|vector\.reduce\.fadd)|f(mul|add) (fast|reassoc|nnan|ninf|nsz|arcp|contract|afn)' "$out/$variant/leaf.ll"; then
    echo "unexpected numerical permission / fused instruction in leaf" >&2
    exit 1
  fi
  "$clang" -O3 -Wno-override-module -ffp-contract=off -c "$out/$variant/leaf.ll" \
    -o "$out/$variant/leaf.o"
  "$clangxx" -std=c++20 -O2 -Wall -Wextra -Werror "${strict[@]}" \
    "$experiment_dir/execution.cpp" "$out/$variant/leaf.o" -o "$out/$variant/execution"
  "$out/$variant/execution"
  # LLVM IR needs sanitize_address on its functions: a frontend flag alone does
  # not create that attribute on already generated IR. This is test-only input
  # instrumentation, never a production semantic rewrite.
  "$opt" "$out/$variant/leaf.ll" -S \
    -passes=forceattrs -force-attribute=sanitize_address \
    -o "$out/$variant/sanitized.ll"
  "$clang" -O1 -g -Wno-override-module -ffp-contract=off -fsanitize=address \
    -c "$out/$variant/sanitized.ll" -o "$out/$variant/sanitized.o"
  "$clangxx" -std=c++20 -O1 -g -Wall -Wextra -Werror "${strict[@]}" \
    -fsanitize=address,undefined "$experiment_dir/execution.cpp" \
    "$out/$variant/sanitized.o" -o "$out/$variant/execution-asan"
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "$out/$variant/execution-asan"
  set +e
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "$out/$variant/execution-asan" --asan-invalid-capacity \
    >"$out/$variant/asan-negative.stdout" 2>"$out/$variant/asan-negative.stderr"
  code=$?
  set -e
  if [[ "$code" == 0 || "$code" == 99 ]] ||
     ! grep -q 'AddressSanitizer: heap-buffer-overflow' "$out/$variant/asan-negative.stderr" ||
     ! grep -q 'research_gemm' "$out/$variant/asan-negative.stderr"; then
    echo "generated-load ASan negative control did not establish instrumentation" >&2
    exit 1
  fi
done
mkdir -p -- "$out/contract-counterexample"
"$mlir_bin/mlir-opt" "$experiment_dir/negative_contract.mlir" \
  --transform-interpreter --test-transform-dialect-erase-schedule --canonicalize \
  --convert-vector-to-scf --convert-vector-to-llvm=vector-contract-lowering=outerproduct \
  --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
  --convert-arith-to-llvm --convert-ub-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts \
  -o "$out/contract-counterexample/lowered.mlir"
"$mlir_bin/mlir-translate" --mlir-to-llvmir "$out/contract-counterexample/lowered.mlir" \
  -o "$out/contract-counterexample/leaf.ll"
grep -q llvm.fmuladd "$out/contract-counterexample/leaf.ll"
"$clangxx" -std=c++20 -O2 -Wno-override-module "${strict[@]}" \
  "$experiment_dir/contract_counterexample.cpp" "$out/contract-counterexample/leaf.ll" \
  -o "$out/contract-counterexample/baseline"
"$out/contract-counterexample/baseline"
# Real FMA execution requires an explicitly selected, previously validated host.
# Do not infer OS/hardware execution authority just from one /proc/cpuinfo flag.
if [[ ${MDSLC_RESEARCH_RUN_V3:-0} == 1 ]]; then
  "$clangxx" -std=c++20 -O2 -march=x86-64-v3 -Wno-override-module "${strict[@]}" \
    "$experiment_dir/contract_counterexample.cpp" "$out/contract-counterexample/leaf.ll" \
    -o "$out/contract-counterexample/v3"
  set +e
  "$out/contract-counterexample/v3"
  code=$?
  set -e
  if [[ "$code" != 1 ]]; then
    echo "expected specific strict-FMA mismatch did not reproduce" >&2
    exit 1
  fi
fi
printf 'Research artifacts: %s\n' "$out"
