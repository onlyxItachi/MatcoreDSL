#!/usr/bin/env bash
# Research-only reproduction. Does not modify a product target or issue a source
# candidate. Explicit v3 opt-in is required because the oracle expects real FMA.
set -euo pipefail
export DEBUGINFOD_URLS=
if [[ ${MDSLC_RESEARCH_RUN_V3:-0} != 1 ]]; then
  echo 'Set MDSLC_RESEARCH_RUN_V3=1 only on an independently validated x86-64-v3 host.' >&2
  exit 2
fi
experiment_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mlir_bin=${MDSLC_RESEARCH_MLIR_BIN:-/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/bin}
clangxx=${MDSLC_RESEARCH_CLANGXX:-/usr/bin/clang++-21}
clang=${MDSLC_RESEARCH_CLANG:-/usr/bin/clang-21}
opt=${MDSLC_RESEARCH_OPT:-/usr/bin/opt-21}
out=${1:-$(mktemp -d /tmp/mdslc-reassociate-register.XXXXXX)}
mkdir -p -- "$out"
"$mlir_bin/mlir-opt" --version
"$clang" --version
strict=(-ffp-contract=off -frounding-math -ftrapping-math)

"$mlir_bin/mlir-opt" "$experiment_dir/peeled_contract.mlir" \
  --transform-interpreter --canonicalize \
  --transform-interpreter=entry-point=vectorize --canonicalize \
  --convert-linalg-to-loops --fold-memref-alias-ops --canonicalize \
  --transform-interpreter=entry-point=hoist \
  --test-transform-dialect-erase-schedule -o "$out/scheduled.mlir"
grep -q 'iter_args.*vector<4x8xf32>' "$out/scheduled.mlir"
grep -q vector.outerproduct "$out/scheduled.mlir"
if grep -Eq 'memref\.(alloc|alloca|copy)|vector\.mask|fastmath' "$out/scheduled.mlir"; then
  echo 'Unexpected allocation, masked contraction, or numerical permission.' >&2
  exit 1
fi
"$mlir_bin/mlir-opt" "$out/scheduled.mlir" \
  --convert-vector-to-scf --expand-strided-metadata --lower-affine \
  --convert-scf-to-cf --convert-vector-to-llvm=vector-contract-lowering=outerproduct \
  --convert-arith-to-llvm --convert-ub-to-llvm --finalize-memref-to-llvm \
  --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts \
  -o "$out/lowered.mlir"
"$mlir_bin/mlir-translate" --mlir-to-llvmir "$out/lowered.mlir" -o "$out/leaf.ll"
grep -q llvm.fmuladd.v8f32 "$out/leaf.ll"
if grep -Eq 'noalias|\b(fast|reassoc|nnan|ninf|nsz|arcp|afn|contract)\b' "$out/leaf.ll"; then
  echo 'Unexpected alias claim or blanket fast-math flag.' >&2
  exit 1
fi
"$clang" -O3 -march=x86-64-v3 -Wno-override-module -ffp-contract=off \
  -c "$out/leaf.ll" -o "$out/leaf.o"
objdump -d "$out/leaf.o" > "$out/leaf.asm"
grep -q vfmadd "$out/leaf.asm"
"$clangxx" -std=c++20 -O2 -Wall -Wextra -Werror "${strict[@]}" \
  "$experiment_dir/execution.cpp" "$out/leaf.o" -o "$out/execution"
"$out/execution"

# Existing IR must actually be instrumented: clang -fsanitize on IR by itself
# does not add sanitize_address to its already emitted function definitions.
"$opt" "$out/leaf.ll" -S -passes=forceattrs -force-attribute=sanitize_address \
  -o "$out/sanitized.ll"
"$clang" -O1 -g -march=x86-64-v3 -Wno-override-module -ffp-contract=off \
  -fsanitize=address -c "$out/sanitized.ll" -o "$out/sanitized.o"
"$clangxx" -std=c++20 -O1 -g -Wall -Wextra -Werror "${strict[@]}" \
  -fsanitize=address,undefined "$experiment_dir/execution.cpp" \
  "$out/sanitized.o" -o "$out/execution-asan"
export ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
export UBSAN_OPTIONS=halt_on_error=1
"$out/execution-asan"
for binary in execution execution-asan; do
  for negative in corrupt-output require-strict; do
    set +e
    "$out/$binary" "--$negative" > "$out/$binary-$negative.stdout" \
      2> "$out/$binary-$negative.stderr"
    code=$?
    set -e
    if [[ $code != 1 ]] || ! grep -q 'FAIL: chosen legal realization' "$out/$binary-$negative.stderr"; then
      echo "Negative control $binary/$negative did not reject the wrong result/profile." >&2
      exit 1
    fi
  done
done
set +e
"$out/execution-asan" --asan-invalid-capacity \
  > "$out/asan-negative.stdout" 2> "$out/asan-negative.stderr"
code=$?
set -e
if [[ $code == 0 || $code == 99 ]] ||
   ! grep -q 'AddressSanitizer: heap-buffer-overflow' "$out/asan-negative.stderr" ||
   ! grep -q research_gemm "$out/asan-negative.stderr"; then
  echo 'Generated full-tile load instrumentation was not established.' >&2
  exit 1
fi

# Retain the losing masked-K4 pipeline: this can compile without delivering
# either a register-carried accumulator or the desired outer-product FMA form.
"$mlir_bin/mlir-opt" "$experiment_dir/tiled_contract.mlir" \
  --transform-interpreter --canonicalize --lower-vector-mask \
  --fold-memref-alias-ops --canonicalize \
  --transform-interpreter=entry-point=lower_contract --canonicalize \
  --lower-vector-mask --transform-interpreter=entry-point=hoist \
  --test-transform-dialect-erase-schedule -o "$out/masked-k4-negative.mlir"
if grep -Eq 'iter_args|vector.outerproduct|vector.fma' "$out/masked-k4-negative.mlir"; then
  echo 'Pinned masked-K4 negative observation changed; inspect before updating claims.' >&2
  exit 1
fi
sha256sum "$out/leaf.o" "$out/sanitized.o" "$out/scheduled.mlir" "$out/leaf.ll"
printf 'Research artifacts: %s\n' "$out"
