#!/usr/bin/env bash
# No dependencies installed, no LLVM build, no GPU spoofing, no benchmark.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: bash run.sh BUILD_DIRECTORY CANONICAL_ISSUER MLIR21_PREFIX HIP_PREFIX" >&2
  exit 2
fi
build_dir=$(realpath -m "$1")
issuer=$(realpath "$2")
mlir_prefix=$(realpath "$3")
hip_prefix=$(realpath "$4")
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$build_dir" in /tmp|/tmp/*|/dev/shm|/dev/shm/*)
  echo "build artifacts must use SSD-backed storage, not shared tmpfs" >&2; exit 2;;
esac
if [[ -n ${HSA_OVERRIDE_GFX_VERSION+x} ]]; then
  echo "HSA_OVERRIDE_GFX_VERSION is forbidden" >&2; exit 2
fi
mkdir -p "$build_dir/tmp"
export TMPDIR="$build_dir/tmp"
opt="$mlir_prefix/bin/mlir-opt"
translate="$mlir_prefix/bin/mlir-translate"
for tool in "$opt" "$translate" /usr/bin/clang++-21 /usr/bin/llc-21 /usr/bin/ld.lld-21; do
  "$tool" --version | head -1 | tee -a "$build_dir/toolchains.txt"
  "$tool" --version | head -1 | grep -q '21.1.8'
done
df -h "$build_dir"
/usr/bin/clang++-21 -std=c++17 -O0 "$source_dir/extract_device.cpp" \
  -I"$mlir_prefix/include" -I/usr/lib/llvm-21/include \
  -L"$mlir_prefix/lib" -Wl,-rpath,"$mlir_prefix/lib" \
  -L/usr/lib/llvm-21/lib -lMLIR -lLLVM-21 -o "$build_dir/extract-device"
"$issuer" --output "$build_dir/strict.ll"
"$opt" "$build_dir/strict.ll.structured.mlir" \
  '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' \
  --convert-linalg-to-parallel-loops --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu --canonicalize \
  --gpu-launch-sink-index-computations --gpu-kernel-outlining --canonicalize \
  -o "$build_dir/outlined.mlir"
"$opt" "$build_dir/outlined.mlir" \
  '--pass-pipeline=builtin.module(gpu.module(lower-affine,convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx1150 runtime=HIP},reconcile-unrealized-casts))' \
  -o "$build_dir/rocdl.mlir"
"$build_dir/extract-device" "$build_dir/rocdl.mlir" "$build_dir"
for stage in fill gemm; do
  suffix=''
  if [[ $stage == gemm ]]; then suffix='_0'; fi
  "$translate" --mlir-to-llvmir \
    "$build_dir/__matcore_strict_gemm_f32_v1_kernel${suffix}.device.mlir" \
    -o "$build_dir/$stage.ll"
  /usr/bin/opt-21 -passes=verify -disable-output "$build_dir/$stage.ll"
  if grep -Eq ' (fast|reassoc|contract|nnan|ninf|nsz|arcp|afn) |llvm.fma|llvm.fmuladd' "$build_dir/$stage.ll"; then
    echo 'unexpected floating-point permission in strict LLVM IR' >&2; exit 1
  fi
  /usr/bin/llc-21 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 \
    -fp-contract=off -denormal-fp-math=ieee -denormal-fp-math-f32=ieee \
    -filetype=obj "$build_dir/$stage.ll" -o "$build_dir/$stage.o"
  /usr/bin/ld.lld-21 -shared "$build_dir/$stage.o" -o "$build_dir/$stage.hsaco"
done
/usr/bin/llvm-objdump-21 --disassemble "$build_dir/gemm.hsaco" > "$build_dir/gemm.isa"
grep -q 'v_mul_f32' "$build_dir/gemm.isa"
grep -q 'v_add_f32' "$build_dir/gemm.isa"
if grep -Eq 'v_(dual_)?(fma|mac|mad)' "$build_dir/gemm.isa"; then
  echo 'unexpected fused arithmetic in strict machine code' >&2; exit 1
fi
/usr/bin/llvm-readelf-21 --notes "$build_dir/gemm.hsaco" > "$build_dir/gemm.metadata"
grep -q 'amdgcn-amd-amdhsa--gfx1150' "$build_dir/gemm.metadata"
/usr/bin/clang++-21 -std=c++20 -O1 -Wall -Wextra -Werror \
  -ffp-contract=off -fno-fast-math -D__HIP_PLATFORM_AMD__ \
  -I"$hip_prefix/include" "$source_dir/runner.cpp" \
  -L"$hip_prefix/lib" -Wl,-rpath,"$hip_prefix/lib" -lamdhip64 \
  -o "$build_dir/runner"
"$build_dir/runner" "$build_dir/fill.hsaco" "$build_dir/gemm.hsaco" \
  | tee "$build_dir/physical-result.txt"
# Falsifiers must execute and fail numerically, not merely fail to compile/load.
for mutation in ftz fma; do
  fp_contract=off; denormal=ieee; expected=subnormal-output
  if [[ $mutation == ftz ]]; then denormal=preserve-sign; fi
  if [[ $mutation == fma ]]; then fp_contract=fast; expected=fma-discriminator; fi
  /usr/bin/llc-21 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 \
    -fp-contract="$fp_contract" -denormal-fp-math=ieee \
    -denormal-fp-math-f32="$denormal" -filetype=obj "$build_dir/gemm.ll" \
    -o "$build_dir/gemm-$mutation.o"
  /usr/bin/ld.lld-21 -shared "$build_dir/gemm-$mutation.o" \
    -o "$build_dir/gemm-$mutation.hsaco"
  if "$build_dir/runner" "$build_dir/fill.hsaco" "$build_dir/gemm-$mutation.hsaco" \
      > "$build_dir/$mutation-negative.txt" 2>&1; then
    echo "negative control $mutation unexpectedly passed" >&2; exit 1
  fi
  grep -q "$expected" "$build_dir/$mutation-negative.txt"
  grep -q 'FAIL: strict numerical mismatch' "$build_dir/$mutation-negative.txt"
done
sha256sum "$issuer" "$source_dir/runner.cpp" "$source_dir/extract_device.cpp" \
  "$build_dir/strict.ll.structured.mlir" "$build_dir/outlined.mlir" \
  "$build_dir/rocdl.mlir" "$build_dir/fill.hsaco" "$build_dir/gemm.hsaco" \
  > "$build_dir/artifact-sha256.txt"
echo 'PASS: compile, link, actual gfx1150 execution, strict FP negative controls'
