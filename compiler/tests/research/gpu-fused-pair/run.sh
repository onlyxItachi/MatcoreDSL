#!/usr/bin/env bash
# Research only; deliberately no production CMake/default-dispatch hook.
# build: one compiler process at a time. execute-* requires device reservation.
set -euo pipefail
gpu_research_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
gpu_research_repo=$(CDPATH= cd -- "$gpu_research_dir/../../../.." && pwd)
gpu_research_mode=${1:?build|execute-nvvm|execute-rocdl|memcheck-nvvm|negative-memcheck-nvvm}
gpu_research_build=${2:?fresh explicit output directory}
gpu_research_reuse=${3:-/home/hamza-usta/mdslc-work/region-optimization-v1/builds/source-candidate}
gpu_research_prefix=/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21
gpu_research_lock=/home/hamza-usta/mdslc-work/region-optimization-v1/builds/heavy-link.lock
export TMPDIR=/var/tmp/mdslc-region-correctness.bvy8fe79
test -d "$TMPDIR"
mkdir -p -- "$gpu_research_build"
case "$gpu_research_mode" in
  build)
    /usr/bin/clang++-21 --version | rg -q 'version 21\.1\.8'
    "$gpu_research_prefix/bin/mlir-opt" --version | rg -q 'version 21\.1\.8'
    # Link already qualified internal stage libraries; do not rebuild MDSLC.
    flock "$gpu_research_lock" /usr/bin/clang++-21 -std=c++20 -O0 \
      "$gpu_research_dir/issue.cpp" -I"$gpu_research_repo/compiler/lib/mlir" \
      -I"$gpu_research_repo/compiler/include" -I"$gpu_research_prefix/include" \
      -I/usr/lib/llvm-21/include \
      "$gpu_research_reuse/lib/libmatcore_cpu_fused_gemm_candidate.a" \
      "$gpu_research_reuse/lib/libmatcore_closed_region_semantics.a" \
      "$gpu_research_reuse/lib/libmatcore_mlir_bufferized_handoff.a" \
      "$gpu_research_reuse/lib/libmatcore_mlir_structured_handoff.a" \
      "$gpu_research_reuse/lib/libmatcore_mlir_semantics.a" \
      "$gpu_research_reuse/lib/libmatcore_ir_v1.a" \
      "$gpu_research_reuse/lib/libmatcore_ir_v0.a" \
      "$gpu_research_prefix/lib/libMLIR.so.21.1" \
      -L/usr/lib/llvm-21/lib -lLLVM-21 -Wl,-rpath,"$gpu_research_prefix/lib" \
      -o "$gpu_research_build/issue"
    "$gpu_research_build/issue" "$gpu_research_build" | tee "$gpu_research_build/issue.log"
    /usr/lib/llvm-21/bin/llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_89 \
      -mattr=+ptx80 -fp-contract=off "$gpu_research_build/nvvm.ll" \
      -o "$gpu_research_build/nvvm.ptx"
    /usr/local/cuda/bin/ptxas --gpu-name sm_89 --fmad=false -v \
      "$gpu_research_build/nvvm.ptx" -o "$gpu_research_build/nvvm.cubin" \
      2> "$gpu_research_build/ptxas.log"
    /usr/lib/llvm-21/bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 \
      -fp-contract=off -denormal-fp-math=ieee -denormal-fp-math-f32=ieee \
      -filetype=obj "$gpu_research_build/rocdl.ll" -o "$gpu_research_build/rocdl.o"
    flock "$gpu_research_lock" /usr/lib/llvm-21/bin/ld.lld -shared \
      "$gpu_research_build/rocdl.o" -o "$gpu_research_build/rocdl.hsaco"
    python3 "$gpu_research_dir/object_contract.py" --build "$gpu_research_build"
    flock "$gpu_research_lock" /usr/bin/clang++-21 -std=c++20 -O2 \
      -ffp-contract=off -fno-fast-math -pthread -DRESEARCH_CUDA \
      -I/usr/local/cuda/include "$gpu_research_dir/execute.cpp" \
      /usr/lib/x86_64-linux-gnu/libcuda.so -o "$gpu_research_build/execute-nvvm"
    flock "$gpu_research_lock" /usr/bin/clang++-21 -std=c++20 -O2 \
      -ffp-contract=off -fno-fast-math -pthread -D__HIP_PLATFORM_AMD__ \
      -I/opt/rocm/include "$gpu_research_dir/execute.cpp" \
      /opt/rocm/lib/libamdhip64.so.7.2.70201 -Wl,-rpath,/opt/rocm/lib \
      -o "$gpu_research_build/execute-rocdl"
    ;;
  execute-nvvm)
    timeout 90s "$gpu_research_build/execute-nvvm" "$gpu_research_build/nvvm.cubin" \
      2>&1 | tee "$gpu_research_build/physical-nvvm.log"
    ;;
  memcheck-nvvm)
    timeout 120s /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
      --error-exitcode 86 --padding 32 "$gpu_research_build/execute-nvvm" \
      "$gpu_research_build/nvvm.cubin" 2>&1 | tee "$gpu_research_build/memcheck-normal.log"
    ;;
  negative-memcheck-nvvm)
    # Deliberately violated workspace capacity only here, under instrumentation,
    # in a separate bounded process/context. Never execute malformed AMD input.
    set +e
    timeout 30s env MDSLC_RESEARCH_CUDA_MEMCHECK_NEGATIVE_V1=1 \
      /usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 86 \
      --padding 32 "$gpu_research_build/execute-nvvm" "$gpu_research_build/nvvm.cubin" \
      --negative-workspace-under-memcheck 2>&1 | tee "$gpu_research_build/memcheck-negative.log"
    gpu_research_status=${PIPESTATUS[0]}
    set -e
    test "$gpu_research_status" -ne 0
    test "$gpu_research_status" -ne 124
    rg -q 'Invalid __global__ (write|read)' "$gpu_research_build/memcheck-negative.log"
    rg -q '__matcore_research_strict_fused_pair_kernel' "$gpu_research_build/memcheck-negative.log"
    printf 'PASS CUDA undercapacity negative detected; subprocess-exit=%s\n' "$gpu_research_status" \
      | tee "$gpu_research_build/memcheck-negative-status.txt"
    ;;
  execute-rocdl)
    timeout 90s "$gpu_research_build/execute-rocdl" "$gpu_research_build/rocdl.hsaco" \
      2>&1 | tee "$gpu_research_build/physical-rocdl.log"
    ;;
  *) printf 'unknown mode\n' >&2; exit 1 ;;
esac
