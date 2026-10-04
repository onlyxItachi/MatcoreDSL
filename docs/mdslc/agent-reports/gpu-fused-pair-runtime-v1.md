# Combined GPU runtime engineering evidence v1

Status: focused WIP evidence, not complete source/physical/package qualification.
Base CPU source checkpoint: `6e25df7e85ae3419d9e688541065433ff3151151`.
Independent contract-first test: original `c191bff9f82f19a4514a3f2a8017a59ec991d134`,
integrated unchanged as `24010a0`. Runtime/API harness implementation: `d2dd74e`.

## Observed focused results

Exact `/usr/bin/clang++-21` 21.1.8 compiled the real Session/ScopedFp test and
both CUDA/HIP API mocks with ASan+UBSan. All were fresh executables under
`/home/hamza-usta/mdslc-work/region-optimization-v1/builds/gpu-pair-mocks.B6An97`.
No real GPU/runtime library was linked into these mock executables.

- Contract-first real Session / mocked GPU: **449 checks, zero failures**.
- Combined CUDA API: modes 0..30, pending, pending-download: **33 processes**.
- Combined HIP API: 11 named controls and faults 1..24: **35 processes**.
- Original CUDA API: modes 0..31, pending, pending-download: **34 processes**.
- Original HIP API: 11 named controls and faults 1..23: **34 processes**.

All 136 API processes passed; no sanitizer diagnostic. These counts are not
CTest counts or physical device cases. Original good-path API call counts remain
31 CUDA / 23 HIP, confirming the added null D/panel fields are inert there.
Separate target inspection, physical execution and source/package tests remain
required; mocks do not prove generated numerical behavior.

A subsequent independent review found that the direct private adapter could
reach the issued leaf for N=0/nonempty E even though the leaf requires N>0.
Source Session already bypassed that geometry correctly. The direct adapter now
fails closed before allocation/launch and the contract says Session owns the
positive-zero bypass. Extended ASan+UBSan binaries additionally test uncertain D
upload, launch and download plus shared sibling poisoning and direct N=0 refusal:
**36 CUDA + 39 HIP processes passed**, no diagnostics. These are a new 75-process
run, not 75 additional distinct baseline cases. The saved extended log is
`gpu-pair-runtime-final-api-mocks.log`; executable suffixes are `-pair-final`.

Saved raw logs under the evidence directory beside that build root:
`gpu-pair-runtime-independent-frontiers.log`,
`gpu-pair-runtime-initial-api-mocks.log`, and
`gpu-pair-runtime-original-api-regression.log`.

## Reproduce the focused build

Commands are from the runtime worktree, with each compilation run serially and
one explicitly reserved compiler slot. Set OUT to a new external build directory;
do not reuse a frozen evidence executable as a mutable build destination.

```sh
OUT=/absolute/task-owned/new-build-directory
CXX=/usr/bin/clang++-21
FLAGS='-std=c++20 -O1 -g -fno-fast-math -ffp-contract=off -frounding-math -fsanitize=address,undefined -fno-omit-frame-pointer -pthread'
INCLUDES='-Icompiler/include -Icompiler/lib/runtime'
export TMPDIR=/var/tmp/mdslc-region-correctness.bvy8fe79
$CXX $FLAGS $INCLUDES -DMDSLC_CLOSED_HOST_TESTING \
  -DMDSLC_CLOSED_HOST_GENERATED_NVVM -DMDSLC_CLOSED_HOST_GENERATED_NVVM_FUSED_PAIR \
  -DMDSLC_CLOSED_HOST_GENERATED_ROCDL -DMDSLC_CLOSED_HOST_GENERATED_ROCDL_FUSED_PAIR \
  compiler/tests/generated_gpu_fused_pair/frontier_test.cpp \
  compiler/lib/runtime/closed_host_v1.cpp compiler/lib/platform/closed_fp_environment_v1.cpp \
  -o "$OUT/frontiers"
$CXX $FLAGS $INCLUDES -I/usr/local/cuda/include \
  -DMDSLC_TEST_GPU_FUSED_PAIR -DMDSLC_CLOSED_HOST_GENERATED_NVVM_FUSED_PAIR \
  compiler/tests/generated_gpu/cuda_adapter_test.cpp compiler/lib/runtime/closed_cuda_candidate_v1.cpp \
  -o "$OUT/cuda-pair"
$CXX $FLAGS $INCLUDES -I/opt/rocm/include -D__HIP_PLATFORM_AMD__ \
  -DMDSLC_TEST_GPU_FUSED_PAIR -DMDSLC_CLOSED_HOST_GENERATED_ROCDL_FUSED_PAIR \
  compiler/tests/closed_candidates/rocdl_adapter_test.cpp compiler/lib/runtime/closed_rocdl_candidate_v1.cpp \
  -o "$OUT/rocdl-pair"
```

The original-route mock builds use the same command lines without either
`MDSLC_TEST_GPU_FUSED_PAIR` or the corresponding `*_FUSED_PAIR` definition,
producing `cuda-original` and `rocdl-original` respectively.

```sh
set -e
"$OUT/frontiers"
for mode in {0..30} pending pending-download; do "$OUT/cuda-pair" "$mode"; done
for mode in good unavailable shape oversize output-alias null environment \
    unknown-completion output-limit zero-K-limit work-limit fault-{1..24}; do
  "$OUT/rocdl-pair" "$mode"
done
for mode in {0..31} pending pending-download; do "$OUT/cuda-original" "$mode"; done
for mode in good unavailable shape oversize output-alias null environment \
    unknown-completion output-limit zero-K-limit work-limit fault-{1..23}; do
  "$OUT/rocdl-original" "$mode"
done
```

## Review and remaining gates

The adversary authored and committed the 467-line frontier test before reading
the WIP runtime implementation, then inspected the implementation and saved
results. No new guard-order/ownership blocker was reported in the focused diff.
This is not final composed approval. The concrete [combined realization
law](accelerator-fusion-authority-review-v1.md) must also survive exact issued
artifact gates, real adapters, single/multi-source execution, unavailable
devices, installed artifacts, complete regressions and hosted CI.
