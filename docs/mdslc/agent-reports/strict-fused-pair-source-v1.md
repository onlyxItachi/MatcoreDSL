# Strict fused pair: source/runtime implementation report

Worktree: `/home/hamza-usta/mdslc-work/region-optimization-v1/fused-source`.
Branch: `mdslc/fused-pair-source-execution-v1`, original base
`52f363ed985bb1eec0f8000dc93efd5b7666f6b3`. Root owns integration, full qualification,
CI/GitHub and canonical checkpoint. No merge/push is performed by this agent.

## Design and boundaries

Root and independent reviewer approved a sealed derived-plan extension plus an
additive private split-frontier runtime entry for exactly pure adjacent strict
lhs-chain GEMMs. Default none and independent forwarding selection remain.
Only explicit Linux x64 generated-strict is eligible; incompatible policies and
zero eligible pairs fail clearly. No whole-region witness transformation,
serialized/module authority, new source math, broad liveness optimizer, GPU/fusion
dispatch or performance claim is introduced. The issuer lane owns its MLIR
library/tool/recipe and tests separately.

An initial conservative proposal would have added a Sema query-use ledger to
reject even discarded C rows/cols queries. Root and reviewer rejected that
unnecessary frontend coupling: admission returns pure Shape bindings without a
runtime operation/frontier when dead. Recursive retained Program use counts
instead include actual read dimensions and both branch condition/body uses.
Dead canonical/helper Shape queries are explicitly tested as eligible; every
live retained intermediate use blocks the window.

The runtime retains original f1 guard precedence and FULL C signed/product/byte
extent, an independent f1 FP enter/check/restore, then f2's original guard order
before output/scratch allocations and its own numerical scope. C allocation is
removed, not required checks. Earlier effects/owning observations and source
failure identities remain. Result assignment is last; no Session/Value layout
fields were added. The appended actual enum and report do not claim materialized
C or two original physical calls. Empty E/N0 skips the leaf after guards, whereas
K0 with nonempty E/N>0 must evaluate 0*Inf/NaN. Trusted bounded private memset
has the same no-arbitrary-effect/FP-control obligation as the approved strict
leaf; callbacks/providers/GPU/interposed libc are not covered.

Details: [STRICT_FUSED_PAIR_SOURCE_V1.md](../STRICT_FUSED_PAIR_SOURCE_V1.md).

## Checkpoints

- Untested initial production checkpoint: `5f34a25` (explicitly WIP, not qualified).
- Issuer dependency: `12c69db`, cherry-pick of
  `8ce0bcd349cf2339162ce423c82377d3c827d642`.
- Independent contract-first runtime tests: `faad4ca`, cherry-pick of
  `d7b0571744daf3ba2c2813708f25e6a0aa1803d9`. The only conflict was additive report
  context; both complete sections were preserved. Authorship was frozen before
  the reviewer read this runtime.
- Focused test/wiring: `8fccd8a`; contract/AGENTS: `c80b319`.
- Root forwarding harness corrections were preserved by focused cherry-picks
  `41d24e7` (origin `fdcde9e`) and `de5d09f` (origin `9a613c5`). No reset/rebase
  was used. The complete fresh focused source head is
  `de5d09f4c59ec035482121086f0422b2476a76ff`.

## Executed focused evidence

Exactly one compiler job with shared heavy-link lock and SSD TMPDIR; production
runtime and independently authored test were frozen during the build/run:

```
env TMPDIR=/home/hamza-usta/mdslc-work/region-optimization-v1/tmp \
flock /home/hamza-usta/mdslc-work/region-optimization-v1/builds/heavy-link.lock \
/usr/bin/clang++-21 -std=c++20 -O1 -g -fsanitize=address,undefined \
-fno-omit-frame-pointer -frounding-math -ftrapping-math -ffp-contract=off \
-DMDSLC_CLOSED_HOST_TESTING=1 -DMDSLC_CLOSED_HOST_GENERATED_STRICT=1 \
-DMDSLC_HAS_FUSED_PAIR_CPU_V1=1 -Wall -Wextra -Werror \
-Icompiler/lib/runtime -Icompiler/include \
compiler/lib/runtime/closed_host_v1.cpp compiler/lib/platform/closed_fp_environment_v1.cpp \
compiler/tests/closed_host/strict_fused_pair_independent_test.cpp \
/home/hamza-usta/mdslc-work/region-optimization-v1/builds/baseline/lib/regions/strict-asan.o \
/home/hamza-usta/mdslc-work/region-optimization-v1/builds/fused-issuer/lib/regions/strict-fused-pair-asan.o \
-lm -o /home/hamza-usta/mdslc-work/region-optimization-v1/builds/fused-runtime.6WJlHf/strict-fused-pair-independent-asan-ubsan
```

`timeout 30s` execution passed `Independent strict fused-pair checks: 204;
failures: 0`, no sanitizer diagnostics. Both generated arithmetic objects were
genuinely ASan-instrumented, not only the host wrapper. This does not claim a
full production registry build or authenticated source route.

The initial actual tool output captured both streams and exit 0. A later saved
repeat used `timeout 30s ... 2>&1 | tee runtime-run-combined.log` with shell
`pipefail`, also exit 0 and 204/0, no diagnostics. The earlier
`runtime-run.log` saves stdout only and must not alone be described as captured
sanitizer stderr. Both logs are under `builds/fused-runtime.6WJlHf`; combined-log
SHA-256 is `b3dbfd4215733646f28f02fba3e0aad666be1ea66cb9bcecb121030032fc6c16`.

Actual nm confirmed the exact appended private method export:
`_ZN7matcore5mdslc7runtime14closed_host_v112SessionAbiV219gemmStrictFusedPairEmmRKNS2_10ValueAbiV2ES6_S6_NS2_7NumericES7_RS4_`,
and the genuine issued fused C-interface definition. Clang21 strict fixture
`-std=c++20 -fsyntax-only -Wall -Wextra -Werror` passed separately.

## Fresh composed focused qualification

A new `builds/fused-source` directory was configured from the clean exact source
head above using Release, Clang21, native/bootstrap, coherent MLIR21.1.8,
experimental regions and the existing BLAS/NVVM/ROCDL candidate set. These extra
candidates do not dispatch the pair. All source/production/test inputs remained
frozen during configure, build and execution. Exact configure:

```
env TMPDIR=/home/hamza-usta/mdslc-work/region-optimization-v1/tmp \
cmake -S compiler -B /home/hamza-usta/mdslc-work/region-optimization-v1/builds/fused-source \
-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
-DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 -DMDSLC_CLANGXX_EXECUTABLE=/usr/bin/clang++-21 \
-DMDSLC_ENABLE_NATIVE_FRONTEND=ON -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON \
-DMDSLC_ENABLE_MATCORE_MLIR=ON -DMDSLC_ENABLE_EXPERIMENTAL_REGIONS=ON \
-DMDSLC_ENABLE_EXPERIMENTAL_GPU_ISSUER=ON -DMDSLC_ENABLE_EXPERIMENTAL_NVVM=ON \
-DMDSLC_ENABLE_EXPERIMENTAL_ROCDL=ON -DMDSLC_ENABLE_OPENBLAS=ON \
-DMDSLC_REQUIRE_OPENBLAS=OFF -DMDSLC_EXPERIMENTAL_CPU_GEMM_SCHEDULE=scalar \
-DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm -DClang_DIR=/usr/lib/cmake/clang-21 \
-DMLIR_DIR=/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir \
'-DCMAKE_CXX_LINKER_LAUNCHER=flock;/home/hamza-usta/mdslc-work/region-optimization-v1/builds/heavy-link.lock'
```

One-job build of `matcore_closed_host_strict_fused_pair_plan_tests`,
`matcore_closed_host_strict_fused_pair_independent_tests`,
`matcore_closed_host_production_tests` and `mdslc-region` passed 140/140 steps.
This is a focused build, not the full standalone target graph. Generated MLIR
and existing CUDA-header warnings were inherited; no build error occurred.
The actual fused object gate ran as part of production DSO assembly.

The exact focused run, with the shared lock also covering child compiler/link
commands, was:

```
env TMPDIR=/home/hamza-usta/mdslc-work/region-optimization-v1/tmp \
flock /home/hamza-usta/mdslc-work/region-optimization-v1/builds/heavy-link.lock \
ctest --test-dir /home/hamza-usta/mdslc-work/region-optimization-v1/builds/fused-source \
--parallel 1 -V \
-R '^(frontend.closed_host_strict_fused_pair_plan_v1|runtime.closed_host.strict_fused_pair_independent_v1|runtime.closed_host.production_v1|driver.strict_fused_pair.generated-strict|driver.strict_fused_pair.program)$'
```

Combined stdout/stderr was saved through `2>&1 | tee composed-focused.log` with
`pipefail`. Exit 0, **5/5 passed**, 79.89 seconds, no skipped tests or failures:

- Independent runtime: 204 checks, zero failures, actual matched production
  registry with genuine issued normal single/pair objects.
- Native-only production control: existing scalar execution plus macro-off
  pair unavailable/sticky/no-output-mutation assertions.
- Derived plan: 283 checks, zero failures, including helper ledgers, dead/live
  dimensions, unequal branch holes and independent windows.
- Single-source: both none and strict-fused-pair each passed 659 checks,
  22 executed cases, zero failures. Actual executable undefined-reference
  discriminator and all incompatible/default/zero-pair/duplicate negatives passed.
- Multi-source: both modes likewise passed 659 checks and 22 cases; the genuine
  program compiler path, call-reference discriminator and negative gates passed.

Logs are under `builds/fused-runtime.6WJlHf`:

- `composed-build.log`, SHA-256
  `0efd62fb6d93392a69a3db0464678972f61485d53cd8873e63e78777c350139a`.
- `composed-focused.log`, SHA-256
  `6793d583c9baae6540c17e9b05d49217f196651ecf64f5ead51bd006975e94d2`.

The independent reviewer accepted exact `de5d09f` for composed qualification,
no blocking code/authority findings, in report commit `499862e`. It inspected
fresh configure/artifact binding and saved execution evidence but did not rerun
tests. Review, physical focused execution, package and full/hosted qualification
remain distinct evidence categories.

## Focused registrations and remaining gates

- `frontend.closed_host_strict_fused_pair_plan_v1`: canonical legality/proposal
  falsifiers, whole Program dimensions/control arms, dead pure queries,
  pure helper-expanded pairs, unequal branch holes and independent windows.
- `runtime.closed_host.strict_fused_pair_independent_v1`: genuine matched
  single/pair testing registry, finite timeout. Existing native-only test
  runtimes remain unchanged; their production control additionally tests
  macro-off pair unavailability and sticky no-mutation behavior.
- `driver.strict_fused_pair.generated-strict` and `.program`: same source none
  versus explicit pair, actual compiled runtime-reference discriminator,
  22 math/status/effect/FP/source cases, forced-policy/zero-pair refusal.
- Existing feature-ON install and source/build-inaccessible package lanes gain
  the same actual pair check on x64; counts are unchanged. Inaccessible tests
  copy their genuine fixture before producer removal and never use deleted
  source paths. No nested package tests have been run by this agent.

Connected source and focused coherent composition/review now pass as recorded.
Installed/source-inaccessible, clean full suite, hosted lanes and canonical
integration remain separate pending gates. The forwarding predecessor merged
normally in PR #78 at `355e281fe0de4e54c03243d332eafbbe2095591b`; this does not
merge or qualify the new pair. Issuer-only experiments and inspection do not
substitute for the remaining composed gates. No further code edits/builds were
started after the focused run; root owns the next integration/full qualification.
