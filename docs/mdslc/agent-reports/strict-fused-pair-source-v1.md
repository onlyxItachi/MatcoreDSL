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
- Later focused test/wiring/docs checkpoints: pending exact commit handoff.

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

Actual nm confirmed the exact appended private method export:
`_ZN7matcore5mdslc7runtime14closed_host_v112SessionAbiV219gemmStrictFusedPairEmmRKNS2_10ValueAbiV2ES6_S6_NS2_7NumericES7_RS4_`,
and the genuine issued fused C-interface definition. Clang21 strict fixture
`-std=c++20 -fsyntax-only -Wall -Wextra -Werror` passed separately.

## Focused registrations and pending execution

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

Connected source, coherent composed builds/tests, independent exact-commit review,
clean full suite, hosted lanes and canonical integration remain separate pending
gates. The forwarding qualification predecessor must complete before this branch
can merge. Issuer-only experiments and inspection do not substitute for these.
