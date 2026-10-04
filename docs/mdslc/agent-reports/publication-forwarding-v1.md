# Publication-to-read forwarding implementation report

Implementation owner: isolated worktree
`/home/hamza-usta/mdslc-work/region-optimization-v1/forwarding`, branch
`mdslc/publication-forwarding-v1`, starting exact canonical
`43ce3cb2557aad4dc3b810ea715c00531c76628f`. The original worktree and other agent
lanes are preserved. Integration/full builds/hosted qualification/merging belong
to the campaign root. This agent has not merged, pushed or changed toolchains.

Implementation: `14367dc825333e8465299149a7250dc8ac889688`.
Focused plan/source/driver/package tests: `6548a2758442757be5b9f74b2b1b237fb9ea9581`.

## Implemented boundary

- Immutable compiler-issued `ClosedHostDerivedPlan`, original evidence ownership,
  exact source/witness reauthentication and canonical bounded-list verification.
  A diagnostic proposal is never stored or turned into an executable seal.
- Same sealed resource/current successful publication version forwarding; every
  publication invalidates previous all-MAY-alias bindings. Conservative branch
  entry/join barriers, with legal within-arm forwarding only.
- Original Program/paired witness, required ordered checks, original source file
  sites/helper bindings, frontiers, candidate selection and owning observations
  unchanged. Plan/mode are bound into private helper/orchestration identity.
- Nonallocating `readForwarded` retains an immutable Value only after all original
  requested extent/shape and descriptor checks in their original order. Sticky
  failure/reentry/completion/FP behavior and publication guarantee are retained.
  No Session/Value layout change or runtime alias/proof interpreter.
- Explicit `--optimization none|publication-read-forwarding` in single- and
  multi-source driver routes, default `none`, orthogonal to forced candidate.
  Additive private DSO export and existing owned-header snapshot integration.

The [contract](../PUBLICATION_READ_FORWARDING_V1.md) gives the exact limitations.
This is not a fusion, target-selection, residency or performance milestone.

## Independent authorship and executed validation

The independent reviewer authored
`compiler/tests/closed_host/forwarded_read_independent_test.cpp` before inspecting
implementation. Authorship commit `f381ef60b8e498d92929726b69870aed5ae9a719` was
cherry-picked here as `cc078cf769ded78d739ed1797e1dc119fdb4c9c6` and registered as
`runtime.closed_host.forwarded_read_independent_v1` against the separate testing
runtime. Its [report](publication-forwarding-independent-v1.md) records its scope.

The integration owner explicitly authorized one compile job after the fusion
experiment released its compiler slot. Actual production runtime sources plus
the independent test were compiled and executed with the coherent installed
Clang 21 tuple, SSD temporary directory and shared heavy-link lock:

```sh
env TMPDIR=/home/hamza-usta/mdslc-work/region-optimization-v1/tmp \
  flock /home/hamza-usta/mdslc-work/region-optimization-v1/builds/heavy-link.lock \
  /usr/bin/clang++-21 -std=c++20 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer -frounding-math -ftrapping-math -ffp-contract=off \
  -DMDSLC_CLOSED_HOST_TESTING=1 -Wall -Wextra -Werror \
  -Icompiler/lib/runtime -Icompiler/include \
  compiler/lib/runtime/closed_host_v1.cpp \
  compiler/lib/platform/closed_fp_environment_v1.cpp \
  compiler/tests/closed_host/forwarded_read_independent_test.cpp -lm \
  -o /home/hamza-usta/mdslc-work/region-optimization-v1/builds/forwarding-runtime.d5jbIn/forwarded-read-independent-asan-ubsan
```

Result: **`Independent forwarded-read checks: 67; failures: 0`**, exit 0, no
ASan/UBSan diagnostic. Production bytes were frozen during compilation/execution.
`nm --defined-only ... | rg readForwarded` confirmed the actual definition equals
the export map's
`_ZN7matcore5mdslc7runtime14closed_host_v112SessionAbiV213readForwardedEmNS2_12ResourceViewEmmRKNS2_10ValueAbiV2ERS5_`.

The new driver fixture also passed `/usr/bin/clang++-21 -std=c++20 -fsyntax-only
-Wall -Wextra -Werror -Icompiler/include -x c++
compiler/tests/closed_driver/publication_read_forwarding.mdsl`. `git diff --check`
passed. Neither static syntax nor the direct runtime binary establishes real
source/driver execution, candidate DSO linkage, installed artifacts or full-suite
qualification.

## Authored integration gates, not yet executed by this agent

- `frontend.closed_host_derived_plan_v1`: exact list/count/order/all fields,
  forged proposals, invalid optimization, None/nonempty proposal refusal,
  source/selected-function mismatch, host-less issuance, moved-from rejection,
  all-MAY-alias writes, unequal nested branches, original witness/site/frontiers.
- `driver.publication_read_forwarding.<policy>`: same physical source both opt
  modes, independent exact math/FP/failure-prefix/owning-observation oracles;
  actual optimized-only undefined `readForwarded` reference, unknown/duplicate
  option refusals. Mandatory native-strict/generated-strict/automatic;
  conditional existing ISA/reassociate/native/provider/GPU candidates.
- `driver.publication_read_forwarding.program`: actual 2-TU opt propagation,
  checked source outcomes and optimized-only call reference.
- Existing feature-ON package test additionally runs installed driver same-source
  forwarding controls. Existing experimental compile test checks different
  optimization-owned private helper identities for the same callable source.
- Existing source/build-inaccessible package test copies the genuine forwarding
  fixture out before deleting its disposable producer source/build, then uses
  only the relocated installed driver and copied consumer to compile both modes,
  requires actual execution and checks optimized-only `readForwarded` references.
  No runner is found through a deleted producer path; the installed driver digest
  and continued producer-tree absence are rechecked afterward.

For permitted candidates, the runner materializes an owned fixture with explicit
existing `reassociate_f32` permissions before compiling both opt modes. It never
uses candidate selection to weaken strict source. A unavailable forced candidate
must refuse at the first GEMM with no publication; it is reported as explicit
refusal, not physical execution. `REQUIRE_EXECUTION=ON` demands real execution.

Pre-build review caught and corrected the host-less admission control accidentally
containing preprocessing and the hand-derived symbol substitution for the new
export. They were corrected before runtime validation; no false passing result
was reported. Remaining complete/affected builds, source admission/execution,
sanitizer negative controls, physical ISA/GPU qualification, review, hosted CI and
canonical merge are pending the integration owner's exact-head evidence.
