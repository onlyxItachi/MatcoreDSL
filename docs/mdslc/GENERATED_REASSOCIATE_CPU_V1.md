# Source-connected generated reassociate GEMM

## Bounded contract

The forced `mdslc-region --candidate generated-reassociate` policy connects an
authenticated source GEMM with explicit `Numerics::reassociate_f32` to a separate
compiler-issued MLIR/LLVM implementation. Strict arithmetic, automatic selection,
the existing-native reference route and canonical OpenBLAS policy are unchanged.

The private realization uses the actual coherent MLIR 21.1.8 Transform, Linalg,
Vector and serial transfer-hoisting machinery: full 4x8 tiles carry accumulators
across K; tails remain increasing-K scalar arithmetic. Those constants describe
one target candidate, not mathematical semantics or a public scheduling API.
No new lower-than-MLIR abstraction or private register allocator was needed.

The separate builtin binds true per-operation reassociation/FMA permission,
positive-zero initialization, gradual underflow, signed-zero meaning and no
finite-only assumptions. It does not permit cross-GEMM reassociation or blanket
fast math. Its source-file table uses the chosen contract's identity, SHA256 and
byte size, not the strict builtin's provenance.

Matcore checks a narrow structural envelope and exact replay of its trusted,
pinned upstream pipeline. Self-consistency of arbitrary renamed IR is not source
authentication; the issuer has no imported-IR execution entry point. Only its
two issued functions survive into the private leaf. Input/input aliasing is legal;
only the fresh output data carries LLVM noalias, not descriptor pointers.

Runtime legality precedes allocation and mutation: numerical compatibility,
compiled candidate availability and hardware/OS AVX2/FMA state. Direct discovery
does not borrow a different implementation's runtime-validation record. The leaf
is compiled with baseline x86-64 plus AVX2/FMA, not the wider x86-64-v3 bundle.
Empty outputs and zero-K still pass candidate legality, then complete locally.
The adapter preserves errno/FP state and the original ordered publication,
owning observation and sticky failure prefix. The canonical Runtime remains the
only Matcore-owned provider policy instance.

## Evidence and integration provenance

Canonical base: `917b5ecf1d325e525bc24e3c94764d2958980a93` (PR #63).
Independent lanes, retained as focused commits:

- Issuer `1c03306ee771f785378112866111288d8d7c5f1e`, classifier correction
  `62379dd6323b2aa8d93d840b1f426b92a1c5d176`:
  [issuer evidence](agent-reports/generated-reassociate-issuer-v1.md).
- Runtime/source adapter `b230b070e249a69ab327139ae2d9e7e2925c6363`:
  [adapter evidence](agent-reports/generated-reassociate-adapter-v1.md).
- Independent review `39051ba492fc223975e8f11e7a6993974d7cbf68`:
  [mutation and execution evidence](agent-reports/generated-reassociate-issuer-independent-v1.md).

Root integration through `55dd87b` preserves the actual chosen builtin source
table and explicit file-id 1, unions both independently added CTest inventories,
and adds six source-identity composition checks. The initial root test addition
failed compilation because OwningOpRef needs dereferencing before operation
attributes; that test-only error was corrected without changing production.

**OBSERVED:** fresh integrated Release / OpenBLAS ON-required build succeeded;
focused **22/22** tests passed in **26.55 s**, including six actual source policies,
both real generated leaf instrumentation modes, strict counterexamples, genuine READ4/READ32
sanitizer controls, capability guards and independent mutation/classifier tests.
The new scalar/vector fault classifier previously accepted an unrelated primary
fault with a later generated allocation frame; the reproduced false positive is
retained and the correction requires the primary error/access/first fault frame.

Frozen integrated Release identities:

- driver: `6423db0db4ea0db91d4c4f747f4160de5508c432b496be08267745d690bf96b5`;
- private candidate DSO: `262cfc9c40bf23d2bfa326fb1a17d78afd91243cea03e3e1c0fbbf98060c9e4c`;
- unchanged canonical runtime DSO: `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014`.

At clean implementation checkpoint
`73cdeb927c160c9eda691aeb7f666b9f22d3a6b3`, full Release/OpenBLAS ON regressions
passed **172/172** in **428.04 s**, including installed-source-inaccessible
execution. The exact OpenBLAS-OFF compiler ASan+UBSan affected selection passed
**117/117** in **252.45 s**; the smaller sanitizer gate passed **21/21** in
**31.33 s**. No test in those successful selections was skipped or not run.

The first full Release run was **171/172**: the installed-source-inaccessible
safety check correctly refused a stale configure-time dirty-source snapshot.
A clean explicit CMake reconfiguration, rebuild and complete rerun produced the
172/172 result above. No assertion or production code was weakened.

Frozen log identities (under the integration worktree's build directories):

- Release `full-clean-ctest.log`:
  `10dc0a34d3e93b3572a4aa8117a18e45ca6edced33532f6f2205b72273050676`;
- sanitizer `focused-ctest.log`:
  `968fdf7f98685e07e7b17e2f5cc10fdbf0d7772244adafc7e71b9f9f1f21af42`;
- preserved first Release `full-ctest.log`:
  `329d74518c74a0d18cf2042f6b37f62770bf898b833fcfe5be0d92394fe20fab`.

Independent connected-source and fresh-installed-SDK testing subsequently
passed **30,147 checks / 13 positive executions + 5 negative controls** in
Release/provider ON, and **27,157 / 11 + 5** in ASan+UBSan/provider OFF. Twenty
synthetic runner controls distinguish exact intended mathematical failures,
real hardware skips, changed artifacts and unrelated sanitizer/crash failures.
See [the independent connected report](agent-reports/generated-reassociate-connected-independent-v1.md).
This installed matrix keeps source/build trees visible; it is distinct from the
full suite's installed-source-inaccessible gate.

The permanent matrix and its runner add two tests: **174** total Release ON,
**119** in the affected sanitizer OFF selection. These test-only additions do
not change the already-tested production compiler, runtime or generated leaf.
At clean `02a98d6081bd2dbfbbc006157eb617947f653233`, both new CTest gates passed:
Release **2/2 / 51.94 s**, sanitizer **2/2 / 60.38 s**, with no skips. Their
actual configured inventories are 174 Release ON and 168 sanitizer OFF total;
the selected sanitizer union contains 119. Independent static review accepted
the exact CMake/wrapper wiring and confirmed unchanged production artifacts.
Logs `connected-final-ctest.log` have SHA256
`ea528a16b835b4c775d341c6455a1465973009069c28acacb18ef7978573c572`
(Release) and
`43e8807ba5e091d88633831878ee032abae70b2a342d055b872200c332dff0c0`
(sanitizer). Exact final-head hosted CI remains a separate gate.

## Claim boundary

This is one synchronous, private-output rank-2 f32 CPU candidate on Linux x86-64
and coherent 21.1.8. A passing source call is real generated execution only when
the local machine admits that candidate; a capability refusal is not execution.
Compiler-ASan instrumentation and actual generated-leaf sanitizer faults are
separate evidence. No whole-region transformation, fusion, automatic buffer
reuse, new device support, numerical-policy expansion or default schedule change.

Research x86-64-v3 timing does not authenticate this narrower compiled artifact
or its source overhead. A new [same-source measurement](agent-reports/generated-reassociate-source-evidence-v1.md)
does bind this actual narrower candidate: one warm single-thread Ryzen AI 9 HX
370 study, six shapes, 540 samples. The new candidate used 27.28–57.62% less time
than the previous generated candidate under identical reassociated source
permissions; OpenBLAS won five of six. The intervention combines schedule,
target features and optimization level, not a causal FMA-only improvement.
[Independent audit](agent-reports/generated-reassociate-source-measurement-independent-v1.md)
recomputed every sample and median and rechecked all frozen paths. No threshold,
universal performance or BLAS-parity claim follows; Issue #15 remains partial.

## Canonical merge gate

Normally merged as `c3b7b0b9a05e11d1b8d8cd3eff1c7ef3551a8280`,
[PR #66](https://github.com/onlyxItachi/MatcoreDSL/pull/66), after all **21/21**
exact-head hosted checks succeeded at `76bcf62183712544b202ce58290f04fef7502e48`.
Parents are previous canonical `b7e6cc03bd492ba9d0b2983dd1ce54611ef971fd` and that
reviewed head. Verified merge tree: `7048558b485d2e06eab87923d219502f47f75dca`.
Production, AGENTS and workflow files exactly match the reviewed head; the only
additional content is the pre-existing PR #65 documentation checkpoint.
[Independent review](https://github.com/onlyxItachi/MatcoreDSL/pull/66#issuecomment-5594984560)
and [final acceptance](https://github.com/onlyxItachi/MatcoreDSL/pull/66#issuecomment-5595189963)
record the gates; agent reviews are not separate human GitHub approvals.

The [PR native matrix](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34304414307)
recorded zero failures: Release MLIR/provider ON 172 registered; OFF 170;
OFF/row-contiguous 171; Debug 171; affected ASan+UBSan 119; focused runtime TSan
4. The three Release MLIR selections and affected ASan selection each skipped
one hardware-dependent packed-AVX512 test; those skips are not execution proof.
The [existing Windows lane](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34304414299)
recorded Release 49 and Debug 32 registered, each with one AVX512 skip, and
focused ASan 1/1. This is not generated-region Windows validation. Hosted counts
depend on their exact configured scopes and do not replace the local counts
above. The 21 checks include duplicate push/PR triggers, not 21 configurations.
