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

The actual OpenBLAS-disabled sanitizer CTest selection contains **117** tests;
the Release ON selection has 172 total. Full integrated regressions, installed
source-connected adversarial review and exact-head hosted CI are pending gates,
not implied by the focused result above.

## Claim boundary

This is one synchronous, private-output rank-2 f32 CPU candidate on Linux x86-64
and coherent 21.1.8. A passing source call is real generated execution only when
the local machine admits that candidate; a capability refusal is not execution.
Compiler-ASan instrumentation and actual generated-leaf sanitizer faults are
separate evidence. No whole-region transformation, fusion, automatic buffer
reuse, new device support, numerical-policy expansion or default schedule change.

Research x86-64-v3 timing does not authenticate this narrower compiled artifact
or its source overhead. No performance threshold or BLAS-parity claim follows.
Issue #15 remains partial. The next performance question is a fresh same-source,
same-contract, same-hardware comparison of the connected candidate and provider.
