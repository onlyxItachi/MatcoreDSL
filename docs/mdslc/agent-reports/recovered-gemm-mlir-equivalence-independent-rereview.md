# Independent rereview: recovered GEMM MLIR equivalence hardening

Date: 2026-08-11

Reviewed commits:

- `230e14278cb9d63af031b105238e4f57a3c210a2` — implementation and tests;
- `1ee35a67433e151e83eb1b10725319c7eb37fdc9` — updated evidence report.

Review mode: independent, read-only implementation review followed by the
focused exact LLVM/Clang/MLIR 21.1.8 tests. This reviewer did not implement the
hardening commit.

## Verdict

Accepted. The three prior medium findings are closed. No high or medium finding
remains in this bounded Milestone D analysis/equivalence scope.

The result remains analysis-only. This verdict does not authorize recovered
C++ rewriting or execution and does not generalize the evidence token into a
public or cryptographic trust API.

## Prior finding 1: mutable frontend result as authorization

Status: closed.

The native frontend now resets its caller-provided `Result`, parses a frozen
source snapshot through real LibTooling, rereads the source for stability,
rejects every frontend diagnostic, verifies Matcore IR v0, and only then issues
`AuthenticatedNativeFrontendEvidenceV1` in explicit recovery-inspection mode.
The evidence is non-default-constructible and stores a
`shared_ptr<const Payload>` whose private payload contains copies of the complete
effective `Options` and `Result`. The recovered bridge accepts only this token
and reads it through the private internal accessor; it no longer accepts mutable
`Result` or `Options` arguments.

The tests mutate the public post-extraction result across strict/relaxed state,
in-bounds and coordinated ranges, binding names, source/function display names,
FP proof, guard order, source bytes, diagnostics, and the original options. The
sealed bridge result remains byte-identical. Copying every relaxed candidate
literal onto a strict public result also cannot change the strict evidence or
authorize recovery. This directly closes the earlier relabel/range/binding and
display-field forgery paths.

The issuer header remains source-internal and is compiled only into the native
frontend library. This is an internal compiler trust boundary, not protection
against arbitrary hostile code already able to compile inside the compiler.
That bounded trust model is appropriate for this milestone and is stated here
to avoid implying a cryptographic seal.

## Prior finding 2: structural equality described as authentication

Status: closed.

Authenticated comparison now requires independently issued native evidence for
both sides. The explicit side authenticates a live native v0 operation against
the sealed source snapshot, compilation/source identity, site ID, bounded call
and argument ranges, and exact line/column, then performs the checked v0 to v1
to Matcore MLIR bridge internally. The recovered side uses its separately sealed
native candidate and complete recovery checks. The comparison exposes only the
equality result and normalized fingerprints, not an executable MLIR wrapper.

The diagnostic-only APIs are honestly named
`fingerprintStructuralMathematicalGemmV1` and
`equivalentStructuralMathematicalGemmV1`. Their comments expressly exclude
source authentication and execution permission. Their normalized contract
retains tensor/result types, dynamic-symbol relationships, strides, layout,
memory space, alignment, mutability, accumulation, requirements, alias/effect
contracts, synchronization, and expanded numerical semantics while explicitly
excluding source/provenance, implementation policy, origin, and profile labels.

The focused test invokes `createClangLibToolingFrontend()` for both the trusted
explicit `.mdsl` call and canonical ordinary-C++ loop. Parsed/golden MLIR is used
only by the separately named structural diagnostic path. Strict recovered
evidence is rejected before authenticated comparison.

## Prior finding 3: non-durable analysis-only marker

Status: closed for the current executable CPU lowering boundary.

`lowerExplicitGemmToCpuRuntimeDispatchV1` clears caller records before every
check, rejects the presence of `mdsl.analysis_only` regardless of its encoded
value, verifies the closed explicit-v1 envelope, and then requires exact native
`mdsl.producer = clang-libtooling-v1`. Bootstrap-produced explicit envelopes
therefore remain inspectable but cannot authorize this executable lowering.
Records are accumulated privately and published only after every site succeeds,
so analysis-only, bootstrap, and later per-site failures are transactional.

The recovered test starts with sentinel output records and verifies that both an
analysis-only module and an otherwise explicit bootstrap-producer module fail
and clear the records. Future executable lowerings must enforce the same taint
and producer boundary; this verdict covers the current CPU runtime-dispatch
lowerer only.

## Build and test evidence

The existing out-of-tree Release build was authenticated as:

- source: this worktree's `compiler/` tree;
- C/C++ compiler: `/usr/bin/clang-21` and `/usr/bin/clang++-21`, version
  `21.1.8`;
- Matcore MLIR: enabled against the pinned MLIR 21.1.8 package;
- build scheduling: `nice -n 10`, at most `-j2`.

The three reviewed targets were current in the serialized build window, and the
focused tests were rerun independently:

```sh
nice -n 10 cmake --build "$MDSLC_MLIR_BUILD" --target \
  matcore_mlir_semantics_tests \
  matcore_mlir_recovered_gemm_tests \
  matcore_mlir_cpu_runtime_lowering_tests -- -j2

ctest --test-dir "$MDSLC_MLIR_BUILD" --output-on-failure -j1 -R \
  '^(mlir\.semantic\.core|mlir\.semantic\.recovered-gemm|mlir\.cpu\.runtime_dispatch_lowering_v1)$'
```

Result: 3/3 CTests passed with zero failures. The implementation evidence also
reports the direct executable counts: semantic core 204/204, recovered bridge
78/78, and CPU lowering 18/18.

## Residual bounded limitations

- Recovery remains Linux-only, internal, and not installed in this commit.
- Runtime guards are recorded but neither established nor executed here.
- Only the reviewed canonical row-major F32 loop is recognized.
- Structural mathematical equality does not imply rewrite permission,
  profitability, guard dominance, backend legality, or numerical equivalence
  outside the exact retained fields.
- Any future executable lowering must independently reject analysis-only data
  and require its authenticated producer; this review does not approve an
  unimplemented backend.

Final rereview verdict: accepted with no unresolved high or medium finding.
