# Independent review: #119 null contraction maps and malformed counts

**Accept the bounded fix; no blocker.** Reviewed source
`a244d6a1f1f69981223425c048de76e268169c71` against
`1c1221e89272eb2ad6ff058b8e26de4653f3a93d` on 2026-11-26. Read the root
`AGENTS.md`, [issue #119](https://github.com/onlyxItachi/MatcoreDSL/issues/119),
the full production/test/report delta, the verifier and its callers. I did
not author the code or tests. This accepts the internal verifier behavior,
not a release or a new source-language capability.

- `MatcoreContractionModel.cpp:254`: verify both lengths equal three, then
  check every handle, then access any context. The configured 21.1.8
  `mlir/IR/AffineMap.h:132` implements the handle's boolean conversion as a
  pointer comparison, so the new predicate itself does not dereference null
  storage or reject a valid map with no results. The existing shared-context
  check, canonical regeneration and full topology equality remain intact.
- The encoder (line 284) and structured checker (line 390) both return
  immediately when the verifier rejects, before attribute construction or
  carrier comparison. The decoder compares against a freshly built and
  encoded canonical model rather than trusting supplied indexing maps.
  Structured/bufferized handoffs, CPU and fused candidates, and vector
  readiness continue to use canonical builders. No builder, serialized
  contract, public API, or lowering decision changed.
- The tests call all three interfaces on all seven nonempty null-slot
  subsets, including first, last and all-null. The new length tests add
  **24 distinct checks**: map and rank vectors independently at sizes 0, 1,
  2 and 4 across those interfaces. The untouched counterpart is valid, extra
  map slots are non-null, and the exact expected count diagnostic prevents
  a downstream mismatch from masquerading as success. Separate controls
  retain mixed-context rejection, clearing stale diagnostics, and successful
  verification/encoding of a non-null zero-result DOT output. Existing
  operation/orientation and upstream carrier checks were not replaced.

Commands run in the candidate worktree, without launching a build:

```sh
gh issue view 119 --repo onlyxItachi/MatcoreDSL --json title,body,url,state,comments
git diff --check 1c1221e89272eb2ad6ff058b8e26de4653f3a93d...a244d6a1f1f69981223425c048de76e268169c71
../build-null-maps-release/bin/matcore_mlir_contraction_model_tests
ctest --test-dir ../build-null-maps-release --output-on-failure -j1 \
  -R '^mlir\.structured\.contraction-model-v1$'
```

The diff check passed. The existing focused binary reported **325 checks,
0 failures**; CTest passed **1/1** with no skips. Its CMake cache points at
this source tree, Release, native frontend, Matcore MLIR and experimental
regions ON, OpenBLAS OFF, Clang/LLVM 21 and private MLIR 21.1.8.
The implementation report records a prior guard-removal negative control and
six affected passing CTests; I did not repeat those runs, rebuild, or run the
full suite or hosted CI. No malformed public source program, emitted artifact,
device execution, or performance result is claimed.
