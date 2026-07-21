# Final adversarial review: standalone CPU bootstrap

## Review target and ownership

- Reviewed integration head: `b2b4706` (`mdslc/bootstrap-v0`).
- Production rewrite fix reviewed: `039f56d`.
- Review worktree: `/home/hamza-usta/MatcoreDSL-wt-final-review`.
- This lane changed only this report. It did not edit compiler, runtime, driver,
  build, test, package, example, or legacy production files.

The review covered valid-C++ source forwarding, structural AST recognition,
trusted-header identity, source ranges and diagnostics, deterministic JSON IR,
generated naming and multi-TU behavior, host rewrite semantics, C ABI/runtime
validation, driver argument classification, dependency files, installation,
consumer rebuilds, native artifacts, and documentation claims.

## Recommendation

Accept the implemented CPU bootstrap for integration within its declared
scope. No unresolved high- or medium-severity implementation defect remains at
the reviewed head. The overall architecture proof must remain **partially
passed** until the AST-JSON bootstrap is replaced by a coherent Clang 21
LibTooling frontend that authenticates the annotation payload and uses native
canonical-declaration and source-manager APIs.

The final handoff still needs the documentation/evidence refresh listed below;
those are integration follow-ups rather than production-code defects.

## Independent validation evidence

Fresh Release tree:

```text
/tmp/matcoredsl-final-review-b2.WHhQXI
```

- CMake selected `/usr/bin/clang++-21` 21.1.8.
- Ninja completed 15/15 steps with `-j2`.
- CTest completed 4/4 in 31.09 seconds with `-j1`.
- Direct integration-matrix execution passed 63/63 active cases with zero
  failures; six future capabilities were reported separately and not counted.
- The previously failing macro-active/raw-string include case now compiles and
  executes correctly.
- Focused namespace, source-line, inactive-include, macro-include, and default
  policy cases passed 5/5.

Fresh Debug tree:

```text
/tmp/matcoredsl-final-review-debug-b2.gYkt4q
```

- Ninja completed 15/15 steps.
- CTest completed 4/4 in 59.15 seconds.
- The strict runtime dynamic-symbol check passed in Debug.

Independent saved-artifact and external-link evidence:

```text
/tmp/matcoredsl-final-review-artifacts-b2.6U5qjd
```

- `gemm.o` is ELF64 x86-64, type `REL`.
- It defines `main`, one stable global C++ call-site wrapper, and one generated
  C backend entry, while leaving `matcore_runtime_gemm_f32_v0` unresolved.
- Ordinary `/usr/bin/clang++-21` linked it against `libmatcore_runtime`.
- `ldd` resolved `libmatcore_runtime.so.0` from the selected Release tree.
- Execution printed `host-before` and `MDSLC CPU GEMM PASS`.
- The requested host, JSON, sites, stubs, backend, component-object, and
  combined-object artifacts were present.

The complete CTest pass also exercised the installed `find_package` consumer,
source and included-header regeneration, stable depfile behavior, no-op
rebuilds, runtime descriptor failures, deterministic generation, serialized IR
mutation rejection, multi-TU symbol isolation, and positive/negative frontend
contexts.

## Resolved implementation findings

The final diff and regressions confirm resolution of the earlier review items:

- trusted operation declarations require canonical equality with the
  tool-owned public header path; a shadow header is not intercepted;
- the serialized v0 verifier checks the complete emitted field set, including
  matrix metadata, alias requirements, effects, policy, ranges, and site IDs;
- site IDs include stable source identity, exact contents, call offset, and
  operation kind, and distinct same-spelled translation units co-link;
- only the direct callee wrapper shape is accepted; explicit indirect forms
  are rejected;
- parsing and generation use one verified source snapshot;
- unsupported final-link modes and opaque linker forwarding are rejected
  instead of being silently reinterpreted;
- source/output and depfile/generated-artifact aliases are rejected;
- generated code no longer guesses a textual header-insertion location;
- global call-site declarations work from named namespaces and class methods;
- multiline replacements preserve source line count and observable `__LINE__`;
- the runtime validates ABI headers, reserved fields, dtype, rank, shape,
  layout, mutability, residency, target, fallback, alignment, overflow, and
  full storage-range aliasing before execution;
- the runtime shared object exposes only the intended C entry in Release and
  Debug.

No hidden allocation, host/device copy, target fallback, or exception crossing
the C ABI was found. The CPU GEMM is synchronous and matches the independent
oracle for the tested shapes.

## Remaining known limitations

- The frontend is deliberately labeled `clang-ast-json-bootstrap-v0`. Clang's
  JSON AST does not expose the `AnnotateAttr` payload, so this is not the
  required final LibTooling implementation.
- Linux, Ninja, Clang 21, one `.mdsl` input, rank-2 row-major contiguous host
  `f32`, CPU target, synchronous execution, and `fallback=error` are the only
  implemented execution contract.
- C++ module translation units, custom install directory conventions, and
  driver-managed shared/static/PIE output modes were not validated. The driver
  directs unsupported link modes through `-c` plus an ordinary external link.
- The source file itself is snapshot-checked; the complete user-header graph is
  dependency-tracked but is not frozen against concurrent edits across the
  split extraction/compilation stages.
- Generated multi-file publication is atomic per file rather than as one
  directory transaction. A publication failure can leave saved temporary
  files, but the driver stops and does not link them.
- CUDA/cuBLAS, AMD/HIP, Metal, MLIR lowering, device capability planning,
  `gemv`, `gevm`, and `relu_gemm` are not implemented or claimed.

## Documentation and evidence follow-ups

Before final handoff, refresh `docs/mdslc/STATUS.md` to:

1. identify `b2b4706` (or the later documentation/report head) as the reviewed
   code state;
2. report 63/63 active matrix cases rather than 60/60;
3. replace the obsolete exact-one-direct-include limitation with the current
   include-free forward-declaration preamble and source-line-preserving rewrite;
4. record the exact-head Release and Debug evidence above; and
5. rerun the sanitizer-instrumented generated path at the final code head, or
   explicitly label the existing sanitizer result as predating `039f56d`.

These follow-ups must be completed before presenting `STATUS.md` as the final
review record. They do not change the recommendation for the reviewed CPU
implementation.
