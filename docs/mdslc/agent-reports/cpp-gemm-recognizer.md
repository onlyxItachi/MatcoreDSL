# Ordinary C++ GEMM recognizer lane report

- Implementation commit: `a0fda2ba08cf413552ab0cfdbdbdfb83e3eddf75`
- Independent-review fixes: `a0952560f4015aa613b7f35bb8c25690e3e383eb`
- Scope: native Clang 21 frontend inspection only
- Default behavior: unchanged
- Source replacement: unavailable
- Recovered Matcore IR v0/v1 emission: unavailable
- Recovered MLIR raising: unavailable until the separately reviewed adapter

## Implemented boundary

The native frontend now has an opt-in
`--inspect-recovered-gemm <report>` mode. A shallow outer-loop AST matcher runs
after Sema, followed by typed validation of the exact six-parameter,
row-major, F32, overwrite GEMM loop described by
`CPP_GEMM_RECOGNITION.md`.

The frontend records one typed `RecoveredGemmCandidate` per inspected outer
loop. Its state is one of `not_recognized`, `recognized_rejected`,
`recognized_guard_required`, or `raised`; this lane cannot produce `raised`.
The record authenticates the source snapshot SHA-256, source and compilation
identities, stable site ID, parameter bindings, exact SourceManager token
ranges, effective Clang FP options, optimization level, and Clang CodeGen
denormal modes. It carries unresolved runtime guards as requirements rather
than optimizer facts.

The authenticated semantic-contract field is populated only after the exact
typed six-parameter loop parses successfully. Plausible and not-recognized
loops no longer receive a fixed GEMM contract from the report serializer.

Inspection emits a deterministic, escaped, line-oriented diagnostic report.
It is deliberately not JSON and not another Matcore optimizer schema.
Recovered candidates never enter the explicit-operation Matcore IR v0/v1
capture path and never forge `matcore::mdsl::gemm` declaration provenance.

## Fail-closed coverage

The focused matrix covers:

- relaxed effective FP semantics as `recognized_guard_required`;
- strict FP, optimization level zero, disabled contraction, `-ffast-math`,
  dynamic rounding, and non-IEEE denormal modes as rejections;
- macro, template, lambda, `optnone`, loop-hint, user-header, volatile, atomic,
  observable-call, and early-return barriers;
- static and thread-local induction storage, cleanup attributes, unsupported
  local/function/parameter attributes, and non-default address spaces;
- effective Clang 21 `FPOptions::getFPEvalMethod()` state, including an
  explicitly extended-evaluation, no-system-header i386/`-mno-sse` rejection;
- OpenACC, OpenMP declare-simd/declare-target, and SYCL host and target-device
  contexts as explicit offload/parallel rejections;
- output accumulation and transposed-B near misses as `not_recognized`;
- renamed identifiers, harmless parentheses, UTF-8, CRLF, missing final
  newline, and two distinct sites;
- exact source digest and the required `outer_loop`, `accumulator_update`, and
  `output_store` proof-range names;
- deterministic report bytes, empty Matcore IR, source-byte preservation,
  bootstrap-mode refusal, dual-stdout refusal, and refusal to combine recovery
  inspection with host/sites/stubs/backend generation;
- transactional refusal of compiler arguments that attempt to produce or
  alias the inspection output; and
- ordinary `.mdsl` object generation, ordinary final link and execution, and
  absence of recovered Matcore runtime symbols.

Every recognized-rejected fixture is also compiled with Clang's normal `-c`
path and checked for absence of a recovered runtime symbol. Host fixtures emit
ordinary ELF relocatables; the separate SYCL target-device case emits the
expected LLVM bitcode artifact.

## Validation evidence

Debug, Clang/LLVM 21.1.8:

```text
frontend.native.focused                     passed
frontend.native.primary                     passed
frontend.native.recovered_gemm_inspection   passed
Recovered GEMM inspection: 336 checks, 0 failures
```

Release, Clang/LLVM 21.1.8:

```text
frontend.native.focused                     passed
frontend.native.primary                     passed
frontend.native.recovered_gemm_inspection   passed
Recovered GEMM inspection: 336 checks, 0 failures
```

Both configurations built `matcore-extract` with Ninja `-j2`; CTest ran
serially. No full repository test suite was run in this lane so other active
lanes could remain serialized.

## Independent-review resolutions

The consolidated review identified three high-severity fail-closed gaps. All
three were reproduced and resolved in `a095256`:

1. Local declarations now require `SD_Automatic`, `TLS_None`, and `SC_None`;
   cleanup and every otherwise unsupported local/function/parameter attribute
   reject recovery. Address-space qualifiers are authenticated recursively
   before canonical qualifier stripping.
2. Effective FP state now composes Clang 21 lexical `CompoundStmt`
   `FPOptionsOverride` records with the update expression override. Only
   `FEM_Source` is accepted; double, extended, unset, invalid, or an ambiguous
   lexical FP scope fail closed. Clang 21 reports `extended` for the exercised
   `-target i386-unknown-linux-gnu -mno-sse -ffp-eval-method=extended`
   fixture.
3. OpenACC and SYCL language/device state, OpenMP declare-simd and
   declare-target attributes, and OpenMP/OpenACC/SYCL statement nodes all
   reject recovery with `offload_or_parallel_context`.

Every new recognized rejection preserves the input bytes, emits an empty
Matcore capture module, compiles through ordinary Clang with the same semantic
flags, and has no recovered Matcore runtime symbol. SYCL device-only output is
Clang's target-device LLVM bitcode rather than a host ELF object; the test
asserts preservation and symbol isolation without mislabeling it as a host
artifact.

## Explicit limitations

- Inspection is opt-in and native-only; the bootstrap frontend is unchanged.
- Recognition is not execution permission. No guard, fallback, rewrite,
  semantic MLIR construction, planner call, or runtime call is generated.
- The current state is suitable for a future in-process adapter to the shared
  verified `mdsl.gemm` builder, but does not claim explicit/recovered semantic
  equivalence.
- Header, template, lambda, macro, barrier, and unsupported numerical cases
  preserve ordinary C++ rather than becoming compiler errors.
- Physical floating-point environment checks and guarded execution belong to
  the CPU integration milestone, not this frontend lane.
