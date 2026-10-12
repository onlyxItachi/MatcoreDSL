# Authenticated source-time strict GEMM patterns v1

Starting canonical main: `3077d9c3f5df1479f2fcadfcceeb9444018d6ba0`.
Status: implementation and qualification in progress; not yet merged or qualified.
This is a bounded continuation of [PR91's HOW pattern](PARAMETERIZED_GEMM_PATTERN_V1.md),
not a new optimizer, source language, numerical permission or scheduling policy.

## Independent architectural challenge

**Observed:** the existing issuer already constructs, checks and lowers a strict
GEMM with M/N output tiles in 1..64. The source driver authenticates a complete
C++ input closure, emits from a sealed Program and reserves trusted artifact
symbols against original host definitions. Its generated leaf is a localized
definition in a compiler-owned candidate DSO, not a replaceable executable symbol.

**Conclusion:** source-time instantiation is the next missing HOW lifecycle,
but simply accepting tile flags or linking another kernel object is insufficient.
The existing `-Bsymbolic`/local-export boundary would keep using the build-time
leaf, or relaxing it would reopen host replacement of compiler-owned code.

The selected design reuses the same runtime and issuer: an authenticated private
relink kit omits only the baseline generated leaf, and the compiler assembles an
isolated per-compilation candidate DSO containing the requested instance. There
is no source-visible callback, executable IR importer, runtime registry mutation,
runtime compiler or user-selected library. Original default routes remain intact.

Rejected alternatives:

- Executable symbol override: conflicts with intrinsic private-DSO ownership.
- User callback/function pointer: expands authority rather than instantiating the
  existing compiler-owned pattern.
- Finite catalogue of every tile pair: still build-time selection and unnecessary
  compile/storage growth; cannot prove an on-demand generation lifecycle.
- Installed runtime source compilation: unnecessary dependence on producer source
  and a second compilation of the runtime contract at every user build.
- Generic executable IR or vendor adapter: unrelated authority expansion.

## Bounded requested behavior

The proposed experimental driver spelling is `--gemm-output-tiles M,N`, with
canonical decimal values in 1..64, explicitly paired with `--candidate generated-strict`.
The request applies to the compilation, not to source tensor dimensions or
per-operation mathematical semantics. The first supported scope is single-TU
native Linux x64/AArch64 executable generation. Ordinary no-flag behavior is
unchanged. Compile-only, multi-TU program mode, other candidate policies and
strict-fused-pair selection must refuse the new request, never silently ignore it.
`none` and the existing separately checked publication/read forwarding may compose.

The existing baseline target and installation sanitizer profile are authoritative.
No CPU ISA request, target auto-selection, GPU schedule, K tiling, fast-math,
padding, measurement or cost model is introduced. Every output retains the same
increasing full scalar K reduction, separate f32 multiply/add, positive-zero
initial overwrite and existing FP environment law. NaN payload identity is not
added to the original strict contract.

## Required identity and lifetime contract

The compiler must bind the exact request and region, frozen source closure,
target/toolchain, owning runtime/relink-kit/exports, issuer-derived stages,
generated LLVM/object and candidate DSO. Any retained manifest is inspection
evidence, not an importable certificate or permission to execute supplied IR.
There is no cache admission or stale-instance reuse path.

The per-output private dependency directory is created before host capture;
existing files/directories/symlinks are not overwritten. All generated dependencies
must be complete and checked before publishing the executable. Failed compilation
must not publish an executable or destroy pre-existing user state. Only the
compiler-created dependency directory may be removed on failure. Dependencies
must outlive temporary staging and compilation; their deployment contract must
be explicit. Source directory/negative-lookup checks are not exempted to make
generation convenient.

The issued DSO needs its own request-bound identity, exact native ELF type/target,
private leaf, expected export boundary and allowed dependency closure. Existing
host-symbol ownership checks remain necessary but do not alone prove these facts.
Every staged input is rechecked through the final native compilation/publication
boundary. Unknown or substituted artifacts fail closed. Trusted post-build loading
remains a deployment precondition, not executable self-authentication or a sandbox.

## Acceptance requirements

1. Two independent tile requests from the same compiler installation must produce
   different parameter-bound derived computation, link that actual private leaf,
   and execute source-level numerical/effect oracles. Correct output alone is not
   a discriminator for use of the requested instance.
2. Illegal/malformed/duplicate requests and unsupported option combinations must
   refuse before output publication. No implicit fallback or extra authority.
3. Deterministic mutation controls must reject changed source/header/parameters,
   stale instances, changed LLVM/object/DSO and same-ABI substitutes where the
   contract admits an interception point. Test hooks stay noninstalled and absent
   from source/CLI/environment behavior.
4. Tail/noncommuting shapes, input aliases, late reads, zero/large-empty/overflow
   cases, strict numerical counterexamples, observations and ordered failure
   prefixes retain their original meaning. Original region witnesses do not change.
5. The installation must generate both instances after relocation and after its
   disposable producer source/build trees are inaccessible, with BUILD_TESTING=OFF.
6. Generated-code checks and sanitizer instrumentation must reach the issued leaf;
   existing source/default/provider/GPU/Windows regressions remain distinct.
7. Independent final review and hosted qualification must identify exact heads,
   skips, failures and supported target scopes before a normal engineering merge.

## Boundaries

No performance, BLAS parity, autotuning, cross-device schedule, universal compiler
interchange, rank-N surface, public API/ABI freeze or expanded numerical contract.
[SPIR-V #89](https://github.com/onlyxItachi/MatcoreDSL/issues/89) and draft PR84 remain
independent and unchanged. A future per-operation or multi-source specialization
requires its own ownership/composition proof; this increment must not promise it.

Validation results and the surviving implementation details will be recorded here
and in linked agent reports after the bounded acceptance requirements are tested.
