# Source-time strict output pattern: core implementation v1

Starting canonical engineering checkpoint:
`3077d9c3f5df1479f2fcadfcceeb9444018d6ba0`.
Core agent worktree: `mdslc-work/source-pattern-v1/core`;
branch: `agent/source-pattern-core-v1`.

## Implemented boundary

`mdslc-region` accepts explicit `--gemm-output-tiles M,N`, with canonical
decimal parameters in `[1,64]` and exactly one forced `generated-strict`
candidate. This initial boundary is single-TU complete executable linking;
`-c`, `--program`, other candidates and `strict-fused-pair` are rejected.
`none` and publication-read forwarding retain their existing independent
source/effect contract. There is no source grammar, runtime callback, submitted
IR, artifact importer, cache, cost policy, ISA request or GPU mapping change.

The driver invokes the existing strict CPU issuer in-process for its native
baseline target and installation-owned ASan profile. It retains the immutable
semantic/structured/bufferized/Transform/scheduled/LLVM witnesses, compiles one
requested leaf and links a unique per-source/request isolated candidate DSO.
The private pinned relink archive reproduces the configured adapter and other
already-DSO-qualified leaf/image closure except the baseline strict leaf. Its
C++ adapter/platform/image objects are PIC; previously qualified external leaf
objects are reused unchanged. The requested leaf explicitly compiles with
`-fPIC`. Canonical Runtime remains the sole provider-policy owner.

The generated DSO retains the existing version-script export boundary and
`-Bsymbolic` local ownership. Native ELF checks require exact stock adapter
export names/binding/type/visibility, local leaf/wrapper functions, no text
relocations or private loader-search path, unique request-bound SONAME and the
pinned stock dependency closure plus explicit canonical Runtime. The issued
leaf object admits only its two strong functions and bounded conforming
`memset`/installation-ASan imports/bookkeeping. The kit is a non-thin native
ELF archive with the genuine adapter owner and no baseline definition or test
authority. Its complete bytes and the exports map are pinned in the driver,
but captured lazily only for the explicit new request.

A private-constructed immutable compilation binding holds the original
authenticated evidence seal, policy, optimization, candidate DSO digest,
worker contract and installation sanitizer profile. Both the instance input factory and region compilation check
that binding. This rejects another source/region seal, policy/optimization,
substituted same-ABI DSO or worker-contract substitution; an editable hash or
inspection manifest does not grant execution authority. Program composition
also refuses this binding.

Successful source-pattern region compilation additionally retains immutable
special-case publication bookkeeping: the exact source issuance binding,
compiler-produced LLVM digest and frozen host/helper snapshot pointers. Typed
source-pattern publication checks this pairing before linking and at every
recheck. Another instance, swapped compiled source/helper or modified LLVM
result is rejected. The raw/default publication helper has no per-source DSO
selection parameter. This is not a general IR certificate/importer and does
not expand the existing trusted in-process default compiler API into a hostile
caller boundary.

## Artifact lifecycle

The new `OUTPUT.mdslc-private` bundle is created exclusively with mode `0700`
before host input capture. Existing files/directories/symlinks are untouched.
All source, helper, installation, issued-stage, object, DSO and inspection
manifest bytes are rechecked before executable publication. Final ELF
dependencies must contain exactly the requested DSO and no stock/foreign
candidate DSO. The existing atomic no-clobber executable publication happens
only after the complete dependency bundle is ready. Normal failures clean only
the directory created by this invocation after verifying its physical identity
and traversed path chain; a substituted path is preserved.

The private manifest binds the source host-closure identity, region,
policy/optimization, exact tuple, target/toolchain, headers, pinned compiler,
linker, archive, export map, canonical runtime/provider/accelerators and all
issued-stage/LLVM/object/DSO identities. It is inspection-only and never read
back as authority. No crash-atomic multi-file publication or post-return
loader/self-authentication guarantee is claimed. An abrupt crash can leave an
orphaned private bundle. Moving the executable/bundle after compilation is not
a newly qualified relocation contract; installation relocation before source
compilation remains an independent acceptance test.

## Test seams and evidence at this handoff

The noninstalled `matcore_region_driver_support` target provides deterministic
`after_issue`, `after_object`, and `after_dso` hooks only through
`SourcePatternInstance::issueForTesting`. Each runs after immutable capture
and is immediately followed by normal mandatory rechecks. No CLI/environment
selection exposes a hook. A compiled-only `Installation::pinnedBuildTreeForTesting`
factory allows standalone compiler tests to use the exact pinned build tuple;
normal copied/installed drivers retain the existing no-build-fallback rule.

Core checks performed: read complete repository `AGENTS.md`, verified exact
clean starting HEAD, inspected the issuer, source closure, artifact ownership,
private DSO and installed-package contracts, and ran `git diff --check` without
errors. No build, executable, sanitizer or target qualification was run by
this core agent before handoff. The integration owner owns the build lock,
compilation, independent tests, installed-consumer tests and full regression
qualification. No performance or expanded-target claim follows from this
implementation.

## Subsequent bounded core corrections

Independent static review identified a cross-wired publication counterexample:
correctly compiled source/instance A could be passed to the old raw publisher
with instance B. The source-specific typed publisher and immutable produced-IR
pairing above close that counterexample and separately reject LLVM result
mutation. The reviewer accepted that narrow shape statically; this is not a
substitute for the direct negative publication tests.

The integration owner reported preliminary ASan source execution of 3x5 passed
but 64x64 refused without publication at the closed leaf import diagnostic for
`__asan_report_load_n`. This agent re-issued the existing built ASan issuer's
64x64 primitive and compiled only that diagnostic object with the source
route's exact `-O2 -fPIC -ffp-contract=off -fsanitize=address` flags. LLVM nm
observed the previously admitted init/memset/register/load4/load8/version and
weak start/stop globals, plus exactly `__asan_report_load_n`. That observed
standard ASan diagnostic import is now admitted only for the sanitizer
profile; no arbitrary ASan prefix is accepted. The diagnostic artifacts are
under `/tmp/mdslc-pattern-asan-imports-aIxcuC`. This was object/import evidence,
not raw-leaf execution, final source execution or complete qualification.
