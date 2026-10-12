# Independent source-time output-pattern architecture review v1

## Verdict and exact scope

Static review accepts integration head
`01362399b31cddb7814e928ed72b8090978564a2` within the documented bounded
source-time strict GEMM contract. No unresolved static architecture or ownership
blocker was found. This is not build, execution, installed-package, sanitizer,
physical-target, performance or hosted-CI qualification.

The engineering baseline was canonical main
`3077d9c3f5df1479f2fcadfcceeb9444018d6ba0`. The final core reviewed was
`e026f15bc306ebd097ccf4e2940a9de0b3f17867`; the reviewed integration head contains
its production code unchanged, plus the independent tests and contract
documentation. This reviewer read the applicable `AGENTS.md` completely, reviewed
the implementation as it evolved, then inspected the immutable final core and
integration changes. Only this report is authored in the review branch.

The accepted scope is an explicit `--gemm-output-tiles M,N` request with canonical
decimal parameters in 1..64, exactly one forced `generated-strict` selection,
single-TU native baseline Linux x64/AArch64 executable generation, and either
`none` or separately checked publication/read forwarding. Compile-only, program,
other candidates and fused-pair requests refuse. Static target branches are not
physical qualification of either architecture.

## Competing ownership designs

The existing adapter calls its generated leaf directly inside an isolated DSO.
`PrivateRuntime.cmake` applies the exact export script and `-Bsymbolic`; the leaf
and implementation definitions are local. An executable-level replacement object
therefore cannot implement the requested schedule through the existing adapter.

- **Selected: pinned PIC relink kit plus the existing in-process issuer.** The
  kit omits only the baseline leaf, retains the same adapter and other leaves,
  and does not embed the canonical Runtime/provider policy owner. The issuer
  accepts checked parameters, not source-submitted IR. The resulting DSO retains
  intrinsic ownership without changing the runtime API or source grammar.
- **Executable override or whole-linking the adapter into the executable.** An
  override conflicts with the existing local binding. Moving implementation/STL
  definitions into ordinary executable resolution would discard the established
  ownership boundary; an archive alone is not an isolation proof.
- **Production callback or registry mutation.** This would introduce a different
  authority/ABI contract. The production adapter intentionally has no candidate
  setter or callback. It is not needed for the requested bounded pattern.
- **Compile installed runtime sources for every request.** This would enlarge
  private compiler inputs and potentially require runtime, STL, CUDA/HIP or other
  development headers at consumer compilation. Already-compiled relink material
  avoids that source/build dependency.
- **Enumerate all tile pairs at build time.** This is a catalogue rather than an
  on-demand lifecycle and conflicts with generating only the requested instance.
- **Install and spawn the issuer executable.** This is possible with another
  pinned tool closure, but the current build-only CLI is not source authority.
  Reusing the issuer in process avoids adding that executable lifecycle.

## Challenges found and disposition

1. **Source/instance input cross-wire.** Initially, an instance for source A could
   supply compiler inputs while the region compiler consumed another valid source
   B. Normal driver ordering was correct, but the pairing was not mechanically
   checked. `SourcePatternCompilationBinding::matches` now checks the genuine
   sealed Program identity, frozen closure, selected region/entry, policy,
   optimization, installation sanitizer profile, candidate bytes and worker
   contract. `compileRegion` checks it before emission; program composition
   rejects this binding.
2. **Compiled-output/private-instance cross-wire.** Correctly compiling A and B
   did not initially prevent linking A's LLVM with B's instance. Successful region
   compilation now creates a private immutable `SourcePatternCompiledBinding`
   holding the exact issuance binding, LLVM digest and host/helper snapshot
   identities. Typed publication rechecks this pairing before native linkage and
   through final publication. The raw/default publication API cannot select a
   source-time DSO. Same-source different-tile swaps and changed well-formed LLVM
   bytes are also excluded.
3. **Final publication owner/output association.** The final core follow-up checks
   the original issuance `Installation` object and the exact output associated
   with the exclusively created bundle. A byte-identical Installation copy or a
   different output cannot be substituted into typed publication.
4. **Dangling LLVM buffer identifiers.** The initial ELF/archive readers supplied
   temporary path strings as non-owning `MemoryBufferRef` identifiers. Stable
   literals now outlive the parsed objects. Artifact payload buffers remain owned
   by the captured artifacts.
5. **Generated-code sanitizer import envelope.** This reviewer flagged that large
   output tiles could require a reporting symbol absent from the initial narrow
   whitelist. The integration/core owners reported an actual 64x64 sanitizer
   refusal for `__asan_report_load_n`. The final code admits that exact symbol
   only in the sanitizer profile, not an arbitrary `__asan_*` prefix. The reported
   measurement and subsequent execution checks are not this reviewer's runs.
6. **Ineligible forwarding test discriminator.** A distinct descriptor that MAY
   alias C is not a source proof of the same resource/version. The initial runner
   incorrectly demanded forwarding from that control. The final runner preserves
   its semantic parity checks and uses a separate same-C-descriptor fixture for
   the actual `readForwarded` call discriminator. The implementation was not
   relaxed to satisfy the incorrect assertion.

## Concrete static acceptance

- `SourcePattern.cpp::SourcePatternInstance` invokes the unchanged closed CPU
  issuer with output-only tiling, the fixed native baseline target and the
  installation profile. It does not accept user IR, objects, feature strings,
  manifests or cached instances. Original witnesses, increasing full scalar K,
  separate f32 multiply/add and existing adapter guards remain in their existing
  issuer/runtime owners.
- `verifyKit`, `verifyLeafObject`, `verifyCandidate` and `verifyExecutable` check
  the native ELF/complete archive envelope, absent baseline kit definition,
  permitted leaf definitions/imports, exact stock adapter export contract,
  local leaf/wrapper, symbolic binding, unique request-bound SONAME and expected
  DSO dependencies. The final executable must name exactly its requested
  candidate rather than the stock or another patterned DSO. Export inventory
  alone is not treated as these ownership proofs.
- The request/source/region/target/toolchain/kit/runtime/stage/object binding is
  retained in compiler-owned construction and checked artifacts. The manifest is
  inspection-only and never imported as authority. The two narrow private binding
  classes have no public or serialized constructor; they are source-pattern
  bookkeeping, not a general certificate or optimizer framework.
- `SourcePatternBundle` creates the new mode-0700 sidecar before host capture and
  checks its identity/path chain. Existing files, directories and symlinks refuse.
  Cleanup only targets the still-identified compiler-created directory; a
  substituted path is preserved. Source directory and negative-lookup checks are
  not exempted. The production driver retains the bundle only after successful
  executable publication.
- `compileSourcePatternAndPublish` requires the original owner/output and exact
  compiled pairing. The private linking helper checks frozen inputs and captured
  artifacts before/after native compilation and again before the atomic
  no-clobber executable hard link, including rechecking the final executable.
- The kit is installed privately, and installed layout resolves it and the pinned
  export script from the installation prefix. Source-time compilation reuses
  compiled material rather than producer sources or build scripts. This is the
  design basis for the source/build-inaccessible installed gate, not a claim that
  this reviewer executed that gate.
- No production runtime/registry, GPU recipe, provider policy, default selection,
  numerical permission, source grammar, performance/cost model, autotuning or
  SPIR-V #89/PR84 work is added by this boundary.

## Independent tests inspected, not executed

The final `mutation_test.cpp` first compiles and validates three legitimate
source/instance pairs after precreating sources, bundles and stages. Its typed
publication controls reject a different physical source, same-source different
tiles and modified emitted LLVM before `host.ll` or `result` are populated. Owner
and output controls require their specific refusal diagnostics. The exact pair
has a positive publication/execution control in the test code.

Existing controls cover input factory/region compiler cross-wires, entry, policy,
optimization, sanitizer and worker changes; source/header changes after each
captured phase; valid foreign LLVM/object/DSO and stale-source substitutions;
inspection artifact changes; and sidecar path substitution with foreign-state
preservation. The connected runner checks actual private artifacts, multiple tile
requests, unchanged WHAT witnesses, numerical/effect oracles, no-clobber defaults
and sanitizer controls.

The separate eligible forwarding fixture preserves dynamic requested extent,
failure at the late-read frontier, earlier publication/owning observation and
same-source outcomes. Its runner requires the checked runtime call only when
forwarding is selected. A mathematical pass alone is not used to identify the
requested instance or transformation.

This reviewer ran no configure, build, CTest, source execution, sanitizer probe,
installed consumer, target run or hosted CI. Read-only Git/status/source checks
and `git diff --check` against the reviewed integration head passed. The
integration owner owns physical/installed/CI results and exact passing/failing
counts; none are inferred here.

## Retained limits

Compilation-time artifact authentication does not authenticate future loading.
The runtime/provider/standard-library loader and conforming allocator/thread
implementations remain trusted deployment preconditions, with no foreign
interposition. Emitted executables use absolute dependency paths: installation
relocation before compilation is distinct from moving an emitted executable or
its bundle afterwards, which is not supported. Crash recovery and hostile
concurrent same-user filesystem mutation are not claimed.

The special-case typed publication checks do not turn the pre-existing trusted
in-process default compiler APIs into a general hostile-caller protection
framework. There is no new executable importer, sandbox or public ABI. Final
engineering acceptance still requires integration qualification on the exact
candidate head; this static no-blocker verdict does not substitute for it.
