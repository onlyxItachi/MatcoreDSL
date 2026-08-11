# Semantic architecture owner report

Date: 2026-08-11

Branch: `mdslc/semantic-compiler-foundation-v1`

Baseline: `e5069758ad04bdb459de2026cad8498b47fda707`

## Ownership

This lane owned only:

- `docs/adr/0009-mdslc-semantic-compiler-foundation.md`;
- `docs/mdslc/ROADMAP.md`;
- `docs/mdslc/STATUS.md`;
- `docs/api/PRE_FREEZE_DECISIONS.md`; and
- this report.

No compiler, runtime, public header, test, workflow, package, legacy source, or
performance artifact was edited.

## Authenticated repository state

Read-only checks established:

- the branch starts at canonical `origin/main`
  `e5069758ad04bdb459de2026cad8498b47fda707`;
- PR #16 is merged normally from `mdslc/native-blas-parity-v1` to `main`;
- all visible PR #16 Linux OpenBLAS-required/OpenBLAS-disabled, generic
  build/test, Windows x64, and repository-hygiene checks passed;
- Issue #15 and GitHub milestone #5 remain open;
- no native-BLAS-parity completion tag exists; and
- the accepted Milestone 1--6 checkpoint tags remain present.

The previous status/roadmap wording still described the Milestone 7 partial
disposition as a local candidate pending hosted checks. This lane corrected it
to the merged, hosted-green partial disposition without turning it into a
parity claim.

## Architecture decision

ADR-0009 freezes the internal semantic responsibility boundary rather than a
public ABI:

- Matcore IR v1 remains the typed, deterministic capture/provenance DTO;
- the textual `mdsl` MLIR dialect becomes the compositional optimizer IR;
- Clang capture, MDSLC planning/transformation, and upstream machine lowering
  have explicit WHAT/HOW/MACHINE ownership;
- lowering consumes information only after downstream structure carries it or
  all dependent optimization decisions are complete;
- recognition and permission are separate, and unproven ordinary-C++ raising
  preserves the original program;
- numerical, domain, effect, alias, mutation, execution-intent, policy, and
  capability semantics are explicit verifier/planner inputs; and
- the first end-to-end proof reuses the existing CPU planner/runtime rather
  than replacing validated code merely to exercise MLIR.

The ADR records the A--H dependency graph, isolated worktree lanes, per-merge
gates, and the risks most likely to create semantic or ABI rework.

## Numerical-semantics correction

Matcore IR v1 carries accumulation dtype but does not encode enough numerical
policy to call the optimizer bridge fully lossless. Follow-up architecture
review therefore froze `explicit-gemm-f32-v1` with:

- F32 accumulation;
- contraction/FMA allowed;
- reassociation only among one output element's K-reduction terms;
- implementation-defined K reduction order with every term included once;
- NaNs preserved without assuming their absence, but no payload, sign,
  signaling-state, or propagation-order guarantee;
- IEEE infinity behavior under the selected legal contraction/order, with no
  `no-infs` assumption;
- relaxed signed zero;
- approximate math forbidden; and
- explicit destination overwrite with input/output aliasing, input mutation,
  and in-place operand transformation forbidden.

This is the existing explicit mathematical eDSL contract, not strict scalar
C++ loop order. It keeps native and OpenBLAS routes eligible only after their
own conformance checks. The bridge may not invent further permissions.

Recovered C++ loop nests require source-derived numerical proof and may not
inherit the explicit eDSL policy from pattern similarity. If proof fails, the
ordinary C++ is preserved.

The initial architecture commit is
`6b7870784264f9e8ef9401d37ef605d81458e715`. The numerical-profile follow-up
is `f603c9b5649e527762d7b3197a27a8738525f8fe`. The M1--M4 review-closure
commit is reported at handoff because recording its own object ID in its
contents would be self-referential.

## Milestone A review closure M1--M4

The accepted architecture review identified four medium findings. This lane
resolved them as follows:

1. **Precondition/fact separation.** V1 alignment and no-alias fields are
   required preconditions, not proven properties of runtime SSA values. Static
   proof or a dominating runtime guard is required before optimization uses
   them. Dynamic failure precedes packing and every destination mutation.
2. **Destination/result identity.** `mdsl.gemm` returns the post-overwrite
   semantic value tied to the explicit write-only destination, not new storage.
   Bufferization aliases result and destination storage, while the observable
   write remains non-DCE even when the SSA result is unused.
3. **Floating-point environment.** `explicit-gemm-f32-v1` requires
   round-to-nearest-ties-even, masked/non-trapping exceptions, and gradual
   subnormals with FTZ/DAZ disabled. Incoming exception flags need not be
   preserved and the exact post-call set may reflect the permitted contraction
   and reduction order. The current runtime does not yet check all these
   conditions; Milestone E must fail closed before mutation and prove backend
   conformance with subnormal/non-finite fixtures.
4. **HOW/MACHINE boundary.** Linalg, Tensor, MemRef, Vector, and generic GPU
   dialects remain HOW-level substrates while implementation alternatives
   exist. MACHINE begins when LLVM or a target-specific dialect/toolchain such
   as NVGPU/NVVM or AMDGPU/ROCDL structurally commits the machine choice.

These are architecture and future acceptance requirements. They do not claim
the current runtime has already implemented the new floating-point-environment
guard.

## Coherent MLIR 21 toolchain evidence

System LLVM/Clang development packages are Ubuntu
`1:21.1.8-6ubuntu1`. System MLIR is version 22. An APT simulation showed that a
system-wide MLIR 21 development install would remove the MLIR 22 surface, so
the toolchain lane made no system package mutation.

It extracted these exact Debian payloads to the user-local development prefix
`/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21`:

| Package | SHA-256 |
| --- | --- |
| `libmlir-21-dev_1:21.1.8-6ubuntu1_amd64.deb` | `ee47ca5eb635afc6d482b683a3d250541ddd446b02c1a66c6dc89743243ae1fd` |
| `libmlir-21_1:21.1.8-6ubuntu1_amd64.deb` | `5fec86b613126963f0247c6c65ce112f26da96a18bf2d6958534569fc939ba97` |
| `mlir-21-tools_1:21.1.8-6ubuntu1_amd64.deb` | `b33d0a9ede6939be48580c0ed12faa38e4a26b0e9a2736e8f9f7e4f55c88f397` |

The extracted payload occupies approximately 761 MiB. Configuration uses its
`MLIRConfig.cmake` with the system LLVM/Clang 21 CMake packages. The toolchain
lane built and executed:

- an in-process `clang-cpp` + MLIR + LLVM integration executable;
- a TableGen-generated toy dialect and operation; and
- a narrow static `MLIRIR`/`MLIRSupport` executable.

The narrow executable had no dynamic `libMLIR` dependency and no local-prefix
RUNPATH. The development prefix is not a product contract and must not be
hardcoded or exported by installed MDSLC artifacts.

## Documentation validation

Commands used by this lane included:

```text
git fetch origin --prune --tags
git rev-parse origin/main
gh pr view 16 --json ...
gh pr checks 16
gh issue view 15 --json ...
git tag --list 'mdslc-*'
llvm-config-21 --version
clang++-21 --version
find ... MLIRConfig.cmake
apt-cache policy libmlir-21-dev mlir-21-tools
git diff --check
bash tests/check_repository_hygiene.sh
```

The final documentation checks and exact commit SHA are recorded at handoff.
No implementation test is claimed by this documentation-only lane.
