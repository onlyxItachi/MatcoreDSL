# Independent parameterized output-pattern core audit v1

Date: 2026-10-11. Reviewer: independent interoperability/audit lane; this lane
did not implement the core. Verdict: **bounded static acceptance; no concrete
correctness or authority blocker found** in the reviewed closed family.

## Immutable scope and evidence boundary

Reviewed core commit: `a5d94e9becc028bf33844c10d5dfc80c6ba2d5a8`, against
canonical starting main `4c222dd68176de19fcf82696939b74be73f80968`.
The four reviewed core files matched that commit with no working-tree diff:

| File under `compiler/lib/mlir/` | SHA-256 of reviewed content |
| --- | --- |
| `MatcoreGemmOutputPattern.h` | `7597f62522da578957edfbb08a32a4348d98bc562186f0e2fc7e704f80a96086` |
| `MatcoreGemmOutputPattern.cpp` | `d325f46ef65de57df4d0c84de703b72384b7564292e1378b42c2758a19700c64` |
| `MatcoreCpuGemmCandidate.h` | `b6ee113038c767178fcf546c1e06b24645959dc7c4a968a77f8cc909a8557694` |
| `MatcoreCpuGemmCandidate.cpp` | `aec8bfa26b008d2d2785cd249bf55ea5bcf58a36ba030c42e7659f1b32756fc8` |

Read-only supporting inspection covered the unchanged source adapter,
descriptor construction and full FP scope, plus the independent fixtures and
root-owned CLI/CMake/install/workflow seams. This reviewer ran no configure,
build, issuer or executable test. Only this report and the separately assigned
interop document were authored by this lane.

## Legality checks challenged

- Coverage and tails: `MatcoreGemmOutputPattern.cpp:127-152,195-312` checks the
  lowered scalar envelope: full positive-zero C fill, seven loops, tile origins
  zero, parameter-bound tile steps, complete A.M/B.N bounds, local unit steps,
  `min(tile, extent-origin)` tails and exact base-plus-local output indices.
  The unit-tile constant bound is legal inside the already bounded tile loop.
  Fill precedes all contraction tiles; return retains original C identity.
- Strict arithmetic: the same envelope requires one unflagged f32 multiply of
  ordered A/B loads and one unflagged add of old C then product, stored back to
  that same output. K is the complete increasing `0..A.K` unit-step loop, not a
  partitioned or vector reduction. No padding or extra reduction terms are
  admitted. Interchanging independent output traversal does not change a
  particular output's K sequence.
- Alias and effects: only A/B/C memory operations and bounded index glue occur;
  no allocation, copy, provider call or publication belongs to this stage.
  A/B may overlap because neither is written. C/input disjointness remains a
  caller obligation, established by the adapter's fresh private result while
  inputs remain alive (`closed_host_v1.cpp:469-488,542-568`). The existing LLVM
  fact marks only aligned C data, not input or descriptor pointers
  (`MatcoreCpuGemmCandidate.cpp:322-356,594-600`).
- Derivation and failure: checked canonical bufferized input is cloned before
  Transform application; failed partial mutations cannot damage a retained
  original. The scheduled module must also match pinned upstream replay for
  the exact parameter pair (`MatcoreGemmOutputPattern.cpp:316-394`). Replay is
  not an independent proof of the upstream implementation. In MLIR 21.1.8,
  the selected equivalence comparison retains properties, attributes, types,
  graph and SSA checks while ignoring locations. Its commutative-operation
  handling is why ordered FP operands are additionally checked by the scalar
  envelope. [Pinned upstream comparison implementation](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/mlir/lib/IR/OperationSupport.cpp#L708-L919).
- Lowering boundary: the issuer still constructs the built-in primitive itself,
  then checks two executable definitions, one scalar fmul/fadd footprint and no
  fast-math permissions. The added call allowance is only the pure two-argument
  i64 `llvm.smin` intrinsic in the built-in leaf; it is not an external-call
  importer (`MatcoreCpuGemmCandidate.cpp:455-605`). Known target/ISA refinement
  follows those checks; feature strings do not prove physical capability.

## Descriptor next-step overflow proof

Raw tiled callers need the documented sufficient condition
`M <= INT64_MAX-(tile_m-1)` and similarly for N. Representable zero-footprint
descriptors alone do not establish this: a signed tile IV can overflow its final
increment for enormous empty dimensions. The raw helper does not add runtime
guards; this is a private caller precondition, not a public source-size limit.

The actual source route satisfies the condition without new guards:

1. `extent()` bounds signed dimensions, element multiplication and f32 bytes
   by `PTRDIFF_MAX` before output allocation
   (`closed_host_v1.cpp:364-377,469-473`).
2. A nonempty result has M,N >= 1, hence each is at most
   `M*N <= PTRDIFF_MAX/4`. For every admitted tile <= 64, the last next IV is
   at most its extent plus 63, safely below `INT64_MAX` on the admitted LP64
   tuples. Positive input byte bounds likewise protect the full K traversal
   and row-major element indices.
3. Empty output and zero K bypass generated invocation before the candidate
   switch (`closed_host_v1.cpp:598-606`). Huge empty dimensions therefore do
   not enter tile loops. A huge nonempty K0 output still encounters the old
   byte-extent failure before allocation, not a new tile-specific refusal.

This proves the relevant source guard implication; it is not an execution of
an enormous raw descriptor. The unchanged full FP scope normalizes nearest-even,
gradual underflow and masked traps, checks controls, and restores caller controls
and status before issuing the result (`closed_fp_environment_v1.cpp:39-95`;
`closed_host_v1.cpp:570-571,709-717`). Thus reordered independent output work
does not expose a new FP trap/status or recoverable failure frontier.

## Provenance and qualification limits

Original semantic/structured/bufferized witnesses are printed before scheduling.
The issuer has no submitted-IR argument; inspection derivation/verifier helpers
do not issue source authority. Extra semantic/authority attributes are rejected,
and the manifest identifies builtin-only source authority. The unchanged source
Program/witness, guarded adapter, frozen host and private-artifact ownership
remain required. Matching replay or hashes is not a source-authentication,
storage-legality, installed-consumer or execution certificate.

The integration owner separately reported 32/32 new output-pattern tests and a
120-test generated-CPU/candidate/source regression selection passing with zero
skips, with tests committed as `e866903`. These are parent-run execution reports,
not tests run by this reviewer; the selections overlap and must not be summed
as distinct coverage. The owner also reported seven retained issuer
configurations/49 artifacts byte-identical to baseline. Those results are
separate from the static guard proof above and do not establish full clean
suite/package, hosted CI, AArch64/GPU physical execution or performance.

Earlier seam review found a missing output-tiled workflow expectation, a tile
macro mismatch and a stale sanitizer count. Reinspection found those resolved:
the harness and CMake both use `MDSLC_PATTERN_TILE_M/N`, workflow gates select
32 pattern tests and 232 focused sanitizer tests, and output-tiled has its own
schedule expectation. Final wiring and full qualification remain integration
owner responsibilities. No public format, new source admission, K tiling,
automatic policy, GPU mapping or general equivalence-prover claim is accepted.
