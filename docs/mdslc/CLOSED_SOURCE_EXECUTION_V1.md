# Authenticated closed source to compiled host orchestration

This boundary connects real host-context admission to an ordinary generated
C++ object and executable using the private synchronous host adapter. It is not
an installed frontend and, at this checkpoint, GEMM uses the adapter's new strict
native scalar candidate, **not** the separately developed MLIR-generated leaf.

## Mechanically checked connection

```text
physical host TU + exact compiler context
  -> immutable authenticated closed-region evidence
  -> reconstruct and pair-verify the complete semantic MLIR witness
  -> compile-time traversal of the admitted frontend-neutral graph
  -> ordinary C++ statements and shape-dependent branches
  -> Clang object + linked production host adapter
  -> actual synchronous host execution
```

`emitClosedHostV1` accepts only an issued real-host admission object. A hermetic
inspection seal alone is insufficient; no raw `Program`, imported MLIR, JSON,
certificate string or caller-selected candidate function can enter this API.
The existing inspection witness is neither rewritten nor granted general
execution authority. Its exact paired graph directs a separate bounded adapter
mapping. The ordinary compiler, linker and runtime remain trusted components;
this boundary is not an object-file signature or hostile native-code sandbox.

The output contains no runtime AST/IR interpreter. Pure helper expansion has
already been authenticated by Clang admission. The emitter generates direct
adapter calls and ordinary unsigned comparisons from the semantic graph.

| Semantic requirement | Generated realization |
| --- | --- |
| Resource versus value identity | By-value view bindings versus opaque immutable result handles. |
| Read at the actual resource frontier | A snapshot at that statement, never an entry snapshot of all inputs. |
| Shapes/layout/access/capacity | Requested dimensions checked at each read; GEMM and publication guards remain lazy and ordered. |
| Numerical permission | Each GEMM keeps its explicit strict/reassociation-permitted profile and operation boundary. |
| Publication | Private host adapter's strong normal-return copy contract, no whole-region rollback. |
| Observation | Owning immutable snapshot in the invocation's observation records. |
| Conditional effects | Only the selected shape arm executes; unsigned 64-bit comparison is not narrowed to signed index or host `size_t`. |
| Failure trace | Return immediately after the first failing adapter call; earlier completed effects remain. |
| Completion | Separate final adapter completion frontier. |

Resource metadata is copied by ordinary by-value parameter binding; individual
resource validity is not preflighted at entry. An unused or untaken-arm resource
cannot fail an earlier source-required publication. Binding valid parameter
objects and the host adapter's lifetime/race/allocator preconditions remain
caller responsibilities, not pointer-authentication claims.

The private entry takes a pristine Session. Reuse is rejected before accessing
resources, and an already failed Session retains its first failure. This is a
test-oriented internal invocation shape, not an owner-facing API decision.

Frontier IDs are assigned at compilation in structural preorder. Gaps for
untaken branches are intentional, not missing events. A source/helper-site table
associates IDs with their semantic operations. The complete paired semantic
fingerprint names the emitted function: source-file and host-context hashes
alone are insufficient because one TU can define multiple regions. Fingerprints
detect substitutions and distinguish artifacts; they are not signatures.

## Falsification and validation

An independent source-to-executable harness found three prototype defects:

- Two different selected functions in the same TU collided when the symbol
  used only file/context hashes. It now binds the complete paired semantic graph.
- Real host identities have a `sha256:` prefix. The emitter now validates that
  established form and explicitly normalizes its digest field.
- An ordinarily moved-from admission seal had an empty payload. Query/pairing
  now reject it and payload accessors throw instead of dereferencing null. The
  moved-to seal remains valid.

All are retained in `compiler/tests/closed_source/closed_source_execution_test.cpp`.
The harness performs **219 admission/emitter checks and 37 actual child-process
execution assertions**, zero failures. It compiles emitted C++ to an ordinary
relocatable object and links it with the production adapter, not the injection
variant. Release passed in 6.25 seconds; the corresponding ASan/UBSan/LSan scope
passed in 8.08 seconds with flags propagated to both separately compiled child
sources. Affected Release regression scope passed **5/5** in 20.60 seconds.
These are implementation-lane results. The integrated checkpoint is recorded below.

PR [#42](https://github.com/onlyxItachi/MatcoreDSL/pull/42) merged normally at
`38e0c9368c65f399b0d8d6c9c39469e2836007c9` on 2026-09-06T19:59:30Z.
Its parents are `3756f681126726fad08b90d7fe9c1c48106575ea` and reviewed head
`38df446a8a9bcdc0c934ab943d2771066aa5a026`; the merged tree is
`c99221aad6f252f8a54108abe788f74771e5fa66`, matching the pre-merge tree calculation.
At the reviewed head, clean Release passed **91/91** (108.14 seconds), the exact
affected Debug ASan/UBSan scope passed **40/40** (33.05 seconds), and hosted checks
passed **19/19**. Independent review covered both the source implementation and
the final union-only CMake/sanitizer integration. See the
[exact-head merge gate](https://github.com/onlyxItachi/MatcoreDSL/pull/42#issuecomment-5561790804).

Cases include lhs/rhs-carried rectangular GEMMs; saved values versus late aliased
reads; overlapping/reused destinations; actual observations; second shape
failure after successful publication; dead-result checks; untaken invalid
resources; high-bit unsigned shape comparisons; zero geometries; reusable
helpers; non-FMA discrimination; caller FP restoration; Session reuse; and
historical/source/context authority substitutions. See the
[independent review](reviews/CLOSED_SOURCE_EXECUTION_INDEPENDENT_V1.md).

## Remaining boundary

Connect the statically emitted operations to the separately verified generated
CPU primitive through a compile-trusted numerical candidate registry. Preserve
guard order, private output, full FP restoration and empty-result short circuits.
Then establish a coherent user-facing source signature and installed compilation
route. The present internal Session argument is not that public interface.

No generated mathematical-kernel connection, arbitrary optimization equivalence,
public syntax, installed consumer, zero-copy, performance, provider parity,
GPU/NPU execution or asynchronous storage semantics is claimed here. Legacy
mutating GEMM, runtime/provider execution and its API remain unchanged.
