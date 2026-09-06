# Private closed-region CPU candidate registry

This builds on the [immutable host adapter](../closed_host/README.md) and the
[strict generated primitive](../../../docs/mdslc/agent-reports/generated-strict-cpu-candidate-v1.md).
It does not authenticate a source program, read serialized execution authority,
install a public API, or alter the compatibility runtime/planner.

`Session(Options{...})` selects from a compile-trusted enum. A default Session
retains the strict native candidate. Explicit `automatic` chooses the linked
strict generated leaf, or strict native when absent. There is no shape/cost
threshold, environment override, plugin callback, or performance policy.

| Request | Implementation | Numerical legality |
| --- | --- | --- |
| `native_strict` | new increasing-K separate multiply/add native loop | strict or reassociate |
| `generated_strict` | fixed verified MLIR 21.1.8 scalar-loop AOT leaf | strict or reassociate |
| `existing_native` | unchanged runtime C ABI, forced `cpu.reference.f32.v1` | reassociate only |
| `authenticated_openblas` | unchanged authenticated runtime/provider adapter, forced one thread | reassociate only, both provider gates must pass |

The legacy reference is not claimed to be strict: its existing compile profile
permits FMA. The registry does not enable packed, vectorized, parallel, or newly
generated arbitrary candidates. No reassociation crosses a GEMM value boundary.

All input Values already own immutable snapshots; candidate output is new private
storage, isolated from external resources and inputs. A value handle is issued
only after successful candidate completion and FP-control validation. The entire
caller MXCSR/x87 state is restored on success/failure. Provider partial-output
failure therefore discards only private storage; earlier publications survive.
This is normal-return behavior, not recovery from faults/crashes or a sandbox
against a corrupt compiler, linked runtime/provider, allocator, or interposer.

Forced availability/numerical requirements apply even to empty math. A first
provider request may run authentication probes even when its requested GEMM is
empty; reports distinguish probe invocation from requested-computation invocation.
After those checks, M/N zero never enters the generated leaf (whose N-zero outer
loop can be very large). K zero with nonempty output produces positive zero
locally. Reports identify these semantic realizations, never pretend BLAS or the
generated leaf executed that GEMM. Old positive-shape/BLAS quick returns are not
used as new-language semantics.

The additional once-only closed-provider gate has five stack-fixture calls:
4x4 K2 expression membership and four K1 signed-zero/subnormal/nonfinite cases.
It reuses the existing C ABI, provider identity gate, local one-thread control,
and execution report. It is not universal proof. The trusted-provider contract
is IEEE f32 product/reduction with allowed FMA/reassociation, no finite-only or
signed-zero elision; new versions/cores still require release execution evidence.

Tests separately exercise actual production linkage, native-only unavailability,
all original adapter attacks, exact strict arithmetic, K1/K2 permitted-expression
membership, blocked/tail special values, lhs/rhs carries, immutable snapshots,
late reads, publication prefixes, FP restoration and immutable report snapshots.
The hostile-runtime executable is test-only: it substitutes the C ABI at ordinary
link time to falsify numeric/variant/thread/control reports and partial failure.
The production archive has no injection setter. Instrumented builds cover the
host and generated leaf; externally installed OpenBLAS internals are not ASan/
UBSan-instrumented by these tests. Existing generated-leaf ASan OOB controls remain
necessary; raw LLVM IR cannot obtain UBSan source checks merely from linker flags.
