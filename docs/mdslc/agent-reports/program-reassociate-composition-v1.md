# Independent multi-source / generated-reassociate composition fixture

2026-09-09. **PREPARED, NOT EXECUTED.** This report deliberately precedes the
composed production driver; it is not evidence that either uncombined PR supports
the other one's functionality.

Base `ed8f7fed1d4a1969328e4f0c9626f0d98151169e`; worktree
`/home/hamza-usta/MatcoreDSL-wt-program-reassociate-test-v1`, branch
`test/program-reassociate-composition-v1`. Scope is only the new
`compiler/tests/program_reassociate/` directory and this report. No production,
existing fixture, CTest/CI, runtime, issuer, build or installed artifact changes.

The shared-parser merge hazard is concrete: PR #67 extracts the candidate policy
list while PR #66 introduces a new policy. Merely compiling the separate branches
does not prove that the final `--program` dispatcher retains the new spelling or
that all selected regions execute with the requested implementation contract.

The independently authored three-TU program tests both policy parsing and real
source semantics. `first.mdsl` contains a permissive full 4x8/K2 GEMM. The host
oracle distinguishes FMA result bits `0x337ffffe` from strict positive zero.
`second.mdsl` contains a strict rectangular 4x8-by-8x3 GEMM. Between calls, normal
host code retains the first owning observation, destroys its Result and mutates
the first published array. Forced generated-reassociate must preserve this
boundary and fail only at the second GEMM's existing frontier; generated-strict
must instead read the intervening mutation and complete both calls successfully.

Exact planned failure oracle: second source GEMM `second.mdsl:6:13`,
`candidate_incompatible`, failed frontier 3 / completed 2 / effect 0, zero second
publication/observation. The already completed first call had one publication,
one owning observation and completion/effect frontiers 6/5. Its owning contents
and the ordinary host's inter-call mutation survive later failure and both Result
lifetimes. Host trace must be 123. Public Result does not report Implementation.

The runner requires successful actual three-TU compilation for both policies,
the pre-cross-TU ownership validation marker, exact numerical/failure output,
empty positive diagnostics, unchanged pinned inputs/outputs, and direct private
DSO/canonical Runtime dependencies. Wrong-label executions must fail exactly with
the planned oracle-only diagnostics, never a signal or sanitizer report. Genuine
known hardware-unavailable exit 77 is SKIP; an absent parser policy is failure.

Static counting gives 152 generated-reassociate / 164 generated-strict assertions
and two negative controls, provisionally 143/155 failure assertions. These are
planned counts only until a frozen composed driver actually executes the fixture.
The integration owner will register CTest and reconcile CI inventories afterward.
No next storage optimization, performance, standalone ABI or installed-SDK claim
is included in this final composition gate.

Prepared source identities (SHA256): `api.h`
`6d0894cd4c24076bed82a5e5d6a3ec4d96a9f60021aac8ac7b9a41904afd8d8f`;
`first.mdsl` `fed84751ad3c09834e9fffa26ba962e9da9f2631729d1ccec010d1769589021d`;
`second.mdsl` `d6e7061543dfdadc459124cf571e1c167977237ffe774a086639c65e15c2b961`;
`host.cpp` `6f0e554620dfed29407b3604dddbb8363524f6f048760dd3f66d09171160797f`;
`run.py` `2ecc98f2cbe04c81729e55770e274417aab37ab5410c004d81522b00ff2bed9d`.
Whitespace validation passed. No C++ syntax compilation, source-driver execution,
hardware probe, or test runner execution has yet been performed for this fixture.
