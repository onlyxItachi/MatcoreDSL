# Independent multi-source / generated-reassociate composition fixture

2026-09-09. **PASS: independently executed on both frozen composed profiles.**
The execution supplement below confirms the originally provisional fixture on
the actual multi-source/generated-reassociate union. No uncombined-branch,
installed-SDK, performance or later hosted-CI result is inferred.

## Original preparation checkpoint (historical)

The following preparation record preceded the composed production driver; at
that time this was a prepared, unexecuted fixture rather than execution evidence.

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

## Independent execution on the frozen composed driver

Actual production tree:
`/home/hamza-usta/MatcoreDSL-wt-authenticated-multi-source-v1`, clean
`e56e5a648295f7b5ec60bff31f30bf37c031c854`. The fixture remained exactly
`c3346f83215873e49c7a6e3dbb80771abedc9f09`; its integrated source files were
verified byte-identical before execution. No fixture, runtime, compiler or build
changes were made during either run. The integration owner owns CTest registration
and the separately running full regression suites.

| Frozen artifact | Release SHA256 | ASan+UBSan SHA256 |
| --- | --- | --- |
| Actual `mdslc-region` | `23ca6986a99a1b0acdeb7a73bc77a6903c3a650c1664475f4bd2595dfe90a02c` | `07d3ce88800b45a0959787750cce36e160db681fc34ce11bc4890c2765d05661` |
| Private candidate DSO | `6a91807fd22391e6b459b12430daaf7b81735cd77445f4a64890feb3d30a9b58` | `ab400474ef92100b890535a37c7dbdec77c1b409a8d5aa69746f0f85ef8f50f0` |
| Canonical Runtime DSO | `df8222de671294b94865f8597c24f239051e44889ce71f01581ef7e97e45c014` | `e6f650fe38d05dc4f89a797b76960e3e8de49e749db60a4463a32e2a0b6f5fa2` |

Driver paths are respectively `build-program-release-native/bin/mdslc-region`
and `build-program-asan/bin/mdslc-region` under that production tree. The runner
received each exact expected driver digest. All six artifact hashes remained
unchanged after execution; the runner also checked every fixture and produced
executable before and after each subprocess.

| Actual execution in each profile | Exit | Assertions | Failures |
| --- | ---: | ---: | ---: |
| `generated-strict` with its correct oracle | 0 | 164 | 0 |
| `generated-reassociate` with its correct oracle | 0 | 152 | 0 |
| Strict executable, deliberately reassociate/refusal oracle | 1 | 152 | 143 expected |
| Reassociate executable, deliberately strict/success oracle | 1 | 164 | 155 expected |

Thus **both Release and ASan+UBSan passed 316 positive assertions and two negative
controls**, with host trace 123 on all four executions per profile. No hardware
skip occurred. All positive stdout identities/counts and empty stderr matched;
the deliberate failing executions contained exactly their expected oracle-only
diagnostic lines, with no signal/crash or sanitizer diagnostic accepted.

The measured values confirmed the full 4x8 FMA-vs-separate-rounding discriminator.
The generated-reassociate executable reached the ordinary host mutation after
first publication/owning observation and then returned `candidate_incompatible`
at the second TU's exact GEMM frontier/source location. Generated-strict instead
computed the rectangular product from the intervening host mutation. Both retained
the first immutable observation after both Result lifetimes and final resource
mutation, preserving input arrays and all destination canaries.

Both policies compiled through actual three-TU ownership validation and emitted
two cross-TU link markers before program publication. Each resulting ELF directly
needs `libmatcore_closed_candidates_isolated_v1.so` and `libmatcore_runtime.so.0`.
The Release package also links its configured OpenBLAS provider, whereas the
sanitizer package is provider-OFF; neither test requested provider execution.
The sanitized executable contains `__asan_init` and UBSan handler definitions.
This verifies the actual sanitizer profile, not independent proof of generated
load instrumentation; the candidate's fault-injection gates remain separate.

Raw evidence root: `/tmp/mdslc-program-reassociate-final.sKE0y9`.
`release/evidence.json` SHA256
`4cff9c68871bb2f523e873a70000277a45734f7a1a780fd1a518f0cb03fc8712`;
`asan/evidence.json` SHA256
`dbf99d0168e905267115f865f904db926506e898ce539c31e9508a1ee213cac7`.

| Produced executable | Release SHA256 | ASan+UBSan SHA256 |
| --- | --- | --- |
| generated-strict | `58d2873fe6c151decd0c30f759540a6e11b04d08cee9d05028be11d80795a6ce` | `78696075f0070d20b64295039e81ae3cc64e2342e40ece7cae37ff25a685b451` |
| generated-reassociate | `57c24043f74015eb60eee9fa220539e7798c6183ce784da560f0093af2937db1` | `f3756d25050f50b941df4096ab966ebc577f0b16a4a41621c51b8113cc88c4ed` |

Verdict: the shared policy parser and genuine multi-source route retain the new
per-GEMM candidate without whole-program numerical preflight, hidden strict
fallback, or cross-call rollback. This is bounded source/host-boundary evidence,
not public Implementation telemetry, cross-region optimization or a timing claim.
