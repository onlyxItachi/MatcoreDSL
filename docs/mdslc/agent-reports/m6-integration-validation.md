# Milestone 6 integration validation

Date: 2026-07-26

Branch: `mdslc/cpu-performance-deep-audit`

Validated head before this report:
`ed9546c4fbe2af4013600ff7c9d41b357a7aa239`

Rewritten `origin/main` base:
`951239f1bee5541a4cf5ad72fab2192de07cf89d`

## Verdict

Milestone 6 is accepted locally for the bounded CPU Performance Deep Audit
scope. The branch adds authenticated benchmark instrumentation, reproducible
audit/summary tools, package-isolation regressions, sanitized evidence, and
engineering analysis. It does not change automatic planner behavior, the
runtime ABI, the public API, or production kernel selection.

The independent final review reports no unresolved high- or medium-severity
finding. This is an audit result, not a native-BLAS-parity result.

## Authenticated performance evidence

The final review independently authenticated two external, untracked schema-v6
bundles:

| Order | Manifest SHA-256 | Cases | Passed reports | Expected rejections | Predeclared skips |
|---|---|---:|---:|---:|---:|
| forward | `b3f872bd0085b15a8cd0cfcc7663af2a41f445355a3e3237c979dc52618362c0` | 711 | 583 | 128 | 58 |
| reverse | `4939c0c77586e4115dfe5c1aab1ff044d716e9a5d060c9f2ef52f265634df7f8` | 539 | 429 | 110 | 49 |

Both bundles identify:

- source commit
  `509ef2b775e501783dfa7f2c4aa21e91f513bd6a`;
- benchmark binary SHA-256
  `a5a07cf06b6274aeba50a66c20713847f2d65a28ab28021e6d27e64a941c31f5`;
- audit runner SHA-256
  `be1db49ce5e82d34fc8b455d86c2fe2ad46ea5363a71b0af43b31104f1fd010d`;
- two warmups, seven measured iterations, 64-byte alignment, a 2048 MiB
  allocation bound, and requested thread counts 1/2/4/12.

The final summarizer authenticated every raw cell, reconstructed the frozen
case matrix from the hash-matched runner, and reproduced
`docs/performance/cpu/cpu-performance-deep-audit-v1.md` byte-for-byte. Its
synthetic adversarial tests reject selective shape deletion, configuration
tampering, arbitrary crash-like “expected rejection” records, command changes,
and forced-variant fallback.

## Build and test matrix

The final validation root was:

```text
/home/hamza-usta/.tmp/mdslc-m6-final-gate.ERDO67
```

The absolute path is operational evidence only; it is not embedded in the
installed package or public documentation.

| Configuration | Result |
|---|---|
| Fresh Release, Clang 21.1.8, OpenBLAS required | 48/48 CTest tests completed successfully |
| Fresh Debug, Clang 21.1.8, OpenBLAS required | 48/48 CTest tests completed successfully |
| Fresh Release, OpenBLAS disabled | 48/48 CTest tests completed successfully |
| ASan + UBSan supported scope | 17/17 passed |
| TSan shared-state scope | 4/4 passed |
| Legacy frontend contract | 14/14 checks passed |

OpenBLAS-disabled forced planning for
`cpu.external.openblas.f32.v1` failed closed with `selected=none` and the
actionable reason `OpenBLAS CBLAS adapter is not linked`.

The first sanitizer command accidentally expanded an empty test regex and ran
unsupported installed-link tests that do not link sanitizer runtimes. It was
discarded as a command-selection error. The corrected, declared
ASan/UBSan-supported 17-test scope passed with no sanitizer report.

An initial OpenBLAS-disabled run overlapped an in-progress documentation edit.
The provenance guard correctly rejected the dirty source. After the branch was
clean and provenance was refreshed, the complete OpenBLAS-disabled 48-test
suite passed. Neither invalid run is counted as acceptance evidence.

## Package and installed-tree proof

The package matrix includes a source-inaccessible installed-consumer test. It
copies the exact clean `HEAD`, builds and installs the standalone compiler,
relocates the prefix through a whitespace/Unicode path, scans for producer
paths, removes the disposable producer source and build trees, and then builds
and runs an external `find_package(MatcoreDSL REQUIRED)` consumer using only
the relocated prefix.

The paired destructive-operation safety test rejects mismatched roots,
symlinked roots, missing sentinels, `..` path spelling, and any test root that
could contain the protected checkout. The focused package tests passed 2/2;
both are also included in the successful 48-test Release, Debug, and
OpenBLAS-disabled suites. The final installed-prefix scan found no source-
checkout or build-tree absolute-path leak.

## Native artifact proof

The existing native pipeline was exercised from the final Release build:

```text
gemm_v0.mdsl
  -> native Clang frontend and verified Matcore IR
  -> generated host/stub/backend sources and objects
  -> gemm_v0.o
  -> ordinary clang++ link
  -> gemm_v0
```

`file` and `readelf` identify `gemm_v0.o` as an ELF64 x86-64 relocatable
object. `nm -C` shows `main`, the stable generated call-site/backend symbols,
and an undefined `matcore_runtime_gemm_f32_v0` C runtime boundary. The final
binary is an ordinary ELF64 PIE executable, dynamically resolves the versioned
Matcore runtime and OpenBLAS provider, and prints:

```text
host-before
MDSLC CPU GEMM PASS
```

## Repository and review checks

Commands used for the final repository checks:

```text
git diff --check
bash tests/check_repository_hygiene.sh
python3 tests/test_frontend_contract.py
```

All passed. Raw benchmark JSON, profiler output, build products, and evidence
bundles remain outside Git.

The final reviewer reproduced the authenticated report, exercised six focused
benchmark/package contract tests, reran repository hygiene and diff checks, and
verified that all confirmed evidence-authentication defects fail closed. The
accepted review is
`docs/mdslc/agent-reports/m6-final-adversarial-review.md`.

## Evidence boundaries

- `kernel.perf_event_paranoid=4` blocked physical performance counters.
  No measured IPC, cache/TLB miss, stalled-cycle, DRAM-traffic, or workload
  frequency claim is made.
- Multi-thread OpenBLAS records configured provider threads but not sampled
  active concurrency, and its placement is not comparable to the bound native
  workers. Those results are diagnostic-only.
- Cold-cache, compute-only, prepacked-B, and planner-regret suites are
  forward-order only.
- The two opposite complete/one-shot process orders reveal order sensitivity
  but are not multi-session confidence intervals.
- Provider initialization and process startup are outside the one-shot
  interval.
- Milestone 6 does not establish native BLAS parity and does not authorize a
  production planner or kernel change by itself.

## Publication state

The local acceptance gate is complete. At this checkpoint, normal pull-request
publication, hosted Linux/Windows checks, normal merge, the immutable
`mdslc-cpu-performance-audit-v1` tag, and tracker closure remain pending.
