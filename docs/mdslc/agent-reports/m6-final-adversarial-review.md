# Milestone 6 final adversarial review

Date: 2026-07-26

Reviewed integration head: `4b9b3f4`

Base: `origin/main` at `951239f1a63905bfb9e2f781fc140f798ef39704`

Scope: independent review of the complete Milestone 6 branch diff, the frozen
schema-v6 evidence bundles, the sanitized summary, and the final handbook.
This reviewer did not implement production changes.

## Verdict

Accepted for the bounded MDSLC Milestone 6 CPU Performance Deep Audit.

There is no unresolved high- or medium-severity finding. The retained
limitations are evidence boundaries, not silently accepted parity claims:
hardware counters were unavailable, multi-thread OpenBLAS placement and active
concurrency were not authenticated, diagnostic/prepack/regret suites are
forward-only, and provider initialization is outside one-shot timing.

## Evidence authentication

The final review authenticated these external, untracked bundles:

| Bundle | Manifest SHA-256 | Plan SHA-256 | Cases | Passed | Expected rejections | Skips |
|---|---|---|---:|---:|---:|---:|
| stable-forward | `b3f872bd0085b15a8cd0cfcc7663af2a41f445355a3e3237c979dc52618362c0` | `c435d53b84856f4b2d0e52957a31a30c58964503550731e85cc0f2112ae3a64a` | 711 | 583 | 128 | 58 |
| stable-reverse | `4939c0c77586e4115dfe5c1aab1ff044d716e9a5d060c9f2ef52f265634df7f8` | `73bfb874fd94561f2b8581c39246f96ed5e723639424dddb33129e13637f3700` | 539 | 429 | 110 | 49 |

Both declare:

- source commit
  `509ef2b775e501783dfa7f2c4aa21e91f513bd6a`;
- benchmark binary SHA-256
  `a5a07cf06b6274aeba50a66c20713847f2d65a28ab28021e6d27e64a941c31f5`;
- runner SHA-256
  `be1db49ce5e82d34fc8b455d86c2fe2ad46ea5363a71b0af43b31104f1fd010d`;
- schema version 6, warmup 2, iterations 7, timer floor 1000
  microseconds, memory bound 2048 MiB, seed
  `5566823262476977457`, and threads 1/2/4/12.

Before the reviewer ran an incremental focused build, the external benchmark
path named by the manifest existed and matched the declared binary SHA-256.
That build later refreshed the development-build binary at the same path. The
raw bundles and manifests were not modified. This distinction is recorded so a
post-build hash of that mutable development path is not misrepresented as the
historical evidence binary.

Running the final summarizer over both bundles authenticated 583 forward raw
cells and reproduced
`docs/performance/cpu/cpu-performance-deep-audit-v1.md` byte-for-byte at
SHA-256
`144075d9527eb20a04eb033e23f869b1ea4f3554ae9b30241f7f7714abfbb426`.

## Adversarial findings and resolutions

### Resolved high: selective-shape omission

An initial adversarial bundle removed all cases and skips for the weak declared
shape `4096x64x4096`, recomputed both manifest plan digests, and was previously
accepted. The final summarizer now reconstructs the frozen matrix from the
hash-authenticated runner and rejects the bundle with exit status 2:

`manifest cases differ from the independently reconstructed frozen matrix`

### Resolved high: measurement-configuration tampering

An initial adversarial raw cell changed alignment to 4 bytes, maximum memory to
4096 bytes, and timer floor to 1 ns, then recomputed its raw hash. It was
previously accepted. The final summarizer authenticates every relevant
configuration field and rejects the first mismatch with exit status 2:

`raw configuration mismatch for alignment_bytes: expected 64, found 4`

### Resolved high: arbitrary expected rejection

An initial manifest recast return code 139 and a segmentation-fault diagnostic
as an expected rejection. It was previously accepted. The final summarizer
derives the only permitted rejection from variant, shape, threads, and mode,
then requires exact return code, category, stderr, and absence of raw output.
The attack now rejects with exit status 2.

### Resolved high: forced-variant fallback

A fresh reproduction changed a forced
`cpu.native-packed.avx2-fma.f32.v1` raw result to report
`cpu.external.openblas.f32.v1`, then recomputed the raw hash. The final
summarizer rejects it with exit status 2:

`forced raw variant silently selected a different implementation`

### Resolved medium: stale handbook and threading wording

The handbook previously described the final collection as future work. It now
records the exact schema-v6 identities, counts, bounded results, and
limitations. The sanitized report now says “configured OpenBLAS threads” and
explicitly states that active provider concurrency was not sampled.

## Fairness and claim review

- The single-thread native/OpenBLAS table compares requested and actual one
  thread with identical placement metadata.
- Complete-hot and one-shot values pair forward and reverse process orders;
  ranges remain visible.
- Allocation-inclusive one-shot includes replan, output/workspace allocation,
  packing, compute, and synchronization. It excludes process startup, input
  initialization, execution-context construction, and provider
  initialization, as documented.
- Prepacked-B preparation is measured once outside steady state and included
  in the separately reported amortized total.
- Compute-only results are explicitly diagnostic and are not compared with
  complete BLAS calls.
- Multi-thread OpenBLAS is unbound and diagnostic-only. The report does not use
  it for parity or comparable planner regret.
- Hardware counters were blocked by `kernel.perf_event_paranoid=4`; documents
  clearly separate static/derived reasoning from physical measurements.
- The complete 35-shape single-thread matrix is published, including the weak
  tall-skinny region. Scalar runtime-bound omissions are predeclared and
  counted rather than selected after measurement.
- OpenBLAS, BLIS, and LLVM archaeology pins upstream revisions and discusses
  architecture. No third-party implementation code was copied.
- No automatic planner behavior, runtime ABI, or public API changed in this
  investigation milestone.

## Repository and package review

The diff adds benchmark instrumentation, schemas, authenticated collection and
summary tools, package-safety tests, sanitized documentation, and the handbook.
Raw JSON, profiler output, build artifacts, and generated bundles remain
untracked. Repository hygiene passed.

The source-inaccessible installed-consumer test uses a disposable copied tree
with deletion guards; its dedicated adversarial safety test passed. No
build-tree-only path was accepted as package evidence.

## Commands and results

Final evidence reproduction:

```text
python3 compiler/tests/performance/summarize_cpu_deep_audit.py \
  --forward-manifest <external-forward>/manifest.json \
  --reverse-manifest <external-reverse>/manifest.json \
  --markdown-out <external-temp>/authenticated.md
```

Result: 583 authenticated raw cells; generated and committed report hashes were
identical.

Focused final tests:

```text
ctest --test-dir /home/hamza-usta/.tmp/mdslc-m6-dev-build \
  -R '^(package\.installed_source_inaccessible_safety|benchmark\.cpu\.(contract|cli_json|provenance_incremental|deep_audit_runner_contract|deep_audit_summary_contract))$' \
  --output-on-failure -j1
```

Result: 6/6 passed.

Repository checks:

```text
bash tests/check_repository_hygiene.sh
git diff --check
```

Result: both passed.

The selective-shape, configuration, arbitrary-rejection, and forced-fallback
reproductions each returned exit status 2 with their expected actionable
diagnostic.

## Residual limitations

These are low-severity or explicit measurement boundaries:

1. `perf_event_paranoid=4` prevented physical counter collection.
2. Multi-thread OpenBLAS authenticates configured team size, not observed
   concurrent workers, and has unequal affinity; those cells are
   diagnostic-only.
3. Cold-cache, compute-only, prepacked-B, and planner-regret evidence has no
   reverse-order replicate.
4. Two process orders do not establish multi-session statistical confidence.
5. The evidence binary itself is identified by SHA-256 but is not committed,
   appropriately preserving repository artifact hygiene.

These limitations must remain visible in Milestone 7. They do not invalidate
the root-cause audit or justify a BLAS-parity claim.

## Final conclusion

Milestone 6 has complete, reproducible, fail-closed evidence for its bounded
audit scope. It answers the required architectural questions without
overstating physical counters, provider placement, or performance parity and
provides a defensible experiment queue for Milestone 7.

Final review: **accepted; no unresolved high- or medium-severity finding**.
