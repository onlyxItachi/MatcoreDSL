# Experimental frontend and owning Result: integration checkpoint

## Provenance and scope

This integration starts from the independently reviewed frontend/Result
`6dbd4b4e0853748b44312aa3f332af2b6303ac3d`, composed with the existing candidate
and source-execution foundation at `de0911eebc788b2d05266efadb9d3902ebc2b7c7`.
History is retained through normal merges, without rebasing or replacing the
earlier implementation/review checkpoints.

- Canonical candidate coexistence: PR #45,
  `2201ef9154655406c6241bd6fd2d49889f1cbdba`.
- Canonical connected generated-source proof: PR #48,
  `322af8d43799304cc289748a5b0c803cf3a0f697`; pre-merge source branch head
  `63f3bccc7ad38d4fc7092c4c86b8ccd799e80c08`.
- Normal integration merges: `58be048` and
  `b748e665be74ed3e688431506a74b47bba281c96`.
- Independent source/ownership report originally committed as
  `8233a7945a3e1d0d28ff4393e79142ed513d6341`, retained here as `6d73cb9`.
- Focused installation/CI integration: `93cc3fc6dd12a77c553b761d0ea3c8950fbe9b39`.

The branch is `mdslc/experimental-region-admission-integration-v1`. All final
local tests below ran with a clean tree at `b748e665...`, before this record was
added. Neither a hosted result nor a canonical merge of this branch is implied
by this local report; the PR and later operator checkpoint own those facts.

## What composes

The [frontend contract](../EXPERIMENTAL_REGION_FRONTEND_V1.md) admits an
experimental named C++ function returning an owning checked Result, with the
existing closed mathematical grammar and an exact terminal `return complete()`.
It adds the canonical source declaration/body/entry witnesses needed by a later
same-signature host-codegen consumer. Source-only Values remain trivial
intrinsics, not runtime proxies or allocations. Owning observations use an opaque
block, with allocation-free one-way Result retirement and out-of-line ownership.

The [independent review](experimental-region-source-auth-independent-v1.md)
retains the real rejected alternatives and pre-fix counterexamples: public STL
layout drift, ambient packing/member attributes, out-of-line ownership method
definitions and hostile completion of private forward declarations. Admission
and host-capture implementation bytes are unchanged from that review. The
runtime composition preserves the already-canonical candidate registry and
shared provider policy ownership; no candidate algorithm or numerical policy
changes in this integration.

The explicit legacy-only installation list prevents the old recursive header
installation from accidentally publishing `region.h` and its detail header
without a production runtime/driver. A fresh-prefix package test checks both
that absence and the presence of `mdsl.h`/`runtime_c.h`. This is not an installed
experimental frontend/API claim. The later opt-in build/install promotion must
replace this temporary absence contract explicitly.

The hosted sanitizer selection adds exactly seven new frontend/Result tests to
the canonical 51-test scope. CTest discovery independently counted **58**, with
all existing selections retained. The fresh-prefix installation test belongs to
the ordinary package matrix rather than the sanitizer-only selection.

## Exact local validation

Linux x86-64; Clang/LLVM/MLIR **21.1.8**. Builds used `/usr/bin/clang-21` and
`/usr/bin/clang++-21`, LLVM/Clang CMake directories under `/usr/lib/llvm-21`, and
the pinned MLIR package at
`/home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/lib/cmake/mlir`.
All configurations enabled Matcore MLIR and BUILD_TESTING. Builds used Ninja
with `-j2`; CTest ran serially.

| Build / test scope | Exact outcome |
| --- | --- |
| Release, OpenBLAS ON and required (actual 0.3.32), focused frontend/Result/adapter scope | 15/15 PASS, 25.18 s |
| Release same configuration, complete CTest registry | **111/111 PASS**, 272.25 s |
| Debug, OpenBLAS OFF, focused same scope before the canonical-only merge | 15/15 PASS, 29.46 s |
| Debug same configuration at final merge, complete CTest registry | **109/109 PASS**, 372.18 s |
| Debug ASan+UBSan, OpenBLAS OFF, semantic pipeline default `matcore-mlir`, exact hosted scope | **58/58 PASS**, 54.75 s |
| `bash tests/check_repository_hygiene.sh`; `git diff --check` | PASS |

Both full suites include installed consumers, C17 ABI, actual source-inaccessible
installation, frontend contracts, runtime/planner and provider-enabled/disabled
surfaces, the existing generated source proof and generated-kernel sanitizer
negative control. The Linux COFF driver-contract test is not Windows execution.
No new Windows execution or performance evidence is inferred from these runs.

Reproduction commands after configuring the named builds:

```sh
cmake --build build-frontend-release -- -j2
ctest --test-dir build-frontend-release --output-on-failure -j1
cmake --build build-frontend-debug -- -j2
ctest --test-dir build-frontend-debug --output-on-failure -j1
cmake --build build-frontend-asan -- -j2
```

ASan configuration uses `-O1 -g -fsanitize=address,undefined
-fno-omit-frame-pointer` for C/C++, `-fsanitize=address,undefined` for executable
and shared linking, and the exact regular expression recorded in
`.github/workflows/mdslc-native.yml`. Test environment:
`DEBUGINFOD_URLS=`;
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`;
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

## Independent integration review and remaining boundary

The root integration reviewer inspected the full frontend/runtime composition,
confirmed production files byte-identical to the reviewed `de0911e...` union,
verified the exact seven-test sanitizer addition and explicit installation
isolation, and accepted the bounded integration pending local/hosted gates.
That is an independent engineering review, not a fabricated GitHub approval.

The next dependency is a supported private runtime ABI and explicit opt-in
production packaging, followed by the separately reviewed authenticated
same-signature source-to-executable driver. None is smuggled into this PR.
The existing executable CPU/provider route and source-proof execution remain
unchanged. This checkpoint does not freeze public syntax/ABI, enable arbitrary
IR execution, prove zero-copy/performance/BLAS parity, or add accelerator support.

Composed production SHA-256 identities:

| File | SHA-256 |
| --- | --- |
| `ClosedRegionAdmission.cpp` | `6dbae447249c0ce64f9389fb5ccd674df416ab67ada42679bee66b80a31dfc81` |
| `ClosedRegionHostAdmission.cpp` | `dbffc7c135839d23d53c271c2b1d30603f7a9369c19d210e4888bf6197923a97` |
| public `region.h` | `2717d27b8787f976e2b5b4af2645b38d1409ce0c91181f4e7413e2a00265b45e` |
| detail `region_storage.h` | `f228886210ffa74c9922a3b75db13caf818cc42a0a455ca22eec432df8b38e32` |
| `closed_host_v1.cpp` | `b895c7f5c83fdba105c34df6ce5b2085a1fc922a25fbe847340ad7cb968a3fb0` |
| `closed_host_v1.h` | `3a5b06766b669b9a690c376dbe32223f56916b2d16ba12a8e13437c005b93581` |
