# Final independent review: #115 GPU tool version gates

**Verdict: accept the bounded correction.** Reviewed source
`ce760b7fc51be65cd7b0b47adbd7a24991cf4f3e` against
`1c1221e89272eb2ad6ff058b8e26de4653f3a93d` on 2026-11-26.
I did not author either #115 implementation or the preceding
[request-changes review](issue115-independent-review-62-v1.md)
(`8f706d640fcaebfbfdf3d62af65d0f5e3c5c0f90`). I read the complete
base-to-candidate diff, repository policy, [issue #115](https://github.com/onlyxItachi/MatcoreDSL/issues/115),
both implementation reports and the original review. No production/test
source was modified in this review.

## What changed at the decision point

The common llc gate and backend-specific ptxas and LLD gates in
`compiler/lib/regions/GpuCandidates.cmake` compare a complete token from the
supported **leading** header. A matching quotation after an unsupported header
cannot supply the token. The NVIDIA check also verifies the release field
against the token on the same record. An attached suffix is part of the token,
not ignorable metadata. The header can still have ordinary trailing build
metadata or an llc target list; those lines do not promote a missing token.

I traced the production include from `compiler/CMakeLists.txt` to the regions
subdirectory and the qualified tool variables through both the single and
fused image commands. All three probes run with fatal process-error handling;
the exact-version checks precede issuance. No diff to tool selection, pinned
versions, disabled/default paths, GPU flags, driver/header checks, Clang
parsing or runtime selection was needed.

## Executed evidence

Linux x86_64, CMake 4.3.2. From the reviewed worktree:

```sh
cmake -DMDSLC_SOURCE_DIR="$PWD/compiler" -P compiler/tests/cmake/toolchain_version_contract.cmake
git diff --check 1c1221e89272eb2ad6ff058b8e26de4653f3a93d..ce760b7fc51be65cd7b0b47adbd7a24991cf4f3e
```

Both passed: seven Clang controls ran followed by **90/90** GPU parser and
production-gate controls. My separate matrix ran
`python3 -B /tmp/mdslc-115-review-cr.C22Vehiz/review_matrix.py` from the
worktree. Each row executes the actual production file through its child
fixture, from `compiler/lib/regions`, with the following CMake arguments:

```sh
cmake "-DMDSLC_SOURCE_DIR=$src" -DGPU_VERSION_GATE_CHILD=ON \
  "-DGPU_VERSION_TOOL=$tool" "-DGPU_VERSION_BANNER=$banner" \
  -P "$src/tests/cmake/gpu_toolchain_version_contract.cmake"
```

**24/24 expected decisions (8 admissions, 16 refusals)**. Admissions required
exit 0 and the selected production-gate marker. Refusals required exit 1, no
admission marker, the selected gate diagnostic, and the expected token or
reason. These are additional probes, not 24 disjoint new code paths.

| Inputs | Observed |
| --- | --- |
| Four original review witnesses: llc/LLD dependency quotes, ptxas conflicting release, malformed ptxas followed by nvcc | 4 refused; conflicting diagnostic names `12.8` |
| Real llc/LLD/ptxas banners; their CRLF forms; upstream llc preamble with indented Debian header; LLD followed by a non-record dependency note | 8 admitted |
| For each tool: attached `-vendor`, internal lone CR, unrecognized preamble followed by a complete genuine banner | 9 refused |
| ptxas `013.3` release with the exact token; intervening note before the CUDA record; second unterminated llc record with no token | 3 refused |

To avoid relying on `-D` preserving input bytes, I also used a separate CMake
script to call each **actual installed tool** with
`execute_process(COMMAND "${REVIEW_EXE}" --version OUTPUT_VARIABLE GPU_VERSION_BANNER COMMAND_ERROR_IS_FATAL ANY)`
before including the same child fixture. From `compiler/lib/regions`, with
`src` the absolute compiler source path:

```sh
harness=/tmp/mdslc-115-review-cr.C22Vehiz/actual-or-cr.cmake
check() {
  cmake "-DMDSLC_SOURCE_DIR=$src" -DGPU_VERSION_GATE_CHILD=ON \
    "-DGPU_VERSION_TOOL=$1" "$2" -P "$harness"
}
check llc -DREVIEW_EXE=/usr/lib/llvm-21/bin/llc
check lld -DREVIEW_EXE=/usr/lib/llvm-21/bin/ld.lld
check ptxas -DREVIEW_EXE=/usr/local/cuda/bin/ptxas
check lld -DREVIEW_LOAD=stdout
check lld -DREVIEW_LOAD=literal
```

All three untrimmed real outputs were admitted: llc **1766 bytes**, LLD
**48**, ptxas **216**, all ending in LF. The last two commands intentionally
failed at the LLD gate: both received the same 48-byte banner ending in a
lone CR (`0x0d`) instead of LF. An initial negative using `-D` admitted what
looked like this same input. The byte check showed that CMake's argument
handling dropped that trailing CR (47 bytes, final `0x29`); a literal or
captured stdout retains it and the production parser rejects it. This was a
fixture-transport distinction, not an admission of raw lone CR. The two
temporary reviewer scripts are not build or test dependencies.

## Limits

The fixture uses the real gate file but mocks discovery and SDK/target setup,
stopping before image creation. The installed programs were invoked only for
`--version`. This is evidence for the specified banner grammar and tuple, not
binary authentication or general compatibility with other vendor formats.
I did not configure/build a product, install anything, run a GPU, generate an
image, execute the broader suite, test other CMake versions, or run hosted CI.
No comparative model-quality conclusion follows from this review.
