# Issue #115: bind exact versions to the selected tool's header

Correction to `b73aeefb53ada96530931b24fb015ed46954cd2f` following the
[independent review of that revision](issue115-independent-review-62-v1.md),
review commit `8f706d640fcaebfbfdf3d62af65d0f5e3c5c0f90`. That reviewer
authored this correction; the review is not independent acceptance of this
new implementation.

## Accepted banner grammar

- After leading whitespace, llc requires an upstream/Ubuntu/Debian
  `LLVM version <token>` record. Only its known `LLVM (http://llvm.org/):`
  preamble can precede the record. LLD starts directly with its
  upstream/Ubuntu/Debian `LLD <token>` record.
- ptxas starts with the NVIDIA ptxas identity. Before the CUDA release record
  it may have the NVIDIA copyright line and the single-token `Built on`
  timestamp line, in that order. The version must come from this header,
  not from a later matching line after unknown text, a malformed record, or
  an embedded nvcc banner. A release such as `12.8` cannot qualify
  `V13.3.73`: the diagnostic includes both the token and the conflicting
  release. The token itself must still equal the required full version.
- Horizontal indentation and LF/CRLF are accepted; a lone carriage return
  is not a supported delimiter. Whitespace-separated trailing metadata and
  LLVM's target list may follow the authoritative header. A subsequent
  recognizable version-record marker is a reason to reject, never a source
  of a substitute token. Attached punctuation/suffixes remain part of the
  token.

This correction changes only the internal GPU parser, its test, and this
report. No change to Clang parsing, discovery, required versions, recipe
flags, SDK/DSO checks, platform restrictions or runtime/default selection.

## Local evidence (Linux x64, CMake 4.3.2, 2026-11-26)

From the candidate worktree:

```sh
cmake -DMDSLC_SOURCE_DIR="$PWD/compiler" -P compiler/tests/cmake/toolchain_version_contract.cmake
git diff --check
```

Before changing the parser, the new test stopped at `llc_quoted_dependency`:
expected an empty token, got `21.1.8` (exit 1). With the correction the seven
existing Clang controls and **90/90 GPU controls passed** (15 admissions,
75 refusals). The four review witnesses were also rerun directly with the
review's child-fixture commands: **all four were rejected by their selected
production gate** (exit 1); on the reviewed candidate all four were admitted.
The existing `Built with` negative fixture still refuses; the parser now
rejects that unsupported preamble without extracting a token from below it.

For an independent-of-fixture input, used Python `subprocess.check_output`
with `text=True` to read untrimmed `--version` stdout and passed it as
the `-DGPU_VERSION_BANNER=` argument to the same child command. All **3/3**
full banners passed: `/usr/lib/llvm-21/bin/llc` (1766 characters including
target listing), `/usr/lib/llvm-21/bin/ld.lld` (48), and
`/usr/local/cuda/bin/ptxas` (216 including copyright/build lines).
The previous and current Clang helper (first 14 lines) have identical SHA256;
the production call sites and shared single/fused recipe files have no delta
against the previously reviewed candidate.

The child exercises the real configure gate with mocked tool discovery, SDK
and target setup. No clean product configure/build, full suite, hosted CI,
native Windows run, installation, image generation, GPU execution, or new
tool/device qualification is claimed. Banners do not authenticate a tool
that deliberately misidentifies itself.
