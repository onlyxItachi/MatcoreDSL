# Issue #115: exact GPU tool version qualification

Date: 2026-11-12. Issue: <https://github.com/onlyxItachi/MatcoreDSL/issues/115>.
Base: `1c1221e89272eb2ad6ff058b8e26de4653f3a93d`.
Worktree: `/home/hamza-usta/mdslc-work/issue-hunt-20261112/exact-gpu-tool-versions`.
Branch: `fix/mdslc-issue115-exact-gpu-tool-versions-v1`.

## Boundary and implementation

Read the full repository `AGENTS.md` and the live open issue before editing.
The assigned worktree was clean at the exact base. The implementation replaces
only the three unanchored GPU executable version gates in
`compiler/lib/regions/GpuCandidates.cmake`. The shared helper lives in the
existing `compiler/cmake/MatcoreDSLToolchainVersion.cmake`; its existing Clang
parser is unchanged.

- llc and LLD accept line-leading upstream, `Ubuntu`, or `Debian` markers.
  Leading indentation and horizontal whitespace are allowed; a marker cannot
  be split across lines. LF and CRLF multiline banners are supported.
- ptxas requires the leading NVIDIA ptxas identity line and takes its token
  specifically from `Cuda compilation tools, release <major.minor>, V<token>`.
  Bare `V` strings, nvcc identity, and unrelated build-note occurrences do not
  establish the ptxas version.
- The complete non-whitespace token is compared with `STREQUAL`. Trailing
  digits, prerelease/build suffixes, and attached punctuation remain part of
  the token and are rejected. Whitespace-separated vendor metadata remains
  valid. Multiple authoritative version lines are rejected as ambiguous.
- Diagnostics identify the tool, required exact version, and complete rejected
  token, or report a missing/ambiguous supported token.

The required versions remain llc **21.1.8**, ptxas **13.3.73**, and LLD **21.1.8**.
Tool discovery, nonzero-process failure, CUDA/HIP header/DSO identity checks,
platform restrictions, flags, numerical semantics, target defaults, and the
single/fused recipe ownership are unchanged. New vendor-prefix or banner
spellings require explicit review; an unknown prefix is not silently admitted.

## Focused regression evidence

From the worktree root, using CMake 4.3.2:

```sh
cmake -DMDSLC_SOURCE_DIR="$PWD/compiler" \
  -P compiler/tests/cmake/toolchain_version_contract.cmake
git diff --check
```

Result: success. The original seven Clang controls still pass. The included GPU
contract reports **81/81 controls passed**: 13 positives and 68 negatives. Each
GPU control checks both the extracted complete token and the decision reached
by including the actual production `GpuCandidates.cmake` file in a child CMake
process. Refusals must come from the selected exact-version gate; a parse error,
unrelated setup failure, or missing admission marker does not count as success.
Rejected nonempty tokens must appear completely in the diagnostic.

The child fixture mocks tool discovery/process output and unrelated target/SDK
setup. It stops immediately after the selected gate, before artifact issuance.
It does not copy the three production decisions or merely call an unused
helper. The existing `toolchain.version_token.contract` CTest registration
already runs this entry point, so no `compiler/CMakeLists.txt` change is needed.

Controls include real Ubuntu/NVIDIA banner formats, upstream/Debian spellings,
multiline LLVM banners, tabs/indentation/CRLF, trailing digits, numeric prefixes,
prerelease/build suffixes, attached punctuation, malformed/missing tokens,
unrelated occurrences, wrong tools, split markers, unknown vendors, and duplicate
authoritative lines in either order.

One initial fixture failed: its intended unrelated ptxas occurrence accidentally
contained a second authoritative version line, correctly producing an ambiguous
token. The fixture was corrected to put the incidental `V13.3.73` on a `Note:`
line. The final suite then passed; no production check was relaxed for it.

## Historical-gate falsification and installed-banner controls

A small local harness evaluated the **unmodified** historical
`GpuCandidates.cmake`, obtained with `git show` at the recorded base, at the
same child fixture's production include site. No checkout/source overwrite or
historical build was performed. For each of llc, LLD, and ptxas:

| Banner mutation | Historical gate | Current gate |
| --- | --- | --- |
| Required token plus trailing `0` | Admitted | Refused |
| Required token plus `-rc1` | Admitted | Refused |
| Required token plus `+vendor` | Admitted | Refused |
| Actual token `99.0.0`, required marker only on a later `Note:` line | Admitted | Refused |

All **12/12 paired falsification controls** exhibited the recorded distinction;
current refusals identified the selected tool's exact-version gate. This
directly demonstrates why retaining the old substring gates fails the contract.

Additionally, actual `--version` output was read (no compilation) from:

- `/usr/lib/llvm-21/bin/llc`: `Ubuntu LLVM version 21.1.8`, full target listing.
- `/usr/lib/llvm-21/bin/ld.lld`: `Ubuntu LLD 21.1.8 (compatible with GNU linkers)`.
- `/usr/local/cuda/bin/ptxas`: NVIDIA multiline banner, release `13.3`, token
  `V13.3.73`, build `cuda_13.3.r13.3/compiler.38244171_0`.

All **3/3 full installed banners** were admitted through the current synthetic
production-gate fixture. The deterministic committed fixtures preserve their
relevant lines; the complete llc target inventory is not a parser requirement.
The retained temporary harness and exact command used for these extra controls:

```sh
cmake -DMDSLC_SOURCE_DIR="$PWD/compiler" \
  -P /tmp/mdslc-issue115-version-falsification.9Z0shq/compare_gates.cmake
```

The `/tmp` harness is local evidence, not a committed dependency or a required
test input. The permanent regression suite needs only CMake and source files.

## Limitations and handoff

These are parser/configure-decision tests with synthetic discovery and output,
plus real version-output reads. No full standalone configure/build/test suite,
GPU execution, emitted-artifact check, benchmark, installed-consumer test, or
new target/toolchain qualification was performed. Version banners do not
authenticate a deliberately lying executable. Unknown vendor prefixes are
intentionally fail-closed, not claimed compatible.

No package installs, GitHub writes, push, PR, merge, runtime/ISA change, or
`CURRENT_STATE.md` update belongs to this change. Integration and independent
review remain with the integration owner.
