# Independent review: #115 exact GPU tool versions

**Verdict: request changes.** The patch fixes the reported substring matches;
it still admits a version belonging to another tool and a contradictory ptxas
banner. These are configure-time decisions reproduced with string fixtures,
not failures observed during compilation or on a device.

Reviewed 2026-11-26 on branch `fix/mdslc-issue115-exact-gpu-tool-versions-v1`.
Reviewed source SHA: `b73aeefb53ada96530931b24fb015ed46954cd2f`;
base and observed origin/main: `1c1221e89272eb2ad6ff058b8e26de4653f3a93d`.
Read the root `AGENTS.md`, [open issue #115](https://github.com/onlyxItachi/MatcoreDSL/issues/115)
(`gh issue view 115 --repo onlyxItachi/MatcoreDSL --json title,body,url,state,comments`),
the entire base-to-candidate diff, the implementer's report, and the production
caller and recipe consumers. No production or test file was edited for review.

## Blocking observations

1. **The first matching line need not describe the selected executable.**
   `MatcoreDSLToolchainVersion.cmake:27–31,43–50` scans *every* line for
   llc/LLD markers; it ignores an unsupported initial banner and accepts an
   indented dependency quotation. In the first two commands below the actual
   header says `21.1.80`, from an unsupported vendor; the only `21.1.8` is
   explicitly labelled as a dependency. Both pass. The ptxas identity check
   only authenticates the *position of the initial text*, not which tool
   emitted a later release line: in the last command a malformed ptxas line
   followed by an nvcc banner supplies the accepted token. The suite checks
   unrelated occurrences embedded in prose and two *recognized* version lines;
   neither catches these cases. Limit authority to the tool's supported header
   structure, preserving the explicitly supported LLVM multiline preamble.
   For ptxas, do not let a later, unrelated release record repair a malformed
   first one.
2. **Contradictory NVIDIA release fields pass.** The expression on line 37
   accepts any numeric `release <major.minor>` and discards it. The third
   command is the regular one-line version format with `release 12.8` beside
   `V13.3.73`. This record is internally inconsistent with the qualified CUDA
   13.3 / ptxas 13.3.73 tuple. Reject conflicting fields; this does not
   require checking arbitrary build IDs or authenticating the binary.

Reproduction from the candidate root (Bash; native CMake 4.3.2 on Linux x64):

```bash
src="$PWD/compiler"
cd "$src/lib/regions"
gate() {
  cmake "-DMDSLC_SOURCE_DIR=$src" -DGPU_VERSION_GATE_CHILD=ON \
    "-DGPU_VERSION_TOOL=$1" "-DGPU_VERSION_BANNER=$2" \
    -P "$src/tests/cmake/gpu_toolchain_version_contract.cmake"
}
gate llc $'Acme LLVM version 21.1.80\nBuilt against:\n  LLVM version 21.1.8'
gate lld $'Acme LLD 21.1.80\nRequired linker:\n  LLD 21.1.8'
header='ptxas: NVIDIA (R) Ptx optimizing assembler'
gate ptxas "$header"$'\nCuda compilation tools, release 12.8, V13.3.73'
gate ptxas "$header"$'\nCuda compilation tools, release 13.3, 13.3.730\nnvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.3, V13.3.73'
```

All four exited **0**, printing `GPU <tool> production gate admitted`.
No substitute executable or system installation is required for these controls.

## Confirmed behavior and limits

- From the root, `cmake -DMDSLC_SOURCE_DIR="$PWD/compiler" -P compiler/tests/cmake/toolchain_version_contract.cmake`
  succeeded: seven existing Clang checks ran before the **81/81** GPU controls.
  `git diff --check 1c1221e89272eb2ad6ff058b8e26de4653f3a93d...HEAD` succeeded.
- As independent checks, captured the untrimmed text stdout of
  `/usr/lib/llvm-21/bin/llc --version`, `/usr/lib/llvm-21/bin/ld.lld --version`,
  and `/usr/local/cuda/bin/ptxas --version` via Python `subprocess.check_output`
  and supplied each text as the corresponding child fixture's
  `-DGPU_VERSION_BANNER=` argument. All **3/3** passed. With otherwise valid
  headers, llc `21.1.80`, LLD `21.1.8-rc1`, and ptxas `13.3.730` all
  exited **1**, naming the full rejected token. No intentionally incompatible
  executable was installed or invoked.
- The top-level production include defines the helper before the regions
  subdirectory includes `GpuCandidates.cmake`. The common llc gate (line 33)
  and per-backend checks (lines 77, 94) precede creation of the affected
  image commands. Single and fused recipes use the same selected executable
  paths; flags, Linux x64 / 21.1.8 restriction, header/DSO checks, defaults,
  and existing Clang parser/caller are unchanged. With GPU options off the
  production file returns before invoking these helpers.
- The child includes the actual gate file, but mocks process/discovery,
  target and SDK setup, forces Linux x64, and exits before image issuance.
  CRLF fixtures passed on Linux; no native Windows, other CMake version,
  clean production configure/build, full CTest, artifact inspection, GPU
  execution, benchmark, or hosted CI was run here. This is not evidence of
  hardware support or of an incompatible tool producing bad code. It is
  a review of this revision, not a comparative model benchmark.
