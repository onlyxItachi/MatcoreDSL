# Independent review: Windows toolset selection

**Accept for hosted qualification, not yet for merge.** Reviewed
`6be2cd0451edae06752191a77b4cebdb6b5bf8f4` against
`1c1221e89272eb2ad6ff058b8e26de4653f3a93d` on 2026-11-26.
I read the complete workflow and [change report](windows-toolset-qualification-v1.md).
No production or workflow edits were made in this review.

- **Baseline evidence.** Read back the [main](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45828441882)
  and [PR](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/45827924148)
  run metadata and logs with `gh run view`. Both show Clang 21.1.8,
  MSVC 14.52.37025 include paths and both RapidJSON probes failing at
  Release configure. Neither log contains STL1000. I independently checked
  the immutable header revisions and the image manifest cited in the change
  report: they support the proposed mismatch and the availability of the
  14.44 component, not the precise diagnostic on those machines.
- **Environment selection.** The required component narrows the existing
  `vswhere -latest -products *` lookup. The generated batch file quotes the
  installation path and calls `vcvarsall.bat x64 -vcvars_ver=14.44`.
  [Microsoft's syntax](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#vcvarsall-syntax)
  identifies this as native x64 and the latest installed servicing patch in
  that family. No SDK version is requested. The separate batch error check
  precedes environment export; PowerShell checks the child's exit code.
  The exported environment allowlist, DIA selection and explicit LLVM
  executable paths are unchanged. The next step checks the selected
  directory leaf and the STL header before configure. Its hash is evidence
  about that file, not a compiler include trace.
- **Failure handling.** Release configure invokes the same command and no
  subsequent native program overwrites its status in that step; the built-in
  [Actions PowerShell wrapper](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#exit-codes-and-error-action-preference)
  propagates the native exit code. The new diagnostic step runs only after
  a failed configure; its bounded log sample might not contain the desired
  probe error, but cannot waive that failure. No build/test/sanitizer step,
  dependency hash, flag, permission, runner selector or upload was relaxed.
- **Local checks.** `git diff --check 1c1221e..6be2cd0` passed. Parsed both
  workflow blobs with `/usr/bin/python3` and PyYAML 6.0.3 `BaseLoader`:
  unchanged job/workflow keys, 19 to 20 steps, exactly one additional
  failure-only step, no `continue-on-error`; apart from the two setup/log
  bodies, the only existing step change is adding the configure ID.
  The default Python lacks PyYAML; the system interpreter needed no install.

This was a Linux static review, not execution of PowerShell, cmd, MSVC, or
an actual SDK selection. Qualification still requires the exact candidate
to complete the hosted Windows workflow, including both configurations,
relocated consumers, ABI/artifact validation, focused ASan and ZIP upload.
