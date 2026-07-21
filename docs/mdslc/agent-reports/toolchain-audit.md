# Agent report: repository and toolchain audit

## Assignment and ownership

- Role: repository and toolchain auditor
- Branch: `mdslc/audit-and-adr`
- Start SHA: `351075e4d8af1880330b7c0474d701ca76776dfa`
- Owned files: `AGENTS.md`, ADR 0001, `PREFLIGHT.md`, `STATUS.md`,
  `ROADMAP.md`, and this report
- Excluded: production code and `LEGACY_REUSE_MAP.md`

The first pass was read-only because the original checkout had user-visible
tracked deletions. Documentation changes were made only after an isolated,
clean worktree was assigned.

## Findings

- Prompt branch and SHA matched the repository exactly.
- The original checkout had 50 tracked deletions; it was not modified.
- The host has sufficient CPU, disk, and swap for a small standalone build, but
  14 GiB RAM justifies Ninja `-j2` and `-j1` link/test limits.
- Clang 21.1.8 is installed and passed a `-x c++ -std=c++20 -fsyntax-only`
  driver-only proof.
- No installed Clang version has the LibTooling, ASTMatcher, or Rewriter
  development headers. Post-Sema frontend compilation is blocked.
- `libclang-21-dev` is the coherent package candidate because its 21.1.8
  version matches the installed compiler, LLVM, and Clang libraries. No package
  was installed.
- Default Clang 22 is unsuitable as a complete tuple: installed core packages
  are 22.1.6 while the available development package and MLIR are 22.1.2.
- Root CMake's MLIR 18.1.3 dependency is not currently discoverable and must not
  be pulled into standalone bootstrap v0.
- NVIDIA RTX 4060 Laptop, AMD `gfx1150`, and RyzenAI `aie2p` devices are
  visible. None has been validated through MDSLC.

## Changes

- Added repository-wide standalone compiler guidance.
- Froze the valid-C++, post-Sema, deterministic-JSON, C-ABI, CPU-first
  architecture in ADR 0001.
- Recorded normalized host, compiler, dependency, and device evidence.
- Recorded honest acceptance status and gate-driven roadmap.

No source, build, generated artifact, legacy file, branch outside this
worktree, package, or system configuration was changed.

## Validation

The handoff validation checks:

- Git scope contains only the six owned documentation files.
- Every path named as a required documentation deliverable exists.
- Markdown has no trailing whitespace or tab characters.
- `git diff --check` passes.
- The selected Clang 21 driver-only syntax proof exits successfully.

This report is included in the focused documentation commit; its SHA is
provided separately to the integration owner.
