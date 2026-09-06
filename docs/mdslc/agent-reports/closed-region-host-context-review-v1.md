# Closed-region host-context input boundary

Starting canonical engineering checkpoint: PR #35,
`e53f2c6302430f574473ba1a644cc42f1169384a`.
Working branch: `mdslc/closed-region-host-context-v1`.

## Ownership and review status

This lane began as an independent input-authentication review. The integration
owner then assigned implementation of
`compiler/lib/frontend/ClosedRegionHostInputs.{h,cpp}` to this lane. Consequently,
this document is **not an independent approval of those files**. The integration
owner and the separate adversarial lane review the combined implementation.
The Clang action, preprocessor observers, semantic walker and evidence issuer
belong to the frontend lane; build integration and physical-race tests belong
to the integration owner. No semantic model or execution route was changed here.

## Observed existing seams

- The existing native frontend authenticates canonical declarations, parsed
  header bytes and `FileEntry` identities. Its compiler-argument construction
  and dependency checks were private implementation helpers, not a reusable
  whole-context admission service.
- `platform_support.h` already provides argument-risk classification, exact
  version comparison, physical identity and path-component snapshots. The
  driver's source-snapshot routines demonstrate before/after byte/identity
  checks without claiming an atomic filesystem transaction.
- Shared inherited-variable rejection does not cover ordinary include-search,
  SDK, toolchain-search or `SOURCE_DATE_EPOCH` inputs. This private path adds
  supplementary rejection without changing the legacy frontend's policy.
- A parsed-header inventory is insufficient for hermetic replay:
  `__has_include` can observe an unopened file, and failed search candidates,
  realpath results, directory iteration and file identities can affect Clang.
  LLVM's existing VFS interfaces expose these observations; no separate
  preprocessor, C++ parser or evaluator is needed.

## Implemented input contract

The private preparation API fixes the configured Clang/resource/loaded-runtime
tuple and constructs the exact complete driver argument vector. Only the
bounded include-directory/define/undefine options and C++20 are accepted.
The tool-owned declaration header is injected by compiler-owned bytes and
synthetic file identity, not by accepting a user-selected trusted pathname.

A recording LLVM filesystem captures positive and negative status/open
outcomes, opened bytes, realpath results, directory contents and locality
queries. It preserves requested path traversal and returned file identities,
including hardlink identity. Successful reads are copied immediately after
before/after snapshot checks. Missing leaves additionally retain ancestor
presence and path identities; unrelated parent-directory timestamps are not
treated as namespace changes. Main source, resource directories, compiler and
loaded Clang library identities are checked before issuing the frozen record.

Freeze and the final pre-seal check revalidate physical outcomes and contents.
These checks detect the bounded tested races; they do not establish an
OS-atomic snapshot against an arbitrarily hostile concurrent writer.

Replay has no physical filesystem fallback. An unrecorded query is an audit
failure, distinct from an originally recorded `ENOENT`. Old source evidence
denotes its captured source context, not whatever files later occupy the same
paths. Replaying it does not require live source/header equality. Current
inherited compiler inputs are rejected before replay; this is an environment
check, not a reread of live source files.

The context fingerprint uses length-bound fields and explicit collection
boundaries. The digest itself grants no authority: the private frontend issuer
must additionally authenticate the parsed main/header identities, admit the
unchanged closed grammar, compare repeated admission and normalized Clang
preprocessing observations, and pass the existing semantic/module verifier.

## Falsified drafts and corrections

The first integrated host run executed 157 checks and failed five. The most
important failure was a genuine source-binding bug: the local helper named
`bind` accepted `llvm::StringRef`, while argument-dependent lookup selected
`std::bind` for several `std::string` calls. Those calls returned discarded
functors instead of appending fingerprint fields. A header-only byte change
therefore failed to change the fingerprint, and a newly admitted context could
incorrectly match an old module. The helper and every call were renamed
`appendBoundField`; the integration owner's header-only mutation and cross-seal
rejection checks retain the counterexample.

Three other initial failures exposed an incorrect historical-evidence policy:
later pairing rechecked live source/header files. The frontend lane removed
that recheck from historical pairing while retaining both physical checks
before seal issuance. Snapshot replay remains mandatory and has no fallback.

## Validation record

The integration owner's retained Release CTest log at 19:33 local time recorded
five focused tests passing: 157 host-context checks, 486 hermetic admission
checks, 71 semantic checks, 44 native two-GEMM extractions, and ordinary
compilation rejection of the private fixture. That binary predates the final
collection-count and replay-environment refinements.

The subsequent retained 19:41 Release log covers the current input-layer source
(binary timestamp 19:39, source timestamp 19:38): four closed-region tests
passed, including **169 host-context checks, 486 hermetic admission checks,
71 semantic checks**, and ordinary fixture compilation rejection. This lane
inspected the log; the integration owner ran the coordinated build/tests.

The host checks include header-only fingerprint changes and rejection of an
old module against newly admitted changed-header evidence; replay of old
headers after physical mutation; recorded negative lookup after file creation;
presence-only lookup after the unopened file is removed; main/header mutation
before freeze; same-byte physical identity replacement; symlink retargeting;
late negative-lookup changes; known absence versus an unrecorded replay query;
altered compiler/resource identity; poisoned inherited inputs at admission and
later pairing; and context changes from options, working directory and search
order. Full integration/hosted evidence belongs to the integration checkpoint.

## Boundaries retained

This is test-only inspection infrastructure, not a public frontend/API,
serialized authority format, filesystem sandbox, or compiler toolchain update.
It does not execute host code, build a graph by running a lambda, authenticate
arbitrary C++ effects, introduce header-defined semantic helpers, transform the
closed semantic model, lower storage, or generate executable regions. The
existing CPU runtime/provider route remains the sole execution authority.
Windows execution, general compiler-option support and performance claims are
not established by this Linux proof.
