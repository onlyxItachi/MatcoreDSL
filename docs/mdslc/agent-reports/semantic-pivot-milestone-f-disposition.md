# Semantic pivot Milestone F disposition

Date: 2026-08-11

Verdict: **accepted bounded technical limit; native BLAS parity remains
unproven and Milestone 7 remains open**.

## Authenticated repository and GitHub state

- PR #16 merged normally into `main` at
  `e5069758ad04bdb459de2026cad8498b47fda707` after its hosted Linux,
  OpenBLAS-disabled, Windows x64, generic, and hygiene checks passed.
- Issue #15 and GitHub milestone #5 remain open.
- No native-BLAS-parity completion tag exists.
- The last issue comment explicitly retains the partial disposition, dormant
  cooperative packed-B preparation, and the prohibition on inferring an API
  freeze from the result.

These facts were re-read from GitHub on the date above. They agree with
`docs/performance/cpu/native-blas-parity-v1.md` and `docs/mdslc/STATUS.md`.

## Why this satisfies pivot Milestone F

The semantic-compiler roadmap permits either a complete authenticated
forward/reverse parity result or an explicitly reviewed bounded technical
limit. The existing merged Milestone 7 evidence is the latter:

- the final code checkpoint passed its declared correctness, Release, Debug,
  OpenBLAS-disabled, sanitizer, package, consumer, ISA-artifact, native-object,
  legacy-contract, and hygiene gates;
- the parity runner and summarizer fail closed on source, binary, plan,
  manifest, result, thread, placement, and direction provenance;
- the retained external forward receipt contains only 258 of 368 planned
  cases and has SHA-256
  `26e75ecbcfbb19d024fa8a5fa9790b65a2deb5743b39f16a4f22dd39381cfe69`;
- no complete same-checkpoint stable-forward/stable-reverse pair exists; and
- the summarizer therefore emitted no parity, scaling, or full-registry regret
  verdict.

The current machine is a shared development host and was observed running
other compiler workloads during this semantic-foundation work. It is not an
authenticated exclusive, quiescent performance session. Reusing partial raw
bundles or launching another opportunistic subset would weaken the frozen
evidence contract and cannot close Milestone 7.

## Product consequence

The CPU beta may retain deterministic best-legal-provider selection, including
OpenBLAS where it is faster, without claiming that handwritten native kernels
match mature BLAS across the declared envelope. Native/OpenBLAS parity is an
R&D metric, not a semantic-compiler correctness criterion.

The following state is deliberately unchanged:

- cooperative packed-B preparation remains production-dormant;
- the benchmark provider, envelope, timing modes, thread rules, placement,
  legality, and regret scope are not changed;
- Issue #15 and milestone #5 remain open;
- no parity tag is created; and
- no public API or backend contract is frozen.

A later exclusive-host run may execute the exact reproduction commands in
`docs/performance/cpu/native-blas-parity-v1.md`. Until a complete authenticated
pair exists, the only defensible Milestone 7 verdict remains **partially
passed**.
