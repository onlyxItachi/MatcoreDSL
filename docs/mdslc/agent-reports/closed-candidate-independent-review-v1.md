# Independent candidate-coexistence arithmetic review

**Final verdict: ACCEPT the registry together with its owning-provider fix.**

Required implementation checkpoints are registry
`78135023dfdd99bd2dd8c3323c2137f6041d64ac` **plus** provider correction
`437bdcc1d15373634f6cacd5439566ea6cd3d689`. The latter serializes the complete
provider-policy transaction inside the existing shared adapter, including its
conformance probe. It does not hold the policy mutex while waiting for the
conformance once flag; the probe acquires the mutex itself. Early exits and
restoration remain within the policy scope. Both legacy and new routes use it.

The reviewer reran the original oracle in **100 fresh Release processes**
(4.73 seconds) and **30 fresh whole-issuer ASan+UBSan processes** (4.44 seconds):
all passed, 47,638 checks per process. The deterministic owning-adapter scope
test also passed: 128 requested GEMMs, maximum one active provider-policy scope,
correct results and one actual provider thread. The implementation author's
preserved pre-fix binary fails that same scope test; the original Release
counterexample below was independently reproduced by this reviewer.

Final provider implementation hashes:

- `cpu_openblas.cpp`: `e1f5cfdbf9f57b67720dd9eaf5058f5d66d38934eca1b8165549ce73013bd5a5`.
- `cpu_openblas.h`: `172ef492ffd6ce7fc377109144f4b181ab3106b0e6f169ecaebb133a42e5bad1`.
- Independent oracle, with failure diagnostics retained:
  `2bcb51e69e3d12af435041b5e9820518e3e91f130545ead038d611ea378b5daa`.

No tensor-resource/global-publication mutex is introduced. Sessions remain
thread-confined. One runtime DSO owns this adapter; concurrent direct external
provider activity or independently embedded adapter copies must supply equivalent
exclusion and are not covered by these tests. Serialization is a correctness
adaptation, not a provider-performance or parallelism claim.

## Historical HOLD: concurrency falsification

The initial instrumented pass below is retained as historical evidence, but a
fresh-process Release repetition subsequently falsified provider independence.
Do not treat this report's earlier passing run as unconditional acceptance.

The reviewer reproduced a distinct-Session concurrent-first-use failure on the
third fresh Release process. The adapter returned `candidate_failure`, issued no
Value, and suppressed publication correctly. A throwaway link-time wrapper around
the real legacy C ABI identified status 19, `OpenBLAS SGEMM failed or did not honor
the thread policy`, with no selected implementation or actual threads. FP
restoration itself did not fail. Arithmetic membership remained correct.

Upstream OpenBLAS 0.3.32 explains the contradiction: its
[`openblas_set_num_threads_local` implementation](https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/v0.3.32/driver/others/openblas_set_num_threads.c)
calls the global setter and writes `blas_omp_threads_local`; the pthread
[`blas_server.c` implementation](https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/v0.3.32/driver/others/blas_server.c)
declares that variable as ordinary global storage, not TLS. The name "local"
does not prove thread-local ownership. Concurrent save/configure/restore sequences
can interfere even though each new Session has private tensor resources.

The recommended correction belongs in the existing shared provider adapter:
serialize its complete policy transaction, including probes, across legacy and
new routes. A lock only in the new wrapper leaves mixed-route interference.
Concurrent external direct OpenBLAS policy mutation must remain an explicit
unsupported host precondition unless it participates in the same authority.
Repeat fresh-process tests and validate mixed-route behavior before acceptance.
The counterexample does not require changing immutable values, output isolation,
source architecture or the per-publication failure guarantee.

## Initial bounded review before repeated-Release falsification

The reviewer authored the earlier host adapter but not this candidate registry,
generated issuer, provider selection, provider gate or implementation tests.
This review independently attacked the new arithmetic permissions, provider gate,
reports and distinct-Session first-use concurrency. Storage/retirement attacks
were also reviewed independently by the separate publication adversary.

Reviewed worktree: `MatcoreDSL-wt-closed-candidate-coexistence-v1`, based on
`f2569546b9d728592c2d93c4a449c24556a1c5e8`, implementation pending its own commit.
The exact reviewed implementation identities are:

- `compiler/lib/runtime/closed_host_v1.cpp` SHA-256
  `08d51142dc47692ab75b41576d91e5ed556efe1745b2918829534cc0e467acf2`.
- `compiler/lib/runtime/closed_host_v1.h` SHA-256
  `c008984b78bbdce06610dcd2c5c8f5ffe5aec979698dd6cf421d14863cc74626`.

## Independent execution evidence

The separately authored `compiler/tests/closed_candidates/independent_arithmetic_test.cpp`
has SHA-256 `52cb545ede9482d87c877529f8fc87f3ebddf95ce9f564bc33e9f78670263cf2`.
It was first built outside production paths and linked against the actual
instrumented production candidate archive and runtime, not a test-hook variant.

**47,638 checks passed, zero failures**, with ASan+UBSan and warnings-as-errors.

- 180 deterministic random rectangular geometries with K=1..4, including signed
  zero, NaN, infinities, subnormals, very large/small exponents, cancellation and
  FMA-sensitive values.
- A distinct numerical oracle enumerates binary addition trees and permutations
  over every original product plus exactly one positive-zero seed. It optionally
  contracts an original product leaf with its sibling into FMA. It does not share
  the implementation's fixed nine-expression K2 probe or compare reassociated
  candidates to one arbitrarily selected reference result.
- Native strict and generated strict candidates require bit equality with a
  volatile separate f32 multiply/add increasing-K oracle; NaN payload is excluded
  exactly as the semantic contract specifies.
- Existing reference and authenticated OpenBLAS require membership in the
  independently enumerated permitted expression family.
- Eight distinct thread-confined Sessions concurrently perform first provider
  use, each with private resources and differing caller rounding/FP flags.
  All return the expected rectangular result, report actual one-thread provider
  execution and restore their own caller FP environment. Exactly one invocation
  reports running the once-only additional closed-provider probe.

The reviewer also reran the complete `closed_candidates` CTest surface in the
whole-issuer Debug ASan+UBSan build: **10/10 passed** (0.20 seconds), including
hostile wrong-numeric, wrong-variant, wrong-thread, control-corruption and partial
output failure substitutes, production symbol checks and original adapter tests.

The actual provider in the reviewed build is OpenBLAS 0.3.32, runtime core
Cooperlake, Linux x86-64. No performance measurements were made.

## Architecture assessment

The registry preserves the required ownership boundary: candidate choice is a
compile-trusted enum; the original native/provider C ABI remains unchanged;
generated execution accepts only its fixed issued primitive; every candidate
computes into isolated output; and successful Value issuance follows numerical
state checks and completion. Explicit incompatible/unavailable choices fail
instead of silently falling back, including zero-size requests. Empty output
and zero reduction are reported as local semantic realizations, not as a claim
that a provider or leaf executed the requested GEMM.

The additional provider gate is a compatibility probe, not a numerical proof for
all inputs, versions or cores. Production still trusts the linked implementation
to satisfy the stated IEEE f32/FMA/reassociation contract, just as it trusts its
compiler and generated machine code. The independent K1..K4 evidence broadens the
tested scope but does not change that trust boundary. No planner crossover policy,
packed-path authority, multithreaded provider authority or performance claim is
introduced by this acceptance.

No arithmetic counterexample was found in this review. Rejecting strict
requests for existing/provider candidates is essential: their allowed FMA is not
the strict increasing-K separate multiply/add contract. The prior generated
N-zero outer-loop limitation remains safely short-circuited in the adapter.

## Remaining limits

- This is not ThreadSanitizer evidence or general concurrency proof. The tested
  Sessions and storage are disjoint; a Session remains thread-confined and no
  concurrent conflicting resource access is authorized.
- No arbitrary hostile linked runtime/provider or allocator is sandboxed.
  Negative substitutes falsify selected boundary assumptions; finite probes
  cannot authenticate arbitrary changing behavior or memory corruption.
- External OpenBLAS internals were not rebuilt with sanitizers. The generated
  leaf, adapter and runtime in the instrumented build were instrumented within
  their documented scopes; UBSan source checks are not invented for raw LLVM IR.
- The normal-return publication guarantee is not crash recovery, concurrent
  atomicity, device export or a whole-region transaction.
- Source admission and product driver integration remain separate authorities.
  Candidate selection alone cannot authorize imported semantic JSON/MLIR or an
  arbitrary C++ region.

Scratch evidence is retained at
`/tmp/mdslc-coexistence-arithmetic-review.em2NTo/arithmetic_attack.cpp`; the durable
oracle above is identical. Reproduction uses Clang 21 C++20 with
`-O1 -g -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -frounding-math
-fsanitize=address,undefined`, the production candidate archive from
`build-candidates-asan/lib`, `-lmatcore_runtime`, its absolute rpath and `-pthread`.
Execution used `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1`.
