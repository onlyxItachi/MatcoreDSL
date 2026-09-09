# Authenticated multi-TU research v1

This is an isolated source-only compiler experiment, **not installed multi-file
product support**. It adds no production target, serialized module, object import,
source concatenation, semantic interpreter, or new mathematical grammar.

One invocation receives an ordinary host translation unit and two separately
admitted region translation units. The host calls both through canonical
`Result`/`Storage` boundaries. Each region retains its own source/semantic witness,
generated helper and unchanged candidate/runtime path. Ordinary Clang LLVM linking
occurs only after program-wide source/ABI/effect-promise/symbol-owner checks.

The original H1 stack base is `486d3e5d74e17fbf305c73fca3e3760ef7da59b6`;
the experiment advanced by fast-forward to combined H1/private-output-fact source
`8fbdc61258173c46f199f9142092fc1bbec6cb04`. The reused build worktree subsequently
had report-only head `75c78f586801a108ec3cc49b4003b83c34a9cb26`; its compiled source
and artifacts did not change. Build records distinguish those identities.

## Reproduce

Use a completed, frozen, coherent Linux x86-64 Clang/MLIR 21.1.8 H1 build with
experimental regions enabled. This builder reads its generated driver identity,
library paths and existing static libraries; it never rebuilds that product tree.
All output directories must be new. Run from the repository root:

```sh
python3 compiler/experiments/authenticated_multi_tu_v1/build.py \
  --build-root /absolute/path/to/frozen/build-library-release \
  --output /absolute/path/to/new/research-build
python3 compiler/experiments/authenticated_multi_tu_v1/runner_test.py -v
python3 compiler/experiments/authenticated_multi_tu_v1/run.py \
  --driver /absolute/path/to/new/research-build/multi-tu \
  --expected-driver-sha256 SHA256_PRINTED_BY_BUILD \
  --output /absolute/path/to/new/research-evidence
```

Repeat against the frozen `build-library-asan` build. The builder retains its
ASan/UBSan flags and recorded optimization profile. Use:

```sh
env DEBUGINFOD_URLS= \
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  python3 compiler/experiments/authenticated_multi_tu_v1/run.py \
    --driver /absolute/path/to/new/asan-research-build/multi-tu \
    --expected-driver-sha256 SHA256_PRINTED_BY_BUILD \
    --output /absolute/path/to/new/asan-research-evidence
```

`driver.cpp` deliberately includes existing internal implementation files to reuse
the actual frozen-host admission, `sameAbi`, artifact checks and publication logic.
This is research instrumentation, **not** the proposed production API factoring.
The proposed product route should expose/reuse appropriate existing internal
library entry points. The compiler-owned declaration witness is compiled from
canonical header bytes and sealed parameter kinds; it is never accepted as an
input file or semantic module supplied by a caller.

## Executed scope and limitations

Both final profiles pass 14 cases/16 commands, including one real 37-check program.
The 13 negatives require exact checked exit `1`, the specific diagnostic, no
output, no post-validation/link/publication marker, and no sanitizer/crash marker.
Five classifier unit tests separately exercise wrong exits, phases and artifacts.
Raw commands, diagnostics, identities and failures are in [evidence](evidence/).

The positive compares noncommuting 2x2 GEMMs, ordinary cross-TU host utility calls,
owning observations after host mutation, and a later failing invocation while an
earlier observation remains valid. It does not establish cross-region atomicity,
whole-region transforms, cross-module tensor optimization or performance.

The driver intentionally accepts only 2..8 source TUs, one selected region per
mathematical TU, ordinary unannotated canonical foreign prototypes and one `main`.
Every host TU currently must include the canonical region headers. Header-free
ordinary utility TUs, arbitrary declaration attributes, objects/archives,
independently sealed modules, opaque cross-TU `Value` helpers, GPU/Windows and
installation/product CI are not proved by this experiment. Ordinary host C++ ODR
semantics remain the C++ compiler/linker's responsibility; the extra duplicate
definition gate protects issued region/helper ownership, not all C++ definitions.

The final sanitized compiler retains all experiment/admission checks. It uses
Clang's out-of-line definition query to avoid instantiating an incompatible
ASan-poisoning allocator slow path into the prebuilt unsanitized Clang/MLIR tuple.
The initial genuine allocator failure and the earlier asm/pure acceptance bug are
preserved, not counted as successful refusals. See the
[full report](../../../docs/mdslc/agent-reports/authenticated-multi-tu-v1.md).
