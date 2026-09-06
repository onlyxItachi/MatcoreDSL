# Generated strict CPU candidate: implementation evidence

Base: canonical `fd64850a0c8cb7d2c0801a64dd94d515dd714130`.
Implementation branch: `mdslc/generated-strict-cpu-candidate-v1`.
This is the implementation owner's report, not independent approval.

## Bounded result

The private no-input issuer constructs a compiler-owned strict GEMM primitive
using the existing frontend-neutral closed-region records and verifier. Its
semantic witness remains `inspection_only_no_execution`: this is not fabricated
Clang/header provenance. No external source, Program, or serialized MLIR is an
issuer input. The resulting built-in implementation is not authorization to
execute an arbitrary source region.

The exact verified primitive supplies the lhs/rhs/result tensor types to a
new strict tensor Linalg projection. Matcore's existing contraction-topology
checker verifies its maps. Separate scalar multiply/add, positive-zero overwrite,
ordered operands, unencoded dynamic f32 tensors, exact destination dataflow and
absence of additional attributes/operations are checked. Upstream One-Shot
Bufferize produces the original scratch destination with zero tensor
allocation/deallocation/out-of-place decisions. The buffer verifier rejects
extra copies, allocations, casts and output-identity changes.

The private issuer then drops only the proven redundant return descriptor,
emits the upstream three-descriptor C wrapper, and runs the pinned LLVM/MLIR
21.1.8 Linalg-to-loops, affine, SCF/CF and LLVM conversion/translation pipeline.
LLVM verification, no-fastmath checks, separate fmul/fadd footprint and a closed
call/symbol set precede LLVM IR publication. CMake compiles that real LLVM IR
with the configured Clang, links an ordinary executable, and executes it.

Linalg's reduction iterator does **not** itself encode increasing-K order.
Only this fixed scalar lowering pipeline is defended. Arbitrary reduction
tiling, vectorization, reassociation and pass substitution require new evidence.

## Exact leaf preconditions and ownership

- Linux x86-64, coherent 21.1.8; no AVX/FMA requirement or performance claim.
- Three live descriptor objects, offset zero, strides `{columns, 1}`;
  lhs `[M,K]`, rhs `[K,N]`, destination `[M,N]`, nonnegative signed-index extents.
- Representable byte/address arithmetic, live f32 storage capacity, alignment,
  access permissions and race freedom remain caller/adapter obligations.
- Empty data may be null. M/N zero does not write; K zero fills C with positive
  zero rather than adopting a provider's possibly different quick-return rule.
- Input/input overlap is legal. Destination must be isolated from both inputs.
- Nearest-even, gradual underflow and compatible masked FP controls are required.
  The leaf does not preserve host FP status, allocate logical values, snapshot
  resources, select providers, attribute failures, or publish results.
- The region adapter, not this leaf, owns those retained obligations.
- The fixed strong built-in symbol must be linked once, not once per source TU.
  Its memref descriptor ABI remains private and uninstalled.

The old explicit mutating GEMM profile permits FMA; the closed strict profile
does not. Reusing the old mutating semantic certificate as strict source
authority was rejected. Existing bufferization model registration and topology
machinery are reused, but no inspection authority field is flipped.

## Validation and retained negative results

The generated-code tests intentionally include an ASan-instrumented object and
an out-of-bounds negative control even in Release and Debug builds. Therefore
Linux MLIR test builds require the matching Clang sanitizer runtime (Ubuntu
package `libclang-rt-21-dev`). The first hosted Release/Debug runs at integration
head `085d2f20b02f5696c85c3961b627d5a569755cfc` failed to link the ASan tests
because those jobs had not installed that runtime; the dedicated sanitizer job
passed. The dependency is now explicit in both affected installation steps;
the test and its negative control remain mandatory.

An initial root full local run returned 88/90 when the workflow correction was
edited during testing: the existing package and benchmark-runner tests correctly
rejected the newly dirty source. This is not execution evidence. A clean committed
reconfiguration and full rerun are required before integration.

| Local validation | Exact outcome |
| --- | --- |
| Release generated candidate + affected existing closed/structured/buffer tests | 10/10 PASS, 0.28 s |
| Whole issuer Debug ASan+UBSan, generated execution and object/negative controls | 7/7 PASS, 0.20 s |
| Stage/issuer adversarial executable | 30 checks, zero failures |
| Implementation-owner physical execution fixture | 174 checks, zero failures |
| Independent randomized execution oracle, both normal and generated-ASan objects | 17,768 checks per run, zero failures |
| Independent additional structural attacks after encoding fix | 9/9 rejected, zero false accepts |

Release command: `ctest --test-dir build-candidate -R
'(generated_cpu|mlir.closed_region_semantics|mlir.structured.gemm-handoff-v1|mlir.bufferization.gemm-handoff-v1)'
--output-on-failure -j1`. Sanitizer command uses `build-candidate-asan`,
`-R generated_cpu`, `DEBUGINFOD_URLS=`,
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1:check_initialization_order=1`,
and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

The physical fixture covers rectangular and unit geometries, zero
M/N/K, null empty K inputs, input/input alias, NaN, infinity-times-zero,
subnormal/signed-zero behavior, a FMA discriminator and three/four-term
increasing-K cancellation discriminators.

The entire issuer also builds/runs with ASan+UBSan, leak checks, strict string
checks and initialization-order checks. No new sanitizer exclusions are added.
Raw generated LLVM does not acquire source-level UBSan checks merely from a
compiler flag; explicit overflow/storage guards remain the adapter's job.

The generated code has its own ASan negative control. In the initial scratch
experiment, `clang -fsanitize=address` on LLVM IR emitted ASan initialization
without instrumenting unmarked functions: intentional generated heap OOB
incorrectly exited zero. Adding LLVM `sanitize_address` function attributes
caused genuine generated-code load/store checks and a heap-buffer-overflow.
The permanent test requires a nonzero ASan failure inside the generated symbol.
Both generated definitions are marked in the instrumented lane.

Object inspection checks real ELF x86-64 symbols, separate `mulss`/`addss`, no
FMA opcodes, and bounded imports. Clang -O2 may lower zero fill to `memset`;
this is a destination overwrite, not a tensor allocation or source-value copy.
Compiler instrumentation imports are admitted only in the ASan object lane.

Independent adversarial review found that an otherwise valid tensor type could
carry a forged target/layout encoding and still pass the first stage verifier.
The issuer never accepted external stages, so this was not executable-source
authority. It nevertheless contradicted the exact primitive check: the verifier
now rejects all nonempty tensor encodings and the exact counterexample is a
permanent regression fixture. Other fixtures reject nested noalias/authority
claims, swapped operands, wrong accumulator, negative-zero seeds, FMA/reassoc
flags, wrong output return/destination and additional operations.

Initial scratch LLVM translation also failed when CF conversion was omitted;
the production pinned pass list includes `createConvertControlFlowToLLVMPass`.

Independent object inspection also found that the baseline N=0 case retains
an outer M loop. Extremely large M with empty output can therefore take
impractically long despite zero tensor footprint. The adapter must short-circuit
empty output before calling the leaf. This is a recorded baseline-lowering
limitation, not a performance claim or a reason to encode a target schedule in
semantic IR. Physical tests cover modest zero extents, not huge empty-loop
performance.

## Not proven here

This is not a public region language, installed compiler feature, authenticated
source-to-region executable, provider coexistence proof, FP/publication adapter,
accelerator backend, BLAS parity result or optimized schedule. Existing CPU
runtime/provider semantics and Windows compatibility targets are unchanged.
Windows and exact-22 compatibility configurations do not expose this 21.1.8
Linux-only candidate lane. Full repository/hosted integration belongs to the
integration owner after independent review.

The next connection is the authenticated source emitter plus ordered resource,
failure and FP adapter, selecting this generated strict primitive only after
its retained runtime predicates are discharged.
