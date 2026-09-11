# Compiler-issued staged GPU candidates v1

This is a correctness-first, opt-in execution boundary, not a GPU optimizer or
performance milestone. It supersedes the disconnected next-step descriptions in
the historical [NVVM](agent-reports/multitarget-nvvm-v1.md) and
[ROCDL](agent-reports/multitarget-rocdl-v1.md) research records without changing
what those experiments proved at their own checkpoints.

## Source and authority

The existing authenticated `mdslc-region` source driver accepts explicitly forced
`--candidate generated-nvvm` and `--candidate generated-rocdl`. There is no new
source syntax. Original C++/header identities, canonical declarations, immutable
logical values, all-MAY-alias external storage and ordered read/publication/
observation/failure frontiers retain their existing contracts. Default native,
automatic generated CPU, and optional external-provider policies are unchanged.
An unavailable or incompatible forced target never borrows CPU or the other GPU.

The whole-region Matcore MLIR remains an exact **untransformed paired witness**.
Only an isolated compiler-issued GEMM primitive obtains device execution authority:

```text
Authenticated source -> sealed Program -> unchanged host orchestration
                                           |
                     forced private candidate registry
                                           |
Built-in strict GEMM semantics -> verified Tensor/Linalg -> One-Shot Bufferize
  -> upstream parallel-loop/GPU mapping -> GPU outlining -> exact outlined check
      +-> GPU-to-NVVM -> LLVM NVPTX -> PTX -> ptxas -> sm_89 image
      +-> GPU-to-ROCDL -> LLVM AMDGPU -> object -> lld -> gfx1150 image
  -> compiler-build embedded fill + GEMM images -> isolated candidate DSO
  -> private device execution -> checked completion -> immutable host Value
  -> existing ordered publication/observation/completion
```

The issuer takes a closed target enum, never user-provided IR, image bytes,
function pointers or runtime artifact paths. It reuses `buildStrictGemmStagesV1`,
not a separately handwritten device GEMM. Standard MLIR lowering owns the
conversion; target facts/scheduling remain below mathematical semantics.

## Exact initial realization

Each output element has one block with one thread. A positive-zero fill kernel
precedes a strict increasing-K GEMM kernel. The outlined witness checks the exact
rank-2 descriptors, indices, stores, loop order, arithmetic and launch shape;
extra operations/attributes and forged alias or numerical facts are rejected.
Fill and GEMM are separate embedded images with the same private kernel name.
This simple recipe is not a tile, vectorization, tensor-core or fusion claim.

Both candidates execute separate `f32` multiply/add under the established
`strict_f32` numerical contract. A more permissive source profile does not force
reassociation. Numerical implementation permission is not inferred from device
instruction availability. Machine-image tests require the exact ELF target,
expected kernel symbol, no unresolved imports, and separate multiply/add with
no fused/matrix instructions; NVIDIA tests also reject explicit FTZ modifiers.
These tests do not prove all floating-point behavior. Physical discriminator
tests independently cover contraction, reduction order, subnormals, signed zero,
nonfinite values, rectangular and degenerate geometries.

The first recipe has explicit **realization qualification bounds**:
`M,N,K <= 65535`, `M*N <= 2^20`, `M*N*K <= 2^26` (zero K handled separately).
These are bounded launch/work prerequisites, not semantic shape restrictions,
cost-model thresholds or cross-target dispatch policy. Runtime shape arithmetic,
availability and qualification are checked before result allocation and zero-K
shortcuts. Each adapter repeats relevant checks before narrowing or launch.

## Storage and failure contract

Inputs at this seam are immutable host Values, not live source Storage. The
explicitly selected staged candidate makes private host snapshots, allocates
device A/B/C, uploads inputs, runs fill/GEMM, downloads into private host staging,
and completes fallible cleanup before copying into the private candidate output.
Transfers and allocations are documented properties of this forced realization,
not source-language effects, implicit residency promises, or zero-copy behavior.
Publication still occurs only at the existing ordered semantic frontier.

All CUDA/HIP calls run on a fresh joined worker thread. This isolates the caller's
CUDA context/HIP device and thread-local error state, FP environment and errno.
There is no public asynchronous Value, persistent device-resident intermediate,
shared GPU worker pool, global device reset or hidden target fallback.

If completion cannot be established, device resources and heap-owned transfer
staging are quarantined for process lifetime and that adapter instance is
poisoned. A driver error is not evidence that queued work stopped; freeing a
possibly live transfer buffer would be incorrect. Failed execution/transfer/
cleanup never issues a new Value or mutates a caller-visible destination.
Earlier successful publication/observation effects remain visible. Standard
thread join failure is fail-stop, not a normal return with dangling worker state.

Valid caller objects/lifetimes, race-free storage, conforming allocation, standard
thread/runtime behavior and trusted dynamic loading remain prerequisites. This
is not a sandbox, a crash-atomic transaction, or recovery from a device/OS failure
that terminates the process. Direct unsafe foreign interposition is unsupported.

## Host linkage and installation

The source driver snapshots and rechecks the actual configured CUDA driver/HIP
runtime DSO bytes alongside the existing candidate/runtime/provider artifacts.
CUDA stubs are rejected. The actual dependency DSOs are linked directly with
RUNPATH so a transitive dependency does not accidentally depend on development
`LD_LIBRARY_PATH`. Full defined-symbol ownership protects them against original
host replacement, including misleading C++ spellings/assembler labels.

The GPU-enabled candidate DSO additionally requires the known libstdc++ worker
ABI and reserves thread creation/join/ownership symbols against original host
definitions. This is a bounded dependency contract, not an attempted model of
all standard-library interposition. The deterministic embedding step binds image
bytes into the private DSO; image/embedding-manifest tests and driver DSO identity
checks cover distinct links, not one interchangeable cryptographic claim.

Installed source compilation exercises the installed driver and its private DSO.
The separate internal static archive test contract must explicitly link its
configured accelerator dependency files, just as it explicitly links its provider;
manual archive linking still does not inherit authenticated source authority.
No GPU, LLVM or MLIR dependency is exported through `MatcoreDSL::Runtime`.

## Toolchain and reproduction

Initial product recipe: native Linux x86-64, coherent Clang/LLVM/MLIR **21.1.8**.
The NVVM recipe uses PTX 8.0 / sm_89, CUDA 13.3 headers and ptxas **13.3.73** with
`--fmad=false`; ROCDL uses gfx1150, LLVM/lld **21.1.8**, IEEE denormal options and
actual HIP **7.2.70201**. Version refusals are intentional qualification gates,
not claims that other releases cannot work. No LLVM/Rust source build is needed.

Add to the existing supported experimental-region CMake configuration:

```sh
-DMDSLC_ENABLE_EXPERIMENTAL_GPU_ISSUER=ON
-DMDSLC_ENABLE_EXPERIMENTAL_NVVM=ON
-DMDSLC_ENABLE_EXPERIMENTAL_ROCDL=ON
```

All three options default OFF. Issuer-only mode needs the existing MLIR/LLVM
tuple but neither CUDA nor HIP SDK/runtime/hardware. Execution options require
their exact supported SDK/toolchain dependencies; no silent download or fallback.

For an enabled/qualified device, the unchanged repository example can be used:

```sh
build/bin/mdslc-region compiler/examples/experimental/two_gemm.mdsl \
  --region pipeline --candidate generated-nvvm -o build/nvvm-example
build/nvvm-example
build/nvvm-example fail-late
# Substitute generated-rocdl for the AMD candidate.
ctest --test-dir build --output-on-failure -R '^generated_gpu\.'
```

GPU-enabled physical/source tests require the actual target and do not skip.
The ordinary hosted Linux matrix exercises the vendor-independent issuer,
qualification and compiled-out target refusals; it is not GPU execution evidence.
Local GPU-disabled visibility tests also prove that an already enabled binary
fails without borrowing another implementation.

## Proven, unresolved, and next ownership

Exact validation/checkpoints belong in the integration evidence report and
`CURRENT_STATE.md`. Real execution was exercised on RTX 4060 Laptop (`sm_89`)
and Radeon 890M (`gfx1150`), not every NVIDIA/AMD device. Real CUDA memcheck and
normal/ASan+UBSan mocked-driver tests are separate evidence. Real HIP+ASan has an
unresolved alternate-signal-stack teardown failure; normal and host-UBSan HIP
execution do not erase that limitation. No physical HIP+ASan pass is claimed.

Unsupported here: whole-region GPU lowering/fusion, device residency/transfer
language APIs, zero-copy, async execution, reuse across GEMMs, dynamic scheduling,
automatic GPU selection, GPU providers, NPU/Metal, Windows GPU region execution,
general rank-N layouts/dtypes, performance or BLAS parity. Issue #15 remains
partial/open, and Issue #20 is not implemented or closed by these private seams.

The reusable gain is a second family of mechanically checked realizations below
the same mathematical/value/effect contract. Future tiling, fusion or residency
must establish its own legality-preserving derivation and observable/failure
trace proof; an available GPU dialect alone confers none of that authority.
