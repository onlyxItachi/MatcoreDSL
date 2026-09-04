# MLIR upstream portability control v1

## Scope and checkpoint

- Canonical campaign base: `327530d287e41c4115365598e76b17e149a1c45a`.
- Compatibility implementation validated below:
  `a30bf0c0427875e08352d850898f0eb84f99f484`.
- Branch: `mdslc/mlir-upstream-portability-v1`.
- Product truth remains the exact LLVM/Clang/MLIR `21.1.8` tuple. The new
  `MDSLC_EXPERIMENTAL_TOOLCHAIN_VERSION=22.1.8` switch is an advanced,
  internal compatibility control. It is empty by default, admits no other
  version, emits a non-product warning, and is not an installed consumer
  contract.

This lane did not migrate the product toolchain, change execution authority,
change the structured GEMM schema, or add a target/backend.

## Toolchain identity

The host's distro LLVM 22 installation is not coherent enough for this test:

- `clang-22` and `llvm-22-dev`: `22.1.6`;
- `libmlir-22-dev`: `22.1.2`;
- `/usr/lib/llvm-22/bin/mlir-opt`: `22.1.6`.

An attempted exact-22.1.8 configuration against those packages failed closed
at `find_package(LLVM 22.1.8 EXACT)`. They were not used as compatibility
evidence.

The coherent control is the official LLVM `22.1.8` Linux x64 archive from
`llvm/llvm-project` release `llvmorg-22.1.8`, extracted without system changes
under `/home/hamza-usta/.local/toolchains/llvm-22.1.8-linux-x64`.

| Artifact | SHA-256 |
| --- | --- |
| `LLVM-22.1.8-Linux-X64.tar.xz` (1,938,859,476 bytes) | `df0e1ecf16caf3489a272a5eea4eec9b0d82878f6477fa309504f918a0006384` |
| `bin/clang++` | `31a40dc31f3c15a47aa119aa148f339bf363d8f61202ddbdacbd3f24b71ba113` |
| `bin/mlir-opt` | `45092a31bf61175e2917ce59e12e71f8e287a73bb6bb241142b362f2c1e03850` |
| `lib/cmake/llvm/LLVMConfig.cmake` | `e512d42bc2ee2527f63c3121f460b478615249413171215552a4606554c3f47f` |
| `lib/cmake/mlir/MLIRConfig.cmake` | `dcb08a02af7b2194b3bc61c910b3c2711d00aa3b010cd50d64bd9802181d7efd` |

The archive reports commit
`ca7933e47d3a3451d81e72ac174dcb5aa28b59d1`. Its LLVM package declares
`LLVM_ENABLE_RTTI=OFF`.

## Observed compatibility changes

1. Clang 22 changed `DeclRefExpr::getQualifier()` and
   `NestedNameSpecifier` from the pointer/kind API used by 21.1.8 to the value
   API with `NamespaceBaseDecl`. A version-bounded adapter preserves the same
   canonical namespace and using-shadow rejection policy.
2. Clang 22 AST JSON may repeat a referenced `FunctionDecl` below a using
   declaration. The later, importing-namespace copy previously overwrote the
   authenticated declaration map entry, causing a prohibited qualified
   re-export to disappear as zero captured operations. Declaration collection
   now retains the strongest authenticated classification for one declaration
   identity. Existing re-export negatives falsify regressions on both 21 and
   22.
3. The official 22.1.8 package is built without RTTI. Only internal targets
   that directly compile against LLVM/Clang/MLIR inherit the package's RTTI
   mode; applying `-fno-rtti` globally was tested and rejected because it
   breaks unrelated runtime code that legitimately uses C++ RTTI.
4. Runtime compiler discovery, diagnostics, and tests encoded `21.1.8`
   literally. They now use the configure-selected exact tuple. The default
   generated value is still `21.1.8`.
5. The source-inaccessible package proof performs a clean nested producer
   build. It must propagate the experimental tuple when—and only when—the
   outer build selected it; otherwise it correctly falls back to product
   `21.1.8` and rejects MLIR 22.
6. No Matcore semantic, structured-GEMM, CPU-runtime, installed-ABI, or
   benchmark-contract change was required by MLIR 22.1.8.

## Architectural implication

The certified semantic-to-structured seam is a useful portability control:
new upstream tuples can be evaluated behind an exact opt-in without weakening
the product pin or mixing migration into semantic work. Compatibility glue is
owned at the frontend/build boundary. Structured transformations and their
dialect behavior remain upstream-owned; Matcore still owns the admission and
semantic verification around them.

This evidence supports keeping a bounded compatibility switch. It does **not**
support declaring 22.1.8 a product-supported tuple, removing the 21.1.8 pin,
or promising cross-version serialized-IR compatibility.

## Validation

At clean implementation commit `a30bf0c0427875e08352d850898f0eb84f99f484`:

- LLVM/Clang/MLIR 22.1.8, Release, MLIR on, native plus bootstrap frontends,
  OpenBLAS off: build passed; CTest `64/64` passed in `121.15 s`.
- Audited LLVM/Clang/MLIR 21.1.8, Release, MLIR on, native plus bootstrap
  frontends, OpenBLAS 0.3.32 on: build `132/132` passed; CTest `64/64`
  passed in `207.22 s`.
- The exact-22 matrix includes native/frontend adversarial tests, exact
  structured-GEMM verification, explicit GEMM CPU execution, runtime/planner,
  installed consumer and source-inaccessible relocation, C ABI, package,
  benchmark provenance, and parity-harness contract tests.
- A configuration using the mixed distro 22 packages was rejected before
  compilation because LLVM `22.1.6` did not satisfy exact `22.1.8`.
- `git diff --check` and Python bytecode compilation of the changed harnesses
  passed.

## Unresolved evidence

- No Windows 22.1.8 distribution/build was tested.
- No 22.1.8 Debug or sanitizer matrix was run.
- OpenBLAS-enabled behavior was covered on product 21.1.8, not on the 22.1.8
  compatibility build.
- The campaign's contraction, bufferization, and vector experiment branches
  were not yet composed with this branch at this checkpoint.
- No performance, Native BLAS parity, zero-copy, GPU, NPU, or target-support
  conclusion follows from this control.
- Newer upstream deprecations should be addressed only when they block an
  admitted exact tuple; warning churn is not a reason to fork upstream
  transformation machinery.

## Disposition

The compatibility implementation is a candidate for canonical integration
after independent review and hosted 21.1.8 checks. Keep 21.1.8 as product
truth. Use 22.1.8 only as an explicitly selected, exact compatibility lane
until a separate migration decision establishes broader platform evidence.
