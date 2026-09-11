# Strict generated CPU target experiment

This leaf-only experiment invokes the real closed MLIR issuer. It does not admit
source programs, register candidates, alter provider policy, benchmark or tune.
The existing source driver remains Linux x86-64-only at the base checkpoint.

The issuer's reviewed row-contiguous M/K/N schedule preserves each output's
increasing K and separate f32 multiply/add. Six isolated object compilations use
baseline x86-64, AVX, AVX2, AVX2+FMA, AVX512F, and baseline AArch64 target requests.
The runtime test wrapper remains baseline x86-64. It gates each extended ISA with
Clang's CPU-support builtin (including OS-enabled state), verifies the fresh
process FP controls and refuses unavailable cases with status 77. A skip is not
execution evidence. These test gates are not production dispatch authority.

Every strict object must retain its two strong issued symbols, separate multiply
and add, no fused operations and no imports beyond `memset`. The wrapper reuses
the unchanged 174-check strict fixture and adds an independent double-precision
oracle for 198 finite rectangular/tail cases (15,444 elements). The chosen
power-of-two values make every intermediate exactly representable in f32.
Neither fixture proves arbitrary mathematical inputs or public runtime behavior.

AArch64 is cross-compiled and inspected, never executed by this x86-only runner.
Clang explicitly overrides the issuer's x86 triple; that experimental override
does not make the production issuer target-generic or authenticate ARM source
execution. Native ARM requires a real native compiler, ABI/FP validation and
source-driver integration. The test fixture contains a conservative native ARM
FPCR gate for that future lane, but that branch is not locally runtime-validated.

Requested target flags and observed register widths are recorded separately.
In particular, permitting AVX512F does not prove LLVM emitted AVX512 instructions.
Strict-no-FMA remains mandatory even when FMA hardware is allowed by the target.

Run the independent classifier controls:

```sh
python3 -B compiler/experiments/multitarget_cpu_v1/test_inspection.py
```

After the integration owner builds the coherent 21.1.8 issuer, use a new output
directory on the designated SSD and an explicit task temporary directory:

```sh
python3 -B compiler/experiments/multitarget_cpu_v1/run.py \
  --issuer /path/to/build/bin/matcore-cpu-gemm-candidate \
  --output /path/to/ssd/builds/cpu-arm/leaf-matrix-01 \
  --tmp-dir /path/to/ssd/tmp/cpu-arm
```

The runner is sequential (one compiler process), refuses an existing output
directory and `/tmp` output/temp paths, and records exact commands, diagnostics,
issuer/LLVM/object hashes, object assembly and results. It never builds LLVM or
changes system packages. Compiler/object tools must all report 21.1.8.
