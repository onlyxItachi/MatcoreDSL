# Milestone 5 AVX-512 and typed-reference lane

Date: 2026-07-22

Base: `e4dc0affff6c540a65435ba25c5cefa4d69cb562`

Branch: `mdslc/m5-avx512-types`

## Ownership and delivered commits

This lane owned only new packed AVX-512 implementation/test files, internal
BF16/INT8 reference implementation/test files, the minimum Matcore IR v1 dtype
verification changes, and this report. It did not modify CMake, the public C
ABI, planner/platform records, or shared runtime dispatch.

- `fe099f6e37502edcf099740bd6f3059a0051e0d3` — typed BF16 and INT8 IR and
  reference semantics.
- `e4bc33929c7402da87bd1aa996cbd3fe833b9696` — packed AVX-512 F32 engine and
  exact artifact test.
- `bc76840466fdfa5983ff62a19f2618c94fc7db0f` — explicit OSXSAVE/XCR0
  execution guard.

## AVX-512 backend contract

`cpu_packed_avx512.h` adds build/runtime availability, workspace query,
prepacked-B query/preparation, transient execution, and prepacked-B execution
entry points. It deliberately reuses the packed GEMM v1 MC=128, NC=256,
KC=256, MR=4, NR=16, 64-byte workspace alignment, B-panel layout, and
`CpuPackedBViewV1`. Tests prepare B with AVX2 and execute with AVX-512, then do
the reverse, proving the format is not ISA-private.

The isolated C-linkage inspection symbol is:

```text
matcore_cpu_packed_avx512_4x16_microkernel_f32_v1
```

Only that function is compiled with `target("avx512f,fma")`; the translation
unit and generic runtime remain baseline x86-64. The microkernel holds four
rows by sixteen columns in four ZMM accumulators, consumes the existing packed
panels, and uses an aligned 4x16 edge tile for complete M/N tails. Packed B
loads are 64-byte aligned. Boundary paths do not read or write beyond the
declared matrices.

Direct execution is fail-closed. Runtime usability requires all of:

- a compiled x86-64 Clang/GNU function-targeted implementation;
- compiler runtime discovery of AVX-512F and FMA;
- CPUID OSXSAVE;
- XCR0 bits 1, 2, 5, 6, and 7 (SSE, YMM, opmask, ZMM-high-256, high-ZMM).

Capability model v2 must remain the authoritative planner gate. The local
check prevents callers from bypassing it by invoking the backend directly.

## Typed executable reference semantics

Matcore IR v1 now verifies exactly these host GEMM dtype contracts:

```text
f32  x f32  -> f32, accumulation f32, requirement f32_arithmetic
bf16 x bf16 -> f32, accumulation f32, requirement f32_arithmetic
i8   x i8   -> i32, accumulation i32, requirement i32_arithmetic
```

BF16 input conversion uses round-to-nearest, ties-to-even. NaNs become a
deterministic quiet BF16 NaN while preserving the sign. BF16 reference GEMM
widens operands exactly and applies one binary32 `fma` per K element in
increasing-K order. I8 products are exact signed products; accumulation and
output are explicitly modulo 2^32, avoiding C++ signed-overflow undefined
behavior. Both references are row-major, allocation-free, validate all
dimensions/pointers/spans first, reject output/input overlap, and leave output
unchanged on rejection.

The deterministic JSON parser/serializer recognizes `i32_arithmetic`.
BF16/F32 and I8/I32 modules round-trip through JSON. The existing v1-to-v0
projection remains explicitly F32-only and rejects both new contracts as
lossy.

No public C ABI was added in this lane. The integration owner must add any
versioned ABI only after reviewing descriptor and compatibility requirements.

## Validation evidence

Validation host:

```text
AMD Ryzen AI 9 HX 370 w/ Radeon 890M
x86_64; AVX2/FMA; AVX-512F/DQ/BW/VL/VNNI/BF16 reported
Clang 21.1.8
```

Manual Release compilation used `clang++-21 -std=c++20 -O2` with
`-Wall -Wextra -Wpedantic -Werror` and linked the existing packed backend
contract, AVX2 implementation (for interoperability), new AVX-512
implementation, and focused tests.

Results:

```text
native packed AVX-512/FMA GEMM: all tests passed
BF16/F32 and I8/I32 reference semantics: all tests passed
AVX-512 artifact verified: zmm=21 packed_fma=4
ELF 64-bit LSB relocatable, x86-64, not stripped
```

The exact runtime-usability function disassembly contains CPUID, OSXSAVE bit
27 testing, `xgetbv`, and an `0xe6` XCR0 mask. The checked execution test then
ran the physical AVX-512 microkernel successfully; this is runtime validation
on this host, not compile-only evidence.

Focused shapes include tiny, square, rectangular, K/M/N tails, KC/MC/NC
boundaries, 4-byte-aligned/misaligned-to-64 tensors, seeded randomized cases,
prepacked reuse, and cross-ISA prepacked interoperability. Results were checked
against an independent double-precision oracle with guard regions.

Additional results:

```text
Debug -O0 AVX-512 and typed-reference tests: passed
ASan+UBSan AVX-512 and typed-reference tests: passed
TSan AVX-512 and typed-reference tests: passed
Matcore IR v1 focused CTest: 1/1 passed
Bootstrap-mode standalone regression CTest: 22/22 passed
git diff --check: passed
```

Temporary binaries and objects stayed under `/tmp` and were not tracked.

## Integration requirements and limitations

- Add the new sources/tests/artifact checker to shared CMake in an integration
  commit; this lane intentionally did not touch shared build ownership.
- Register `cpu.native-packed.avx512-fma.f32.v1` only when capability v2 says
  hardware, OS state, compiler body, implementation, and runtime validation
  are legal.
- Generalize the existing shared `isa_unavailable` status text, which currently
  names AVX2 specifically although the status enum is shared.
- Public BF16/INT8 C descriptors and wrappers are not implemented here.
- AVX-512 BF16/VNNI optimized paths are not implemented; only their portable
  typed reference semantics exist.
- AMX is neither implemented nor executed.
- No AVX-512 performance advantage is claimed. Planner promotion requires the
  common benchmark contract and measured evidence.
- Windows and non-x86 build validation remains for the dedicated portability
  phase; non-x86 builds compile the AVX-512 surface as unavailable and never
  execute the kernel.
