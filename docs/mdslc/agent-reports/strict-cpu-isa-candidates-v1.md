# Forced strict CPU ISA candidates v1

Implementation-owner evidence, not independent approval, hosted qualification,
performance evidence or permission to merge. Initial base: ARM integration
`e1fe9df4df8d3f625754ef762e06ba25fd67ec21`; ARM repair `8157eab` is a stack dependency.

## Closed realization and authority

Three explicitly forced choices are connected through authenticated source,
paired closed Program semantics, the existing compiler-built strict row-contiguous
MLIR recipe, generated LLVM, a privately bound candidate DSO and the checked
Session: `generated-strict-avx`, `generated-strict-avx2`, and
`generated-strict-avx512f`. Automatic selection and the default generated schedule
are unchanged. There are no cost, tuning, provider or performance policy changes.

`StrictCpuIsaV1` is compiler-private and separate from the public/versioned V2
capability record, which has no AVX wire bit. No V2 bit is invented or reinterpreted.
The issuer accepts only fixed enum refinements on Linux x86-64 with the existing
row-contiguous schedule. Arbitrary triples/features, unknown enums, scalar-ISA,
ARM-ISA and reassociate-ISA requests are refused. All five pre-LLVM stage strings
remain identical to the existing row recipe. The only added LLVM realization
facts are distinct leaf/wrapper names, fixed target CPU/features and preferred
vector width. Default scalar LLVM remains byte-identical to the baseline issuer.

Each object has a distinct `__matcore_strict_gemm_f32_{avx,avx2,avx512f}_v1` and
corresponding MLIR C interface. Runtime reports distinguish all three actual
implementations. New private enum values append existing values; future GPU/CPU
integration must preserve whichever private values have already become canonical.
The exact private DSO export contract is unchanged; leaf names and discovery
remain local. Real source host definitions of both private leaf and wrapper
symbols cannot replace generated execution, while their ordinary host calls work.

## Machine facts are not arithmetic permission

Direct CPUID discovery checks maximum basic leaf availability. XGETBV is executed
only after hardware XSAVE, OSXSAVE and AVX. The pure classifier requires known raw
facts, Linux x86, the x86-64 baseline (including MMX), LLVM's SSE predecessor chain,
and XCR0 XMM/YMM state. AVX2 additionally requires CPUID leaf7 AVX2. AVX512F also
requires AVX512F, FMA, F16C and XCR0 opmask/ZMM state (`0xe6`). Unknown discovery,
missing predecessor bits and incomplete OS state fail closed. Discovery does not
run a legacy packed-kernel selftest or claim runtime validation of unused features.

These dependencies follow pinned [LLVM 21.1.8 X86 target definitions](https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/Target/X86/X86.td).
In particular `+avx512f` implies FMA/F16C availability. `-mno-fma` would disable
that ISA and is not used. Strict semantic authority still permits only increasing
K, separately rounded unflagged f32 multiply/add and gradual underflow. No fast
math or contraction flags are added. All variants use `-ffp-contract=off`.
This is not support for every AVX subextension, f16 arithmetic, fused strict math
or reassociation; the bounded recipe uses f32 arithmetic/broadcast/movement only.

Runtime capability checks precede allocation and empty/zero-K shortcuts. Failed
forced choices cannot fall back, invoke a leaf, issue a Value or erase prior
publication/observation. Existing exact FP environment restoration, aliases,
source authentication, ownership and failure-order machinery is retained.

## Mechanical and physical evidence before freeze

Local host: AMD Ryzen AI 9 HX 370, Linux x86-64. LLVM/Clang/MLIR are coherent
21.1.8; OpenBLAS is OFF. Builds use the dedicated SSD tree, two compile jobs and
the shared link lock. The initial connected build passed all 365 build actions;
subsequent focused rebuilds include the extended checks.

Build-time object checks require exact symbols, ELF machine, IR digest binding,
bounded imports, a closed vector opcode set and separate arithmetic. Normal AVX
objects contain actual YMM multiply/add. AVX2 additionally contains register-source
`vbroadcastss`, an AVX2-exclusive operation. AVX512F contains actual ZMM multiply/add
and narrower/scalar tails. No fused instructions occur. A deliberate test-only
contract-enabled LLVM corruption produces real ZMM FMA and is rejected by the
object arithmetic guard; it is not accepted by any source or primitive issuer.

Normal object SHA256 values:

| ISA | SHA256 |
| --- | --- |
| AVX | `0c92031ea61918442ce2e842c1b0eecf4fa0b581ce0faf600f2a985056acf40a` |
| AVX2 | `4a5f5db48e844015ddbd4f672a463bd3956aaee97edaceddd7e76775dd7c007d` |
| AVX512F | `e55c29743df6a5a6276e384cfec7378b843359bdfffa1850c6adb5fa312833ea` |

The focused matrix passed 48/48 tests with no skips; four added FMA-object and
source leaf-ownership controls then passed 4/4. All three variants executed on
physical hardware, both normal and ASan-instrumented generated leaves, and through
real source output/alias/failure fixtures. Each direct leaf run checks 303 conditions,
including a 129-column full-width FMA-distinguishing oracle; each independent
random run checks 85383 results/canaries and each input-overlap run checks 86898
conditions. Deliberate output corruption is caught. ASan negative controls report
the generated top frame and actual READ32 for AVX/AVX2 or READ64 for AVX512F.

Issuer checks: 45; pure capability checks: 102; synthetic unavailable/late-failure
adapter checks: 1086. The production registry matrix checks 247255 conditions,
including distinct actual reports and wide execution. Private DSO tests confirm
the unchanged exact export set. Test-only synthetic discovery and leaf substitutes
are separately linked, never production hooks or physical execution evidence.

Full frozen standalone/package qualification and independent final review remain
required after this implementation checkpoint. Native hosted ARM qualification
belongs to PR71; this lane adds no ARM ISA variant and preserves the ARM baseline.
