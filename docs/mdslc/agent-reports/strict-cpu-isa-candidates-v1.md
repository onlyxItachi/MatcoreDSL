# Forced strict CPU ISA candidates v1

Initial implementation-owner record; final independent/hosted qualification and
the canonical merge are recorded below. No performance claim. Initial base: ARM integration
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

## Hosted count-gate correction

The frozen connected implementation at `24d21930e136b93b813993e03bfe0d55e1e3b677`
passed the full local standalone suite: 227/227, zero skips or failures. The closed
installed source/build-inaccessible package control also passed (110.20 seconds).
Independent review subsequently found stale exact test-count guards in the native
workflow. [PR73 run 34604296914](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34604296914)
at that same head confirms three Release jobs built successfully, then refused
the expanded 13/12/12 candidate listings against stale 12/11/11 expectations before
execution. These failed qualification controls are retained as negative evidence.

Fresh configure-only checks with coherent LLVM/Clang/MLIR 21.1.8 confirm 12 closed
candidate tests with OpenBLAS OFF, and 13 with the required 0.3.32 provider ON.
The workflow's exact Debug ASan+UBSan, row-contiguous configuration registers 226
tests and selects 175 with its unchanged combined sanitizer/region expression.
All 43 new `generated_cpu.strict_isa.*` tests and `closed_candidates.strict_isa_guard`
are selected: the previous 131-test sanitizer scope grows by exactly 44 tests.
The separate region counts remain 41/42, private-Value count four and package count
two. Only the three stale count constants change; no test, regex, skip behavior,
execution option or production code changes. Configure/listing validation is not
a new hosted or fully instrumented execution claim; fresh PR checks remain required.

## Qualified canonical checkpoint

Normal [PR #73](https://github.com/onlyxItachi/MatcoreDSL/pull/73) merge:
`1c8f534bba81760aaa211f13b0695254c1d93e60`.
Actual parents: canonical `7a3033063dbc91c8bb4cf57aab93ae2c7e88caa4` and reviewed
implementation `665cf3756c58196b5f58acc2f784e92f7b94b52c`.
Hosted specimen `17734ff29dfd884cab30e0448f718ac1a9d7abd0` had parents `8157eab`
and `665cf37`, tree `3f698d6197fd2ba0e9cb688b76ed32c0dfbe768e`.
Final canonical differs from that implementation only in the prior ARM operator
checkpoint/evidence Markdown files; its compiler subtree is identical:
`0b4657000aaf0106ce85b4a59067802a364c001c`.

[Native hosted run 34605270373](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34605270373)
passed all eight jobs at the exact corrected head:

| Configuration | Actually passed | Explicit skips |
| --- | ---: | ---: |
| Release, MLIR OFF, OpenBLAS OFF | 82 | 0 |
| Release, MLIR OFF, OpenBLAS ON | 82 | 1 |
| Release, scalar MLIR, OpenBLAS OFF | 214 | 14 |
| Release, MLIR, OpenBLAS ON | 230 | 0 |
| Release, row-contiguous, OpenBLAS OFF | 229 | 0 |
| Debug, MLIR ON | 215 | 14 |
| Global ASan + UBSan affected scope | 161 | 14 |
| TSan runtime scope | 4 | 0 |

Each group of fourteen is thirteen strict AVX512F hardware/OS-unavailability
skips and one legacy packed-AVX512 effective-runtime-eligibility skip. The latter
combines hardware, OS and compiler facts; its log does not isolate the failing
subpredicate. None is counted as execution. The thirteen ISA entries cover four
normal, six instrumented, and three source/ownership tests. Separate no-skip
Release jobs actually execute those AVX512F paths; local leaf-ASan evidence above
is also separate from a global sanitizer runner that lacked the needed state.

Production installed checks passed, including Result 37, candidate 247255 on the
row-contiguous Release runner (217029 on the scalar runner), private Value 85 and
32000 ownership cycles. These differing counts follow actual availability, not
a change to the candidate's mathematical contract.

[Native ARM run 34605270306](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34605270306)
passed scalar **97/97** (751.07 s) and row **99/99** (736.21 s), no skips.
Installed source/build-inaccessible execution passed in 178.97 s and 171.49 s.
Four added inspection/facts/guard tests explain the increase from PR #71's 93/95;
this is not AVX execution on ARM. Artifacts: scalar ID `10266982744`, digest
`feaf5ad9095ea2ef653e6ec78aa0d7c33ac65c772e1e7b5b72b81c531a696b20`;
row ID `10266647856`, digest
`36d5bf3d51828f3644295f7892ac422bb85a1c16169b77ffe1731ed085645cec`.
[Windows 34605270285](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34605270285),
[legacy CI 34605270280](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34605270280)
and [hygiene 34605270284](https://github.com/onlyxItachi/MatcoreDSL/actions/runs/34605270284)
also passed. Redundant cancelled push runs are not qualification evidence.

Final independent adversarial review accepted the exact head after inspecting
the fixed LLVM-only refinement, unchanged pre-LLVM stages/default behavior,
complete feature/XSTATE guards, strict separate arithmetic, actual vector
artifacts, wide ASan controls, private ABI and source/leaf ownership. Independent
replays passed 102 fact checks and 1086 synthetic guard checks, separately from
303 real direct-leaf checks per ISA. No performance, all-AVX-subset, AMX, dtype, ARM-ISA, public-wire or
automatic-selection expansion is claimed.
