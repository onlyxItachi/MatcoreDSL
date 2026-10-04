# GPU pair source work-cap falsifier

Fixture checkpoint: `7acacd5ce8b328361f1bf8e3120b793a45e1dab3`, based on root
composition `8dd9dd0482b74cf90f2f786b4e914879c3c13b9a`. Only the new
`compiler/tests/generated_gpu_fused_pair/work_cap.mdsl` changes test code.

This separate fixture keeps the existing 22-case same-source oracle unchanged.
All-one inputs produce exactly 2048 at every output element. M=64,K=32,N=64,P=32
is the exact combined 2^18 boundary and must succeed. M=65 gives 266240 combined
work, while each original GEMM has only 133120 work and a legal output extent.
The none route must succeed; the combined route must reject at frontier7 after
guard-only frontier6, retaining effect5, its publication/owning observation and
every old output sentinel. The host expectation macro changes no region body.

The fixture checks source location, old owning observation lifetime, input
immutability, canaries and complete hostile caller fenv/MXCSR/errno preservation.
The direct region call is in an ordinary no-math wrapper before FENV_ACCESS ON,
matching the existing exact multi-source call ABI. `main()` returns explicitly.

With root's one-job reservation, both commands returned 0 with no diagnostics:

```sh
env TMPDIR=/var/tmp/mdslc-region-correctness.bvy8fe79 /usr/bin/clang++-21 \
  -x c++ -std=c++20 -fsyntax-only -fno-fast-math -ffp-contract=off -frounding-math \
  -I compiler/include compiler/tests/generated_gpu_fused_pair/work_cap.mdsl
env TMPDIR=/var/tmp/mdslc-region-correctness.bvy8fe79 /usr/bin/clang++-21 \
  -x c++ -std=c++20 -fsyntax-only -fno-fast-math -ffp-contract=off -frounding-math \
  -DMDSLC_EXPECT_COMBINED_WORK_REFUSAL=1 -I compiler/include \
  compiler/tests/generated_gpu_fused_pair/work_cap.mdsl
```

These are ordinary C++ syntax checks only: no link, source admission, generated
GPU execution or installed-package qualification occurred in this author lane.
Root owns runner/producer-inaccessible wiring and subsequent qualification.
No performance or equal-outcome claim applies to the over-cap case.
