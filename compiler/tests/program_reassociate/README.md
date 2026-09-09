# Shared-parser multi-source reassociate composition gate

This is an independent test-only fixture for the final PR #66 / PR #67 union.
It needs a real composed `mdslc-region` containing both `--program` and the
`generated-reassociate` policy. A missing policy is a test failure, not a skip or
an acceptable old-driver refusal. No production or existing tests are changed.

Three genuine source translation units are compiled together: `host.cpp`,
`first.mdsl` and `second.mdsl`; `api.h` contains declarations only. Both regions
retain their physical main-file definitions. The first executes one complete
4x8 output tile with K=2 and explicit reassociate permission. Every output
distinguishes the selected FMA implementation (`0x337ffffe` f32 bits) from strict
separate multiply/add (`+0`). The second has a strict 4x8-by-8x3 GEMM.

Ordinary host code observes the first Result, keeps its owning observation after
Result destruction, mutates the first publication, and then calls the second TU.
Forced generated-reassociate must reach that boundary and return checked
`candidate_incompatible` at second-region frontier 3, completed 2, effect 0,
`second.mdsl:6:13`, with no second publication/observation. The prior owning
observation and ordinary host mutation must survive. Generated-strict must
produce strict first arithmetic and execute the second GEMM using the intervening
host mutation. Both policies preserve full source arrays and output canaries.

Provisional statically counted expectations are 152/164 positive assertions and
two deliberately failing numerical/refusal-identity swaps. These counts must be
confirmed on the composed driver before any execution claim. Expected wrong-label
failures are exactly 143/155; no crash or sanitizer diagnostic qualifies.

```sh
python3 compiler/tests/program_reassociate/run.py \
  --driver /absolute/composed/build/bin/mdslc-region \
  --expected-driver-sha256 ACTUAL_SHA256 --output /fresh/evidence/path
```

The actual source installation owns the sanitizer profile. The runner does not
inject sanitizer flags, accepts no imported object substitute, preserves hashes
of every source/driver/runtime and produced executable around each command,
requires empty positive stderr and checks direct private/canonical ELF ownership.
Only the exact diagnostic-free runtime hardware-unavailable exit 77 becomes SKIP;
CTest registration must use `SKIP_RETURN_CODE 77`. CTest/CI registration belongs
to the multi-source integration owner and is intentionally absent here.
Public Result is not candidate identity telemetry; actual numerical differences
and the owning compiler/DSO contract are distinct evidence. No timing, installed
SDK, inaccessible-source or cross-region optimization claim follows.
