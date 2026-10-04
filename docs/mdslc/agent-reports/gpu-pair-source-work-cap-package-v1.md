# GPU pair work-cap producer-inaccessible wiring

Owner: forwarding implementation agent. Isolated branch
`test/mdslc-gpu-pair-work-cap-package-v1`, based on composed
`8afff36f061d7af8f1c13dc1c6be8e657d4880e9`. File-only wiring commit:
`cf5d807e6e1fc4fbd434439bd657a086422e1001`.

Only `compiler/tests/closed_driver/installed_source_inaccessible.py` changes.
The runner stages external source/program copies of `work_cap.mdsl` before
removing the exact disposable producer source/build trees. Private host
`MDSLC_EXPECT_COMBINED_WORK_REFUSAL` is 0 for `none`, 1 for `strict-fused-pair`;
the admitted region body is identical. Program copies rename the explicitly
returning `main` to `math_main` and reuse the ordinary external host wrapper.

For each enabled NVVM/ROCDL candidate, both routes and modes compile using the
relocated installed driver after producer removal. The runner checks the actual
undefined `gemmStrictFusedPair` reference is present only in the fused mode,
requires exit zero, and matches positive-check/zero-failure summaries: boundary
executed in both modes, over-cap executed for `none`, refused at f2 for fused.
The fixture checks exact failure source/frontiers, retained earlier publication
and owning observation, unchanged old output, caller state, and the legal M64
boundary. Cap summaries intentionally are not compared for parity. The original
22-case same-source oracle and its comparison remain unchanged and independent.

Author validation was syntax only (exit zero), with no module/script execution:

```sh
python3 -B -c 'from pathlib import Path; p = Path("compiler/tests/closed_driver/installed_source_inaccessible.py"); text = p.read_text(); compile(text, str(p), "exec", dont_inherit=True, optimize=0); compile(text, str(p), "exec", dont_inherit=True, optimize=2); print("PASS Python syntax at optimize=0 and optimize=2; no execution")'
git diff --check
```

No compiler, link, GPU dispatch, nested package run, or physical acceptance is
claimed here. Integration owner must run the genuine producer-inaccessible
package gate at the final frozen composed checkpoint.
