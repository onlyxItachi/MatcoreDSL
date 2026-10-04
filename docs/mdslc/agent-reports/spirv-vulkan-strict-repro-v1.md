# SPIR-V strict-f32 research: author qualification and reproduction notes

Code checkpoint: `706e44e85ecc9f6bbb77fb520d4859d701be7ab3`, based on
`f55b86e5d0fbeaa8d10d5857bc2bf2ce71168df4`. The integration owner's observation
report at `97a9b1ef68b6ebd18b3e2c899d912318c424cadb` is accurate against the saved
records. This addendum changes documentation only. Independent final review
resumed after the interruption and remains a separate gate.

## Exact scope and controls

This is one static M=2,K=3,N=4 nonfused research specimen, not a product target,
source-authenticated helper, LLVM SPIR-V backend, runtime adapter or performance
result. The old Metal research commit `564bef047` supplied the upstream lowering
recipe and oracle starting point, not shader arithmetic or production authority.
The arithmetic, indexing and control flow remain compiler-produced. The control
patch adds only capabilities, the float-controls extension, execution modes and
NoContraction decorations; stripping these additions exactly restores the
original assembly body. SPIR-V validation and post-assembly control checks both
ran successfully.

The probe requires Vulkan 1.2 at loader and selected-device boundaries, queries
the three actual f32 preservation/RTE properties and explicitly enables no
optional device feature or extension. This scalar-f32 route uses the core-1.2
float-controls facility; it does not infer support from device identity alone.
The binary gate rejects RelaxedPrecision, FPRoundingMode, FPFastMathMode and
missing arithmetic NoContraction. The importance of these controls is described
by the [SPIR-V float-controls specification](https://github.khronos.org/SPIRV-Registry/extensions/KHR/SPV_KHR_float_controls.html)
and the [Vulkan float-controls properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFloatControlsProperties.html).

Before any Vulkan API call, nine host adversarial assertions execute and all
80 expected result bit patterns are frozen. The order discriminator is
`[2^24, 1, -2^24]`: increasing-K gives 0 and the independently checked reverse
order gives 1. The host uses separately rounded volatile product/add operations,
RTE and cleared FTZ/DAZ; FMA, subnormal and rounding discriminators are asserted.
Non-NaNs compare exact bits; NaNs compare classification only. The signed-zero
case tests negative-zero inputs accumulated from +0, whose expected output is
+0; it is not a proof of every negative-zero output behavior.

HOST-to-COMPUTE, fill-to-GEMM and COMPUTE-to-HOST buffer barriers are explicit.
Descriptor-offset alignment is independent of nonCoherentAtomSize: noncoherent
maintenance uses offset zero and VK_WHOLE_SIZE over the mapped allocation.
Normal cleanup follows checked fence completion. Submit/wait uncertainty takes
research process fail-stop without freeing possibly live resources; this is
not a reusable runtime quarantine or recoverable-failure qualification. See
the [Vulkan buffer barrier contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferMemoryBarrier.html).

## Saved outcomes, kept separate

- The final AMD RADV run returned 0: 10 cases, 20 dispatches, 10 checked
  completions, 80 comparisons with 0 mismatches, 1280 canary checks and 180
  immutable-input checks. Host FP controls were unchanged. All 30 allocations
  were coherent; the noncoherent branch has no physical qualification.
- Explicit NVIDIA and llvmpipe selections each returned 78 with
  REFUSED_FLOAT_CONTROLS and no dispatch. Their wrappers reported the expected
  refusal, not an executed arithmetic pass or a fallback.
- The final reproduction records 22 tool commands, all return code 0, and
  13 pure metadata/control rejection controls. Three additional direct host
  gates rejected missing fill modes, missing GEMM modes and missing GEMM
  NoContraction before Vulkan discovery; each invocation returned 1 and saved
  PROBE_ERROR. These are rejection checks, not additional physical math cases.
- The earlier unmodified default modules were SPIR-V 1.0, so their initial
  rejection only established the version pin. The final negative fixtures
  reassemble the unchanged default body as SPIR-V 1.3, ensuring missing-control
  rejection is not masked by a version mismatch.

Every observation still says `strict_contract_qualified: false`. The AMD result
separately says `bounded_strict_shader_sample_matched: true`. Repeated runs of
the same ten cases are not additional unique coverage. No hosted CI, package,
dynamic-shape, fusion, residency or production failure-law claim is made.

## Reproduction and artifact binding

Evidence root:
`/home/hamza-usta/mdslc-work/region-optimization-v1/builds/spirv-strict.9jpv1U7G`.
The following commands show the actual saved directories, run from the research
worktree. Re-execution requires a coordinated compiler slot and exclusive AMD
device window; the addendum itself performed only read-only record/hash checks.

```sh
SPIRV_EVIDENCE=/home/hamza-usta/mdslc-work/region-optimization-v1/builds/spirv-strict.9jpv1U7G
env TMPDIR=/var/tmp/mdslc-region-correctness.bvy8fe79 \
  python3 compiler/experiments/spirv_strict_f32_v1/reproduce.py \
  --mlir-bin /home/hamza-usta/.local/toolchains/mlir-21.1.8-6ubuntu1/usr/lib/llvm-21/bin \
  --spirv-bin /home/hamza-usta/mdslc-work/multitarget-v1/builds/metal/tool-root/usr/bin \
  --output "$SPIRV_EVIDENCE/artifacts-final"
env TMPDIR=/var/tmp/mdslc-region-correctness.bvy8fe79 \
  python3 compiler/experiments/spirv_strict_f32_v1/probe.py build \
  --clangxx /usr/bin/clang++-21 \
  --link-lock /home/hamza-usta/mdslc-work/region-optimization-v1/builds/heavy-link.lock \
  --output "$SPIRV_EVIDENCE/final"
env TMPDIR=/var/tmp/mdslc-region-correctness.bvy8fe79 \
  python3 compiler/experiments/spirv_strict_f32_v1/probe.py run \
  --output "$SPIRV_EVIDENCE/final" --artifacts "$SPIRV_EVIDENCE/artifacts" --device 1
```

The final host was also run with `--device inventory`, `--device 0` and
`--device 2`; these saved records are `final/execution-{inventory,0,2}.json`.
The physical execution used `artifacts`, not `artifacts-final`. The latter was
generated subsequently to add the same-version negative fixtures. Both
directories have byte-identical strict fill/GEMM modules with the hashes below;
the physical argument paths are not rewritten in the saved evidence.

| Identity | SHA256 |
|---|---|
| Checked-in `probe.cpp` | `df56ec5667d0a1e27768225040c46ba5c93124a1cd6c3f2fbf7ca35a15d769b9` |
| `final/probe` | `26edfe3bf43979ec5990e583841b60bf63fe2dacad99ebd51f30b4bff48d1969` |
| `final/probe-build.json` | `f1279e9afefa5c091204b580b61892fba9f207a314b917edf9b2456bd53fe18a` |
| `artifacts-final/manifest.json` | `41da45d7965bf4494fe2d1d94fcc68916d9ed04385a10788cdafc685354adeb4` |
| Strict fill SPIR-V, both directories | `1bfc915dc5c018f2cf6474aade818d8965ef742582ead8c1a60b4175f0811edb` |
| Strict GEMM SPIR-V, both directories | `139dcb3930b858104b6a016530a16401b3c73df125e164cc2399f96d1721f53b` |
| `final/execution-1.json` | `d20b5758a2a3ea5d18637f92ff518117672e6a11e34d641d2ee994b5c4cfde12` |
| `final/result-1.json` | `ccb87420e4ac805c824147e604e3a3774538cd470fd0009732a58527340d1bb1` |

The build record binds exact Clang 21.1.8, full argv, source and executable
hashes, and empty captured stdout/stderr with return code 0. Execution records
bind full argv, shader/source/executable hashes, actual child status and captured
stdout/stderr; the final AMD and refusal records have empty stderr. The manifest
binds MLIR tools, SPIRV-Tools and every generated stage. The observation report
retains the refusal-result hashes. Raw artifacts remain external, not committed.

Final independent lineage/control/oracle/descriptor/barrier review is the next
research gate. Its acceptance would still not issue source execution authority.
