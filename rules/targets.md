# Target Rule

- MatCore must preserve target support for `x86` with `AVX2` and `AVX-512`.
- MatCore must preserve target support for `ARM`.
- MatCore must preserve target support for `NVPTX`.
- MatCore must preserve target support for `AMDGCN`.
- MatCore must preserve target support for `NPU` and `TPU` dialect routes.
- `x86` CPU execution is the initial runnable backend.
- `ARM`, `NVPTX`, `AMDGCN`, `NPU`, and `TPU` must remain explicit target routes for later lowering.
- Do not silently collapse unsupported targets into the CPU path.

