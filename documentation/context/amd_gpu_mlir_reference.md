# AMD GPU backend in MLIR 18.x: matrix-ops reference

## Bottom line

**Upstream MLIR 18.x has the AMD-specific matrix intrinsics and ROCDL lowerings, but it does _not_ provide a turnkey, production-grade `linalg.matmul -> MFMA/WMMA` pipeline.**

For production GEMM on AMD in the MLIR 18 era, the usual pattern is:
- use **upstream MLIR** for final AMD-specific ops/lowerings, and
- use **rocMLIR or IREE/custom transform pipelines** for tiling, scheduling, vectorization, and tuning.

---

## 1) Which MLIR ops to use

### CDNA / Instinct (MI250 = `gfx90a`, MI300 = `gfx942`): use `amdgpu.mfma`

`amdgpu.mfma` is the upstream MLIR 18 wrapper for CDNA matrix-core instructions.

Use it when you already know the wave-level matrix shape and datatype. Key attributes:
- `m`, `n`, `k`, `blocks`
- `cbsz`, `abid`, `blgp` for lane/block permutation behavior
- `reducePrecision` for the reduced-precision f32 MFMA variants

In MLIR 18.x source, `amdgpu.mfma` covers:
- f32 accumulation from f32/f16/bf16/fp8 inputs
- i32 accumulation from int8 inputs
- f64 MFMA on newer CDNA parts

Practical MFMA micro-op shapes exposed in 18.x include:
- **FP32:** `16x16x4`, `32x32x2` (plus reduced-precision variants)
- **FP16:** `16x16x16`, `32x32x8`
- **BF16:** `16x16x8`, `32x32x4`; on `gfx90a+`, also the `1k` forms such as `16x16x16bf16_1k`
- **FP8/BF8 (MI300/gfx94x only):** `16x16x32`, `32x32x16`

### RDNA3 (`gfx11`): use `amdgpu.wmma`

`amdgpu.wmma` is the upstream MLIR 18 wrapper for RDNA3 WMMA.

Important limits in 18.x:
- **WMMA is RDNA3/gfx11 only**
- fixed **`16x16x16`** matrix op shape
- wrapper-level input types are effectively **f16 / bf16 / int8** in 18.x
- ROCDL exposes more low-level variants than the 18.x AMDGPU wrapper does

Important caveat:
- for **f16->f16** or **bf16->bf16** WMMA, the result vector contains only **8 valid values**, selected via `subwordOffset`

### Shared-memory/LDS control in 18.x

Upstream 18.x AMDGPU is still fairly thin. The main AMD-specific LDS op you should expect in 18.x is:
- `amdgpu.lds_barrier`

Do **not** assume newer trunk-only AMDGPU ops exist in 18.x (for example later `scaled_*`, `sparse_*`, or async LDS/tensor-load wrappers).

---

## 2) AMD-specific lowering passes in MLIR 18

The key upstream passes are:

- **`convert-amdgpu-to-rocdl{chipset=...}`**  
  Lowers `amdgpu.mfma` / `amdgpu.wmma` to ROCDL intrinsics and rejects unsupported chip/shape/type combinations.

- **`convert-gpu-to-rocdl{chipset=..., runtime=HIP|OpenCL, index-bitwidth=...}`**  
  Lowers generic `gpu` ops to ROCDL:
  - thread/block ids -> `rocdl.workitem.*` / `rocdl.workgroup.*`
  - lane id -> ROCDL mbcnt sequence
  - workgroup memory -> **ROCDL address space 3**
  - math ops -> **OCML** calls (`__ocml_*`)

Also typically used around them:
- `gpu-kernel-outlining`
- bufferization / memref lowering
- `convert-vector-to-llvm`
- `convert-memref-to-llvm`
- `reconcile-unrealized-casts`

If using MLIR's built-in offloading flow, pair device lowering with:
- `rocdl.attach-target`
- `gpu.module-to-binary`

That path produces an AMD code object (HSACO) for ROCm/HIP runtime loading.

---

## 3) How to lower `linalg.matmul` to AMD GPU in practice

## What upstream 18.x can do

Upstream 18.x gives you the **building blocks**:
- `linalg.matmul`
- transform/linalg tiling
- promotion to workgroup memory
- `gpu` kernel outlining
- `amdgpu.mfma` / `amdgpu.wmma`
- ROCDL lowering

## What it does **not** give you

It does **not** give you a polished stock pipeline that automatically turns generic `linalg.matmul` into tuned MFMA/WMMA kernels for MI250/MI300.

So for production use, do one of these:
1. **custom upstream-MLIR pipeline** that explicitly rewrites the innermost tiles to `amdgpu.mfma`, or
2. **rocMLIR / IREE** for the schedule+tuning phase, then upstream-style ROCDL lowering underneath.

## Safe pass-pipeline sketch

A practical upstream-style outline is:

1. **Tile `linalg.matmul` to a workgroup tile** (`Mwg x Nwg x Kwg`)  
2. **Promote A/B tiles to workgroup memory (LDS)**  
3. **Tile again to wave/MFMA tiles**  
4. **Vectorize or explicitly rewrite the innermost tile to `amdgpu.mfma`**  
5. `gpu-kernel-outlining`  
6. inside `gpu.module`:  
   - `convert-gpu-to-rocdl{chipset=gfx90a|gfx942, runtime=HIP, index-bitwidth=32|64}`  
   - `convert-amdgpu-to-rocdl{chipset=gfx90a|gfx942}`  
   - `convert-vector-to-llvm`  
   - `convert-memref-to-llvm`  
   - `reconcile-unrealized-casts`
7. optionally `rocdl.attach-target` + `gpu.module-to-binary`

### Shared memory management

On AMD, workgroup/shared memory is **LDS**.

In the ROCDL path:
- workgroup memory is **address space 3**
- synchronize LDS fills/reads with `gpu.barrier` or `amdgpu.lds_barrier`

A useful LDS sizing formula for matmul staging is:

`bytes = (Mwg*Kchunk + Kchunk*Nwg) * element_size * buffering_factor`

For MI250/MI300, LDS is **64 KiB per CU**, so double-buffered staging must stay within that budget.

Examples:
- fp16/bf16, `Mwg=Nwg=128`, `Kchunk=32`  
  A+B staging = `(128*32 + 32*128) * 2 bytes = 16 KiB`; double-buffered = **32 KiB**
- fp16/bf16, `128x128x64`  
  double-buffered A+B staging = **64 KiB** (already at the limit)

This is why `Kchunk` matters as much as `Mwg/Nwg`.

---

## 4) Tile sizes: what is safe to say for MI250 / MI300

These are **starting points**, not upstream-MLIR-18 defaults.

### Distinguish 3 levels

1. **Instruction shape**: one MFMA/WMMA op (e.g. `16x16x16`)  
2. **Wave tile**: one wavefront's logical output tile  
3. **Workgroup tile**: the outer tile staged through LDS  

### MI250 (`gfx90a`, CDNA2)

Use **MFMA**, not WMMA.

Good starting points:
- **wave tiles:** `16x16` or `32x32`, chosen to match the MFMA family for the datatype
- **workgroup tiles:** start with **`64x64`** or **`128x128`** in MxN
- **K chunks:**
  - fp16/bf16: start with **16** or **32**
  - fp32: start with **4** or **8**

### MI300 (`gfx942`, CDNA3)

Also use **MFMA**.

Good starting points:
- same **`64x64` / `128x128`** workgroup-tile starting range
- same LDS budgeting logic as MI250 (still 64 KiB LDS/CU)
- for **fp8/bf8**, align wave tiles to the CDNA3 MFMA families:
  - `16x16x32`
  - `32x32x16`

### Practical rule

For MI250/MI300 matmul kernels:
- start with **`128x128` workgroup tiles** when occupancy/LDS permit
- fall back to **`64x64`** for smaller problems or higher register/LDS pressure
- map the innermost compute to **`16x16` or `32x32` MFMA wave tiles**

---

## 5) ROCm runtime integration

Be careful about scope:

- **`convert-gpu-to-rocdl` is device-side lowering**
- host-side execution still needs the GPU runtime / wrapper ABI path

What the AMD path looks like:
1. `gpu.module` -> ROCDL -> LLVM AMDGPU IR  
2. attach target (`gfx90a`, `gfx942`, etc.) and compile to HSACO  
3. load/launch through ROCm/HIP runtime

In the upstream ROCDL path, `runtime=HIP` is part of the GPU-to-ROCDL lowering configuration. The pass also lowers math functions to **OCML**.

So the integration story is:
- **device code:** ROCDL/OCML/AMDGPU backend
- **host launch:** MLIR GPU runtime wrappers over ROCm/HIP

---

## 6) How this differs from the NVIDIA path

AMD and NVIDIA are not symmetric in upstream MLIR 18.x.

### NVIDIA (upstream 18.x)
- clearer documented path through **NVVM / nvgpu**
- stronger “official pipeline” feel for tensor-core lowering
- warp assumptions are typically **warp32**

### AMD (upstream 18.x)
- thinner path: mostly **intrinsic wrappers + ROCDL lowering**
- less turnkey for matmul scheduling/tuning
- must think in **wavefronts** (usually **wave64** on MI250/MI300)
- `amdgpu.mfma` is the main matrix-core op for Instinct/CDNA

Practical implication:
- do **not** port NVIDIA warp/tile assumptions directly to AMD
- on AMD, wave64/LDS pressure/occupancy are first-class tuning variables

---

## 7) Known limitations and workarounds in MLIR 18.x

### Limitation: no fully automatic upstream matmul->MFMA path
**Workaround:** use rocMLIR, IREE, or emit `amdgpu.mfma` explicitly after tiling/vectorization.

### Limitation: WMMA is not for MI250/MI300
`amdgpu.wmma` is **gfx11/RDNA3 only**.  
**Workaround:** for MI250/MI300, always target **MFMA**.

### Limitation: AMDGPU wrapper surface is narrower than ROCDL
Example: ROCDL exposes more WMMA variants than `amdgpu.wmma` does in 18.x.  
**Workaround:** if needed, drop lower in the stack to ROCDL/LLVM intrinsics.

### Limitation: chipset validation happens late
Unsupported shape/type combinations are often rejected in `convert-amdgpu-to-rocdl`, not when the op is first created.  
**Workaround:** always compile with the exact `chipset=` (`gfx90a`, `gfx942`, `gfx11xx`) and keep MFMA/WMMA shapes chip-specific.

### Limitation: BF16 lowering has representation caveats
Upstream 18.x AMDGPU-to-ROCDL lowering uses bitcast-style workarounds for BF16 in places.  
**Workaround:** test BF16 kernels per target chip; do not assume a perfectly clean native BF16 path through every stage.

### Limitation: current online AMDGPU docs are newer than 18.x
The current `mlir.llvm.org` AMDGPU/ROCDL docs describe ops added after 18.x.  
**Workaround:** when pinned to MLIR 18, verify features against the **release/18.x source tree**, not just the latest docs site.

---

## Recommended defaults for MatcoreDSL

If the target is **MI250/MI300**:
- prefer **`amdgpu.mfma`**
- treat **WMMA as non-applicable**
- use LDS/workgroup promotion aggressively
- start tuning from:
  - **workgroup tile:** `64x64` or `128x128`
  - **wave tile:** `16x16` or `32x32`
  - **K chunk:** `16/32` for fp16/bf16, `4/8` for fp32, `16/32` for fp8 on MI300
- run `convert-gpu-to-rocdl{runtime=HIP, chipset=gfx90a|gfx942}` and `convert-amdgpu-to-rocdl{chipset=...}`

If you need a **production** end-to-end matmul compiler flow rather than a low-level backend target, prefer **rocMLIR or IREE-style codegen** over stock upstream MLIR 18 alone.

---

## References

- LLVM release/18.x source:
  - `mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPU.td`
  - `mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td`
  - `mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp`
  - `mlir/lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp`
  - `mlir/test/Conversion/GPUToROCDL/gpu-to-rocdl.mlir`
- ROCm hardware specs: MI250=`gfx90a`, MI300X=`gfx942`
- AMD matrix-core references (CDNA/RDNA): ROCm blog + GPUOpen WMMA docs
- rocMLIR and IREE tuning docs for production scheduling heuristics
