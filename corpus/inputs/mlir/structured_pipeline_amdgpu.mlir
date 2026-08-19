// Diagnostic Case: structured_pipeline_amdgpu.mlir
// Purpose: Wave-level AMD amdgpu.mfma matrix instruction pattern.
module {
  func.func @amdgpu_mfma_32x32x8_f16(
      %a: vector<4xf16>,
      %b: vector<4xf16>,
      %c: vector<16xf32>) -> vector<16xf32> {
    %res = amdgpu.mfma %a * %b + %c {
      m = 32 : i32,
      n = 32 : i32,
      k = 8 : i32,
      blocks = 1 : i32
    } : (vector<4xf16>, vector<4xf16>, vector<16xf32>) -> vector<16xf32>
    return %res : vector<16xf32>
  }
}
