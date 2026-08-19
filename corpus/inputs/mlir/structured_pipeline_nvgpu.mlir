// Diagnostic Case: structured_pipeline_nvgpu.mlir
// Purpose: Warp-level NVIDIA mma.sync / ldmatrix staging pattern.
module {
  func.func @nvgpu_mma_sync_m16n8k16(
      %a: vector<4x2xf16>,
      %b: vector<2x2xf16>,
      %c: vector<2x2xf32>) -> vector<2x2xf32> {
    %res = nvgpu.mma.sync (%a, %b, %c) {
      mmaShape = [16, 8, 16]
    } : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
    return %res : vector<2x2xf32>
  }
}
