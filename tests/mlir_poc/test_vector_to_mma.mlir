// MW-0 Test B: Prove that vector.contract (16x8x16 f16) lowers to
// nvgpu.mma.sync via ConvertVectorToGPUPass(useNvGpu=true).
//
// Run: mlir-opt-18 --convert-vector-to-gpu="use-nvgpu" test_vector_to_mma.mlir

func.func @test_vector_contract_to_mma(
    %A: vector<16x16xf16>,
    %B: vector<16x8xf16>,
    %C: vector<16x8xf16>) -> vector<16x8xf16> {
  // vector.contract that should map to mma.sync m16n8k16
  %result = vector.contract {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,  // A: [M, K]
      affine_map<(m, n, k) -> (k, n)>,  // B: [K, N]
      affine_map<(m, n, k) -> (m, n)>   // C: [M, N]
    ],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
  } %A, %B, %C : vector<16x16xf16>, vector<16x8xf16> into vector<16x8xf16>
  return %result : vector<16x8xf16>
}
