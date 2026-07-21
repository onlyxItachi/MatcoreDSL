// MW-0 Test B2: Alternative vector.contract shape that matches
// the mma.sync m16n8k16 fragment layout more precisely.
// A: [16, 16] = [M, K], B: [8, 16] = [N, K] (transposed), C: [16, 8] = [M, N]
//
// Run: mlir-opt-18 --convert-vector-to-gpu="use-nvgpu" test_vector_to_mma_v2.mlir

func.func @test_vector_contract_to_mma_v2(
    %A: vector<16x16xf16>,
    %B: vector<8x16xf16>,
    %C: vector<16x8xf16>) -> vector<16x8xf16> {
  // Transposed B layout: (k, n) -> (n, k) for the contract
  %result = vector.contract {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,  // A: [M, K]
      affine_map<(m, n, k) -> (n, k)>,  // B: [N, K] (transposed)
      affine_map<(m, n, k) -> (m, n)>   // C: [M, N]
    ],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
  } %A, %B, %C : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  return %result : vector<16x8xf16>
}
