// Diagnostic Case: vector_contract_f32.mlir
// Purpose: Vector contraction operation lowering to SIMD / matrix instructions.
#contraction_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#iterator_types = ["parallel", "parallel", "reduction"]

module {
  func.func @vector_contract_16x16x16(
      %lhs: vector<16x16xf32>,
      %rhs: vector<16x16xf32>,
      %acc: vector<16x16xf32>) -> vector<16x16xf32> {
    %res = vector.contract
      indexing_maps = #contraction_accesses
      iterator_types = #iterator_types
      kind = #vector.kind<add>
      %lhs, %rhs, %acc : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
    return %res : vector<16x16xf32>
  }
}
