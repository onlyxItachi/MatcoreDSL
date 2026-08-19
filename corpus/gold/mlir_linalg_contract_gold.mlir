// MDSLC Gold Fixture: Canonical Linalg Matmul to Vector Contraction Bridge
module {
  func.func @gold_matmul_contract(
      %A: vector<6x4xf32>,
      %B: vector<4x16xf32>,
      %C: vector<6x16xf32>) -> vector<6x16xf32> {
    %res = vector.contract
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      iterator_types = ["parallel", "parallel", "reduction"]
      kind = #vector.kind<add>
      %A, %B, %C : vector<6x4xf32>, vector<4x16xf32> into vector<6x16xf32>
    return %res : vector<6x16xf32>
  }
}
