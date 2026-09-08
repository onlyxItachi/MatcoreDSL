// Research-only: C is private and disjoint from both immutable inputs.
// A and B may alias. A descriptor pointer is not its pointee data pointer;
// inspect upstream conversion before treating this attribute as useful evidence.
module {
  func.func @research_gemm(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                           %c: memref<?x?xf32> {llvm.noalias})
      attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<?x?xf32>)
    linalg.generic {
      indexing_maps = [affine_map<(m,k,n)->(m,k)>,
                       affine_map<(m,k,n)->(k,n)>,
                       affine_map<(m,k,n)->(m,n)>],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
      outs(%c : memref<?x?xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %acc: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    }
    return
  }
}
