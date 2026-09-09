// Research only: this input is not authenticated product execution authority.
// Row-major f32 A/B may alias, private C must be disjoint from both inputs.
// Compatible nonnegative extents are validated by the research caller.
// Per-GEMM reassociation/FMA is explicit; no finite-only or signed-zero waiver.
module attributes {transform.with_named_sequence} {
  func.func @research_gemm(%a: memref<?x?xf32>, %b: memref<?x?xf32>,
                           %c: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<?x?xf32>)
    linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
                  outs(%c : memref<?x?xf32>)
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root : (!transform.any_op) -> !transform.any_op
    %tiled, %i, %j, %k = transform.structured.tile_using_for %mm tile_sizes [4, 8, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled vector_sizes [4, 8, 4] : !transform.any_op
    transform.yield
  }
  transform.named_sequence @lower_contract(%root: !transform.any_op {transform.readonly}) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %functions {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.lower_contraction
    } : !transform.any_op
    transform.yield
  }
  transform.named_sequence @hoist(%root: !transform.any_op {transform.readonly}) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %done = transform.structured.hoist_redundant_vector_transfers %functions : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
