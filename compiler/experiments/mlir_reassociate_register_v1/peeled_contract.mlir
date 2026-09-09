// Research only; per-GEMM reassociate_f32, fresh disjoint output, inputs MAY alias.
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
    %tiled, %i, %j, %k = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    %j_full, %j_tail = transform.loop.peel %j : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %i_full, %i_tail = transform.loop.peel %i : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  transform.named_sequence @vectorize(%root: !transform.any_op {transform.readonly}) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %new = transform.structured.vectorize_children_and_apply_patterns %functions : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %new {
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
