// Research input only. Same private-output contract as tiled_vector.mlir.
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
    %generic = transform.structured.generalize %mm : (!transform.any_op) -> !transform.any_op
    %permuted = transform.structured.interchange %generic iterator_interchange = [0, 2, 1] : (!transform.any_op) -> !transform.any_op
    %tiled, %i, %j, %k = transform.structured.tile_using_for %permuted tile_sizes [4, 1, 8] interchange = [0, 2, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled vector_sizes [4, 1, 8] : !transform.any_op
    transform.yield
  }
}
