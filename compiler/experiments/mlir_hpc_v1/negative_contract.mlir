// Rejected research alternative: ordinary matmul vectorization may become
// vector.contract, whose lowering must not be assumed to preserve strict f32.
module attributes {transform.with_named_sequence} {
  func.func @research_gemm(%a: memref<1x2xf32>, %b: memref<2x1xf32>,
                           %c: memref<1x1xf32>) attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0.0 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<1x1xf32>)
    linalg.matmul ins(%a, %b : memref<1x2xf32>, memref<2x1xf32>)
                  outs(%c : memref<1x1xf32>)
    return
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %new = transform.structured.vectorize_children_and_apply_patterns %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
