// RESEARCH ONLY: this tensor specimen has no authenticated source authority.
// C=A*B; E=C*D. C has exactly one use, as E's lhs; no epilogue/effects.
module attributes {transform.with_named_sequence, llvm.target_triple = "x86_64-pc-linux-gnu"} {
  func.func @research_strict_fused_pair(%a: tensor<?x?xf32>,
      %b: tensor<?x?xf32>, %d: tensor<?x?xf32>,
      %output: tensor<?x?xf32>) -> tensor<?x?xf32>
      attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %m = tensor.dim %a, %c0 : tensor<?x?xf32>
    %n = tensor.dim %b, %c1 : tensor<?x?xf32>
    %scratch = tensor.empty(%m, %n) : tensor<?x?xf32>
    %init = linalg.fill {research.producer_fill} ins(%zero : f32) outs(%scratch : tensor<?x?xf32>)
      -> tensor<?x?xf32>
    %c = linalg.matmul {research.producer} ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %einit = linalg.fill ins(%zero : f32) outs(%output : tensor<?x?xf32>)
      -> tensor<?x?xf32>
    %e = linalg.matmul {research.consumer} ins(%c, %d : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%einit : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %e : tensor<?x?xf32>
  }
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %consumer = transform.structured.match attributes{research.consumer} in %root
      : (!transform.any_op) -> !transform.any_op
    %producer = transform.structured.match attributes{research.producer} in %root
      : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match attributes{research.producer_fill} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %consumer tile_sizes [4, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused, %updated = transform.structured.fuse_into_containing_op %producer into %loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill, %updated_fill = transform.structured.fuse_into_containing_op %fill into %updated
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
  // Negative control: legal row fusion is not itself a smaller allocation.
  transform.named_sequence @__no_empty_fold(%root: !transform.any_op {transform.readonly}) {
    %consumer = transform.structured.match attributes{research.consumer} in %root
      : (!transform.any_op) -> !transform.any_op
    %producer = transform.structured.match attributes{research.producer} in %root
      : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match attributes{research.producer_fill} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %consumer tile_sizes [4, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused, %updated = transform.structured.fuse_into_containing_op %producer into %loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill, %updated_fill = transform.structured.fuse_into_containing_op %fill into %updated
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
  // Negative control: column tiles recompute every producer panel per P tile.
  transform.named_sequence @__column_recompute(%root: !transform.any_op {transform.readonly}) {
    %consumer = transform.structured.match attributes{research.consumer} in %root
      : (!transform.any_op) -> !transform.any_op
    %producer = transform.structured.match attributes{research.producer} in %root
      : (!transform.any_op) -> !transform.any_op
    %fill = transform.structured.match attributes{research.producer_fill} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %rows, %columns = transform.structured.tile_using_for %consumer tile_sizes [4, 2, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %fused, %updated = transform.structured.fuse_into_containing_op %producer into %columns
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_fill, %updated_fill = transform.structured.fuse_into_containing_op %fill into %updated
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.fold_tensor_empty
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
