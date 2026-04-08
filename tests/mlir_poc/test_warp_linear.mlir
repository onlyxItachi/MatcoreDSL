// MW-0 Test A2: Linear warp mapping (1D, 4 warps along X only)
// Uses block_dims = [128, 1, 1] with linear warp IDs
//
// Run: mlir-opt-18 --transform-interpreter test_warp_linear.mlir

func.func @test_warp_linear(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  gpu.launch blocks(%bx, %by, %bz) in (%nbx = %c1, %nby = %c1, %nbz = %c1)
            threads(%tx, %ty, %tz) in (%ntx = %c128, %nty = %c1, %ntz = %c1) {
    // 4 warps linearly along X
    scf.forall (%w) in (4) {
      %val = memref.load %arg0[%w, %w] : memref<64x64xf16>
      memref.store %val, %arg1[%w, %w] : memref<64x64xf16>
    } {mapping = [#gpu.warp<x>]}
    gpu.terminator
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %launch = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %mapped = transform.gpu.map_nested_forall_to_threads %launch block_dims = [128, 1, 1] sync_after_distribute = true warp_size = 32 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
