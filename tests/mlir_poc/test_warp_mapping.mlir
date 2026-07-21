// MW-0 Test A: Prove that gpu.warp mapped scf.forall lowers 
// via MapNestedForallToThreads in MLIR 18.
//
// Run: mlir-opt-18 --transform-interpreter --split-input-file test_warp_mapping.mlir

// The module contains a gpu.launch with a nested scf.forall using gpu.warp mapping.
// The transform sequence maps the warp forall to thread indices.

func.func @test_warp_forall(%arg0: memref<64x64xf16>, %arg1: memref<64x64xf16>) {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  gpu.launch blocks(%bx, %by, %bz) in (%nbx = %c1, %nby = %c1, %nbz = %c1)
            threads(%tx, %ty, %tz) in (%ntx = %c128, %nty = %c1, %ntz = %c1) {
    // 4 warps (2x2) each handle a 32x32 subtile
    scf.forall (%wy, %wx) in (2, 2) {
      %offset_m = arith.muli %wy, %c1 : index  // simplified
      %offset_n = arith.muli %wx, %c1 : index
      // Each warp reads its subtile
      %val = memref.load %arg0[%offset_m, %offset_n] : memref<64x64xf16>
      memref.store %val, %arg1[%offset_m, %offset_n] : memref<64x64xf16>
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    gpu.terminator
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %launch = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %mapped = transform.gpu.map_nested_forall_to_threads %launch block_dims = [64, 2, 1] sync_after_distribute = true warp_size = 32 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
