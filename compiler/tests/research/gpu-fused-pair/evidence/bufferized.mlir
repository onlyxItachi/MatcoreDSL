module {
  func.func @__matcore_strict_fused_gemm_f32_v1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    linalg.fill ins(%cst : f32) outs(%arg3 : memref<?x?xf32>)
    %dim_1 = memref.dim %arg2, %c1 : memref<?x?xf32>
    scf.for %arg5 = %c0 to %dim step %c4 {
      %0 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%arg5)[%dim]
      %dim_2 = memref.dim %arg0, %c1 : memref<?x?xf32>
      %subview = memref.subview %arg0[%arg5, 0] [%0, %dim_2] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_3 = memref.subview %arg1[0, 0] [%dim_2, %dim_0] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %dim_0], strides: [%dim_0, 1] : memref<?x?xf32> to memref<?x?xf32>
      linalg.fill {matcore.pair.fill} ins(%cst : f32) outs(%reinterpret_cast : memref<?x?xf32>)
      linalg.matmul {matcore.pair.producer} ins(%subview, %subview_3 : memref<?x?xf32, strided<[?, 1], offset: ?>>, memref<?x?xf32, strided<[?, 1]>>) outs(%reinterpret_cast : memref<?x?xf32>)
      %subview_4 = memref.subview %arg2[0, 0] [%dim_0, %dim_1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
      %subview_5 = memref.subview %arg3[%arg5, 0] [%0, %dim_1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      linalg.matmul {matcore.pair.consumer} ins(%reinterpret_cast, %subview_4 : memref<?x?xf32>, memref<?x?xf32, strided<[?, 1]>>) outs(%subview_5 : memref<?x?xf32, strided<[?, 1], offset: ?>>)
    }
    return
  }
}
