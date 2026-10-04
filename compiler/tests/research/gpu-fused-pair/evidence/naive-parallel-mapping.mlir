module {
  func.func @__matcore_strict_fused_gemm_f32_v1(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg3, %c0 : memref<?x?xf32>
    %dim_2 = memref.dim %arg3, %c1 : memref<?x?xf32>
    gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %dim_1, %arg12 = %dim_2, %arg13 = %c1) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1, %arg15 = %c1, %arg16 = %c1) {
      memref.store %cst, %arg3[%arg5, %arg6] : memref<?x?xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    %dim_3 = memref.dim %arg2, %c1 : memref<?x?xf32>
    scf.for %arg5 = %c0 to %dim step %c4 {
      %0 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%arg5)[%dim]
      %dim_4 = memref.dim %arg0, %c1 : memref<?x?xf32>
      %subview = memref.subview %arg0[%arg5, 0] [%0, %dim_4] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      %subview_5 = memref.subview %arg1[0, 0] [%dim_4, %dim_0] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %dim_0], strides: [%dim_0, 1] : memref<?x?xf32> to memref<?x?xf32>
      gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %0, %arg13 = %dim_0, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c1, %arg16 = %c1, %arg17 = %c1) {
        memref.store %cst, %reinterpret_cast[%arg6, %arg7] : memref<?x?xf32>
        gpu.terminator
      } {SCFToGPU_visited}
      gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %0, %arg13 = %dim_0, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c1, %arg16 = %c1, %arg17 = %c1) {
        scf.for %arg18 = %c0 to %dim_4 step %c1 {
          %1 = memref.load %subview[%arg6, %arg18] : memref<?x?xf32, strided<[?, 1], offset: ?>>
          %2 = memref.load %subview_5[%arg18, %arg7] : memref<?x?xf32, strided<[?, 1]>>
          %3 = memref.load %reinterpret_cast[%arg6, %arg7] : memref<?x?xf32>
          %4 = arith.mulf %1, %2 : f32
          %5 = arith.addf %3, %4 : f32
          memref.store %5, %reinterpret_cast[%arg6, %arg7] : memref<?x?xf32>
        }
        gpu.terminator
      } {SCFToGPU_visited}
      %subview_6 = memref.subview %arg2[0, 0] [%dim_0, %dim_3] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
      %subview_7 = memref.subview %arg3[%arg5, 0] [%0, %dim_3] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
      gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg12 = %0, %arg13 = %dim_3, %arg14 = %c1) threads(%arg9, %arg10, %arg11) in (%arg15 = %c1, %arg16 = %c1, %arg17 = %c1) {
        scf.for %arg18 = %c0 to %dim_0 step %c1 {
          %1 = memref.load %reinterpret_cast[%arg6, %arg18] : memref<?x?xf32>
          %2 = memref.load %subview_6[%arg18, %arg7] : memref<?x?xf32, strided<[?, 1]>>
          %3 = memref.load %subview_7[%arg6, %arg7] : memref<?x?xf32, strided<[?, 1], offset: ?>>
          %4 = arith.mulf %1, %2 : f32
          %5 = arith.addf %3, %4 : f32
          memref.store %5, %subview_7[%arg6, %arg7] : memref<?x?xf32, strided<[?, 1], offset: ?>>
        }
        gpu.terminator
      } {SCFToGPU_visited}
    }
    return
  }
}
