// Exact upstream-generated graph reviewed by the isolated physical proof.
// Diagnostic self-consistency only, not a general IR importer or authority.
constexpr llvm::StringLiteral kOutlinedReference = R"mlir(
module attributes {gpu.container_module} {
  gpu.module @strict_pair {
    gpu.func @__matcore_strict_fused_gemm_f32_v1_kernel(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %arg3: memref<?x?xf32>, %arg4: memref<?x?xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>, known_grid_size = array<i32: 1, 1, 1>} {
      %cst = arith.constant 0.000000e+00 : f32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
      %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
      %dim_1 = memref.dim %arg3, %c0 : memref<?x?xf32>
      %dim_2 = memref.dim %arg3, %c1 : memref<?x?xf32>
      scf.for %arg5 = %c0 to %dim_1 step %c1 {
        scf.for %arg6 = %c0 to %dim_2 step %c1 {
          memref.store %cst, %arg3[%arg5, %arg6] : memref<?x?xf32>
        }
      }
      %dim_3 = memref.dim %arg2, %c1 : memref<?x?xf32>
      scf.for %arg5 = %c0 to %dim step %c4 {
        %0 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 4)>(%arg5)[%dim]
        %dim_4 = memref.dim %arg0, %c1 : memref<?x?xf32>
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
        %1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%arg5, %strides#0]
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [%0, %dim_4], strides: [%strides#0, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %base_buffer_5, %offset_6, %sizes_7:2, %strides_8:2 = memref.extract_strided_metadata %arg1 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
        %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%dim_4, %dim_0], strides: [%strides_8#0, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
        %reinterpret_cast_10 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%0, %dim_0], strides: [%dim_0, 1] : memref<?x?xf32> to memref<?x?xf32>
        scf.for %arg6 = %c0 to %0 step %c1 {
          scf.for %arg7 = %c0 to %dim_0 step %c1 {
            memref.store %cst, %reinterpret_cast_10[%arg6, %arg7] : memref<?x?xf32>
          }
        }
        scf.for %arg6 = %c0 to %0 step %c1 {
          scf.for %arg7 = %c0 to %dim_0 step %c1 {
            scf.for %arg8 = %c0 to %dim_4 step %c1 {
              %3 = memref.load %reinterpret_cast[%arg6, %arg8] : memref<?x?xf32, strided<[?, 1], offset: ?>>
              %4 = memref.load %reinterpret_cast_9[%arg8, %arg7] : memref<?x?xf32, strided<[?, 1]>>
              %5 = memref.load %reinterpret_cast_10[%arg6, %arg7] : memref<?x?xf32>
              %6 = arith.mulf %3, %4 : f32
              %7 = arith.addf %5, %6 : f32
              memref.store %7, %reinterpret_cast_10[%arg6, %arg7] : memref<?x?xf32>
            }
          }
        }
        %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %arg2 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
        %reinterpret_cast_15 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [%dim_0, %dim_3], strides: [%strides_14#0, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
        %base_buffer_16, %offset_17, %sizes_18:2, %strides_19:2 = memref.extract_strided_metadata %arg3 : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
        %2 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%arg5, %strides_19#0]
        %reinterpret_cast_20 = memref.reinterpret_cast %arg3 to offset: [%2], sizes: [%0, %dim_3], strides: [%strides_19#0, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        scf.for %arg6 = %c0 to %0 step %c1 {
          scf.for %arg7 = %c0 to %dim_3 step %c1 {
            scf.for %arg8 = %c0 to %dim_0 step %c1 {
              %3 = memref.load %reinterpret_cast_10[%arg6, %arg8] : memref<?x?xf32>
              %4 = memref.load %reinterpret_cast_15[%arg8, %arg7] : memref<?x?xf32, strided<[?, 1]>>
              %5 = memref.load %reinterpret_cast_20[%arg6, %arg7] : memref<?x?xf32, strided<[?, 1], offset: ?>>
              %6 = arith.mulf %3, %4 : f32
              %7 = arith.addf %5, %6 : f32
              memref.store %7, %reinterpret_cast_20[%arg6, %arg7] : memref<?x?xf32, strided<[?, 1], offset: ?>>
            }
          }
        }
      }
      gpu.return
    }
  }
}
)mlir";
