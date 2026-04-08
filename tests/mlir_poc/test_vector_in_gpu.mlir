// MW-0 Test B3: vector.contract inside gpu.func context.
// ConvertVectorToGPU likely only processes ops inside GPU modules.
//
// Run: mlir-opt-18 --convert-vector-to-gpu="use-nvgpu" test_vector_in_gpu.mlir

gpu.module @gpu_kernels {
  gpu.func @test_mma_contract(
      %A: memref<16x16xf16, 3>,
      %B: memref<8x16xf16, 3>,
      %C: memref<16x8xf16>) kernel {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f16
    
    // Load A tile from shared memory
    %a = vector.transfer_read %A[%c0, %c0], %cst {in_bounds = [true, true]}
        : memref<16x16xf16, 3>, vector<16x16xf16>
    
    // Load B tile from shared memory (transposed layout for mma)
    %b = vector.transfer_read %B[%c0, %c0], %cst {in_bounds = [true, true]}
        : memref<8x16xf16, 3>, vector<8x16xf16>
    
    // Load accumulator
    %c_init = vector.transfer_read %C[%c0, %c0], %cst {in_bounds = [true, true]}
        : memref<16x8xf16>, vector<16x8xf16>
    
    // Contract: C += A * B^T → maps to mma.sync m16n8k16
    %result = vector.contract {
      indexing_maps = [
        affine_map<(m, n, k) -> (m, k)>,
        affine_map<(m, n, k) -> (n, k)>,
        affine_map<(m, n, k) -> (m, n)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
    } %a, %b, %c_init : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    
    // Store result
    vector.transfer_write %result, %C[%c0, %c0] {in_bounds = [true, true]}
        : vector<16x8xf16>, memref<16x8xf16>
    
    gpu.return
  }
}
