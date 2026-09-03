// Vectorized Form: vector.contract on 16x4 tile over K=64 with 8-wide vectors
#contraction_trait = {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d2, d1)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

module {
  func.func @matmul_vectorized_16x4_f32(%A: memref<16x64xf32>, %B: memref<64x4xf32>, %C: memref<16x4xf32>) {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    
    // Load accumulators
    %acc_init = vector.transfer_read %C[%c0, %c0], %f0 {in_bounds = [true, true]} : memref<16x4xf32>, vector<16x4xf32>
    
    // Unrolled K reduction loop
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %acc_res = scf.for %k = %c0 to %c64 step %c1 iter_args(%acc = %acc_init) -> (vector<16x4xf32>) {
      %a = vector.transfer_read %A[%c0, %k], %f0 {in_bounds = [true, true]} : memref<16x64xf32>, vector<16x1xf32>
      %b = vector.transfer_read %B[%k, %c0], %f0 {in_bounds = [true, true]} : memref<64x4xf32>, vector<1x4xf32>
      %acc_next = vector.contract #contraction_trait %a, %b, %acc : vector<16x1xf32>, vector<1x4xf32> into vector<16x4xf32>
      scf.yield %acc_next : vector<16x4xf32>
    }
    
    // Store accumulators
    vector.transfer_write %acc_res, %C[%c0, %c0] {in_bounds = [true, true]} : vector<16x4xf32>, memref<16x4xf32>
    return
  }
}
