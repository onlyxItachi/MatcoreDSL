// Research-only static instantiation of the canonical bufferized strict GEMM.
// This is not an authenticated issuer, admitted source, or runtime candidate.
// M=2, K=3, N=4. Upstream lowering, not handwritten shader arithmetic.
module attributes {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    #spirv.resource_limits<>>
} {
  func.func @spirv_static_specimen(%a: memref<2x3xf32>, %b: memref<3x4xf32>, %c: memref<2x4xf32>) {
    %zero = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%zero : f32) outs(%c : memref<2x4xf32>)
    linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<3x4xf32>) outs(%c : memref<2x4xf32>)
    return
  }
}
