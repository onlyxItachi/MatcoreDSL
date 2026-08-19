// Canonical Input: Structured Linalg Contraction (16x4 float32 tile)
module {
  func.func @matmul_16x4_f32(%A: memref<16x64xf32>, %B: memref<64x4xf32>, %C: memref<16x4xf32>) {
    linalg.matmul
      ins(%A, %B : memref<16x64xf32>, memref<64x4xf32>)
      outs(%C : memref<16x4xf32>)
    return
  }
}
