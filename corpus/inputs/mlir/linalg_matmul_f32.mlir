// Diagnostic Case: linalg_matmul_f32.mlir
// Purpose: Canonical structured linalg.matmul on static rank-2 f32 memrefs.
module {
  func.func @matmul_f32(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) {
    linalg.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
                  outs(%C : memref<128x128xf32>)
    return
  }
}
