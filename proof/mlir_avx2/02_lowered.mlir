module {
  func.func @matmul_16x4_f32(%arg0: memref<16x64xf32>, %arg1: memref<64x4xf32>, %arg2: memref<16x4xf32>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
    %1 = arith.cmpi slt, %0, %c16 : index
    cf.cond_br %1, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb7
    %3 = arith.cmpi slt, %2, %c4 : index
    cf.cond_br %3, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%c0 : index)
  ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
    %5 = arith.cmpi slt, %4, %c64 : index
    cf.cond_br %5, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %6 = memref.load %arg0[%0, %4] : memref<16x64xf32>
    %7 = memref.load %arg1[%4, %2] : memref<64x4xf32>
    %8 = memref.load %arg2[%0, %2] : memref<16x4xf32>
    %9 = arith.mulf %6, %7 : f32
    %10 = arith.addf %8, %9 : f32
    memref.store %10, %arg2[%0, %2] : memref<16x4xf32>
    %11 = arith.addi %4, %c1 : index
    cf.br ^bb5(%11 : index)
  ^bb7:  // pred: ^bb5
    %12 = arith.addi %2, %c1 : index
    cf.br ^bb3(%12 : index)
  ^bb8:  // pred: ^bb3
    %13 = arith.addi %0, %c1 : index
    cf.br ^bb1(%13 : index)
  ^bb9:  // pred: ^bb1
    return
  }
}

