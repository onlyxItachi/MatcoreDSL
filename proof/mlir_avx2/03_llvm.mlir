// Lowered LLVM Dialect Form: 16x4 tile over K=64 using <8 x float> vectors and FMA
module attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-windows-msvc"} {
  llvm.func @matmul_llvm_16x4_f32(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %zero = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %k_start = llvm.mlir.constant(0 : i64) : i64
    %k_end = llvm.mlir.constant(64 : i64) : i64
    %step = llvm.mlir.constant(1 : i64) : i64
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    %c4 = llvm.mlir.constant(4 : i64) : i64
    %c5 = llvm.mlir.constant(5 : i64) : i64
    %c6 = llvm.mlir.constant(6 : i64) : i64
    %c7 = llvm.mlir.constant(7 : i64) : i64
    %c8 = llvm.mlir.constant(8 : i64) : i64

    llvm.br ^bb1(%k_start, %zero, %zero, %zero, %zero, %zero, %zero, %zero, %zero : i64, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)

  ^bb1(%k: i64, %acc0_0: vector<8xf32>, %acc1_0: vector<8xf32>, %acc0_1: vector<8xf32>, %acc1_1: vector<8xf32>, %acc0_2: vector<8xf32>, %acc1_2: vector<8xf32>, %acc0_3: vector<8xf32>, %acc1_3: vector<8xf32>):
    %cond = llvm.icmp "slt" %k, %k_end : i64
    llvm.cond_br %cond, ^bb2, ^bb3

  ^bb2:
    // Load A slice (16 floats = two 8-wide vectors)
    %a_idx0 = llvm.mul %k, %c8 : i64
    %a_ptr0 = llvm.getelementptr %arg0[%a_idx0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a0 = llvm.load %a_ptr0 : !llvm.ptr -> vector<8xf32>
    
    %a_idx1 = llvm.add %a_idx0, %c8 : i64
    %a_ptr1 = llvm.getelementptr %arg0[%a_idx1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a1 = llvm.load %a_ptr1 : !llvm.ptr -> vector<8xf32>

    // Load B row scalars and broadcast to vector<8xf32>
    %b_row_base = llvm.mul %k, %c4 : i64
    %b_ptr0 = llvm.getelementptr %arg1[%b_row_base] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %b0_s = llvm.load %b_ptr0 : !llvm.ptr -> f32
    %b0 = llvm.insertelement %b0_s, %zero[%k_start : i64] : vector<8xf32>
    %b0_vec = llvm.shufflevector %b0, %b0 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>

    %b_idx1 = llvm.add %b_row_base, %c1 : i64
    %b_ptr1 = llvm.getelementptr %arg1[%b_idx1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %b1_s = llvm.load %b_ptr1 : !llvm.ptr -> f32
    %b1 = llvm.insertelement %b1_s, %zero[%k_start : i64] : vector<8xf32>
    %b1_vec = llvm.shufflevector %b1, %b1 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>

    %b_idx2 = llvm.add %b_row_base, %c2 : i64
    %b_ptr2 = llvm.getelementptr %arg1[%b_idx2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %b2_s = llvm.load %b_ptr2 : !llvm.ptr -> f32
    %b2 = llvm.insertelement %b2_s, %zero[%k_start : i64] : vector<8xf32>
    %b2_vec = llvm.shufflevector %b2, %b2 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>

    %b_idx3 = llvm.add %b_row_base, %c3 : i64
    %b_ptr3 = llvm.getelementptr %arg1[%b_idx3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %b3_s = llvm.load %b_ptr3 : !llvm.ptr -> f32
    %b3 = llvm.insertelement %b3_s, %zero[%k_start : i64] : vector<8xf32>
    %b3_vec = llvm.shufflevector %b3, %b3 [0, 0, 0, 0, 0, 0, 0, 0] : vector<8xf32>

    // 8 FMA updates
    %n_acc0_0 = llvm.intr.fmuladd(%a0, %b0_vec, %acc0_0) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %n_acc1_0 = llvm.intr.fmuladd(%a1, %b0_vec, %acc1_0) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %n_acc0_1 = llvm.intr.fmuladd(%a0, %b1_vec, %acc0_1) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %n_acc1_1 = llvm.intr.fmuladd(%a1, %b1_vec, %acc1_1) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %n_acc0_2 = llvm.intr.fmuladd(%a0, %b2_vec, %acc0_2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %n_acc1_2 = llvm.intr.fmuladd(%a1, %b2_vec, %acc1_2) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %n_acc0_3 = llvm.intr.fmuladd(%a0, %b3_vec, %acc0_3) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>
    %n_acc1_3 = llvm.intr.fmuladd(%a1, %b3_vec, %acc1_3) : (vector<8xf32>, vector<8xf32>, vector<8xf32>) -> vector<8xf32>

    %k_next = llvm.add %k, %step : i64
    llvm.br ^bb1(%k_next, %n_acc0_0, %n_acc1_0, %n_acc0_1, %n_acc1_1, %n_acc0_2, %n_acc1_2, %n_acc0_3, %n_acc1_3 : i64, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)

  ^bb3:
    // Store accumulators into C (16x4 floats = 8 vectors of 8 floats)
    %c_ptr0 = llvm.getelementptr %arg2[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, vector<8xf32>
    llvm.store %acc0_0, %c_ptr0 : vector<8xf32>, !llvm.ptr

    %c_ptr1 = llvm.getelementptr %arg2[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, vector<8xf32>
    llvm.store %acc1_0, %c_ptr1 : vector<8xf32>, !llvm.ptr

    %c_ptr2 = llvm.getelementptr %arg2[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, vector<8xf32>
    llvm.store %acc0_1, %c_ptr2 : vector<8xf32>, !llvm.ptr

    %c_ptr3 = llvm.getelementptr %arg2[%c3] : (!llvm.ptr, i64) -> !llvm.ptr, vector<8xf32>
    llvm.store %acc1_1, %c_ptr3 : vector<8xf32>, !llvm.ptr

    %c_ptr4 = llvm.getelementptr %arg2[%c4] : (!llvm.ptr, i64) -> !llvm.ptr, vector<8xf32>
    llvm.store %acc0_2, %c_ptr4 : vector<8xf32>, !llvm.ptr

    %c_ptr5 = llvm.getelementptr %arg2[%c5] : (!llvm.ptr, i64) -> !llvm.ptr, vector<8xf32>
    llvm.store %acc1_2, %c_ptr5 : vector<8xf32>, !llvm.ptr

    %c_ptr6 = llvm.getelementptr %arg2[%c6] : (!llvm.ptr, i64) -> !llvm.ptr, vector<8xf32>
    llvm.store %acc0_3, %c_ptr6 : vector<8xf32>, !llvm.ptr

    %c_ptr7 = llvm.getelementptr %arg2[%c7] : (!llvm.ptr, i64) -> !llvm.ptr, vector<8xf32>
    llvm.store %acc1_3, %c_ptr7 : vector<8xf32>, !llvm.ptr

    llvm.return
  }
}
