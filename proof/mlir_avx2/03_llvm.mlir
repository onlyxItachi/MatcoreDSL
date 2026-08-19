// Lowered LLVM Dialect Form: 16x4 tile over K=64 using <8 x float> vectors
module attributes {llvm.data_layout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-pc-windows-msvc"} {
  llvm.func @matmul_llvm_16x4_f32(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(64 : i64) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    
    // Accumulators: 8 vector registers for 16x4 tile
    llvm.br ^bb1(%1, %0, %0, %0, %0, %0, %0, %0, %0 : i64, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)
    
  ^bb1(%k: i64, %acc0_0: vector<8xf32>, %acc1_0: vector<8xf32>, %acc0_1: vector<8xf32>, %acc1_1: vector<8xf32>, %acc0_2: vector<8xf32>, %acc1_2: vector<8xf32>, %acc0_3: vector<8xf32>, %acc1_3: vector<8xf32>):
    %cond = llvm.icmp "slt" %k, %2 : i64
    llvm.cond_br %cond, ^bb2, ^bb3
    
  ^bb2:
    // Load column slice of A (16 elements = 2 x 8-wide vectors)
    %a_offset0 = llvm.mul %k, %3 : i64
    %a_ptr0 = llvm.getelementptr %arg0[%a_offset0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %a0 = llvm.load %a_ptr0 {alignment = 32 : i64} : !llvm.ptr -> vector<8xf32>
    
    // Fused multiply-accumulate steps...
    %k_next = llvm.add %k, %3 : i64
    llvm.br ^bb1(%k_next, %acc0_0, %acc1_0, %acc0_1, %acc1_1, %acc0_2, %acc1_2, %acc0_3, %acc1_3 : i64, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)
    
  ^bb3:
    llvm.return
  }
}
