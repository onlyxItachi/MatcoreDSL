; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @matmul_llvm_16x4_f32(ptr %0, ptr %1, ptr %2) {
  br label %4

4:                                                ; preds = %15, %3
  %5 = phi i64 [ %50, %15 ], [ 0, %3 ]
  %6 = phi <8 x float> [ %42, %15 ], [ zeroinitializer, %3 ]
  %7 = phi <8 x float> [ %43, %15 ], [ zeroinitializer, %3 ]
  %8 = phi <8 x float> [ %44, %15 ], [ zeroinitializer, %3 ]
  %9 = phi <8 x float> [ %45, %15 ], [ zeroinitializer, %3 ]
  %10 = phi <8 x float> [ %46, %15 ], [ zeroinitializer, %3 ]
  %11 = phi <8 x float> [ %47, %15 ], [ zeroinitializer, %3 ]
  %12 = phi <8 x float> [ %48, %15 ], [ zeroinitializer, %3 ]
  %13 = phi <8 x float> [ %49, %15 ], [ zeroinitializer, %3 ]
  %14 = icmp slt i64 %5, 64
  br i1 %14, label %15, label %51

15:                                               ; preds = %4
  %16 = mul i64 %5, 8
  %17 = getelementptr float, ptr %0, i64 %16
  %18 = load <8 x float>, ptr %17, align 32
  %19 = add i64 %16, 8
  %20 = getelementptr float, ptr %0, i64 %19
  %21 = load <8 x float>, ptr %20, align 32
  %22 = mul i64 %5, 4
  %23 = getelementptr float, ptr %1, i64 %22
  %24 = load float, ptr %23, align 4
  %25 = insertelement <8 x float> zeroinitializer, float %24, i64 0
  %26 = shufflevector <8 x float> %25, <8 x float> %25, <8 x i32> zeroinitializer
  %27 = add i64 %22, 1
  %28 = getelementptr float, ptr %1, i64 %27
  %29 = load float, ptr %28, align 4
  %30 = insertelement <8 x float> zeroinitializer, float %29, i64 0
  %31 = shufflevector <8 x float> %30, <8 x float> %30, <8 x i32> zeroinitializer
  %32 = add i64 %22, 2
  %33 = getelementptr float, ptr %1, i64 %32
  %34 = load float, ptr %33, align 4
  %35 = insertelement <8 x float> zeroinitializer, float %34, i64 0
  %36 = shufflevector <8 x float> %35, <8 x float> %35, <8 x i32> zeroinitializer
  %37 = add i64 %22, 3
  %38 = getelementptr float, ptr %1, i64 %37
  %39 = load float, ptr %38, align 4
  %40 = insertelement <8 x float> zeroinitializer, float %39, i64 0
  %41 = shufflevector <8 x float> %40, <8 x float> %40, <8 x i32> zeroinitializer
  %42 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %18, <8 x float> %26, <8 x float> %6)
  %43 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %21, <8 x float> %26, <8 x float> %7)
  %44 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %18, <8 x float> %31, <8 x float> %8)
  %45 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %21, <8 x float> %31, <8 x float> %9)
  %46 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %18, <8 x float> %36, <8 x float> %10)
  %47 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %21, <8 x float> %36, <8 x float> %11)
  %48 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %18, <8 x float> %41, <8 x float> %12)
  %49 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %21, <8 x float> %41, <8 x float> %13)
  %50 = add i64 %5, 1
  br label %4

51:                                               ; preds = %4
  %52 = getelementptr <8 x float>, ptr %2, i64 0
  store <8 x float> %6, ptr %52, align 32
  %53 = getelementptr <8 x float>, ptr %2, i64 1
  store <8 x float> %7, ptr %53, align 32
  %54 = getelementptr <8 x float>, ptr %2, i64 2
  store <8 x float> %8, ptr %54, align 32
  %55 = getelementptr <8 x float>, ptr %2, i64 3
  store <8 x float> %9, ptr %55, align 32
  %56 = getelementptr <8 x float>, ptr %2, i64 4
  store <8 x float> %10, ptr %56, align 32
  %57 = getelementptr <8 x float>, ptr %2, i64 5
  store <8 x float> %11, ptr %57, align 32
  %58 = getelementptr <8 x float>, ptr %2, i64 6
  store <8 x float> %12, ptr %58, align 32
  %59 = getelementptr <8 x float>, ptr %2, i64 7
  store <8 x float> %13, ptr %59, align 32
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"CodeView", i32 1}
