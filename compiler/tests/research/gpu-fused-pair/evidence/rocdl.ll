; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @__matcore_research_strict_fused_pair_kernel(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, ptr %21, ptr %22, i64 %23, i64 %24, i64 %25, i64 %26, i64 %27, ptr %28, ptr %29, i64 %30, i64 %31, i64 %32, i64 %33, i64 %34) #0 !reqd_work_group_size !1 {
  br label %36

36:                                               ; preds = %48, %35
  %37 = phi i64 [ %49, %48 ], [ 0, %35 ]
  %38 = icmp slt i64 %37, %24
  br i1 %38, label %39, label %50

39:                                               ; preds = %36
  br label %40

40:                                               ; preds = %43, %39
  %41 = phi i64 [ %47, %43 ], [ 0, %39 ]
  %42 = icmp slt i64 %41, %25
  br i1 %42, label %43, label %48

43:                                               ; preds = %40
  %44 = mul nuw nsw i64 %37, %26
  %45 = add nuw nsw i64 %44, %41
  %46 = getelementptr inbounds nuw float, ptr %22, i64 %45
  store float 0.000000e+00, ptr %46, align 4
  %47 = add i64 %41, 1
  br label %40

48:                                               ; preds = %40
  %49 = add i64 %37, 1
  br label %36

50:                                               ; preds = %36
  br label %51

51:                                               ; preds = %147, %50
  %52 = phi i64 [ %148, %147 ], [ 0, %50 ]
  %53 = icmp slt i64 %52, %3
  br i1 %53, label %54, label %149

54:                                               ; preds = %51
  %55 = mul nsw i64 %52, -1
  %56 = add i64 %55, %3
  %57 = call i64 @llvm.smin.i64(i64 %56, i64 4)
  %58 = mul nsw i64 %52, %5
  br label %59

59:                                               ; preds = %71, %54
  %60 = phi i64 [ %72, %71 ], [ 0, %54 ]
  %61 = icmp slt i64 %60, %57
  br i1 %61, label %62, label %73

62:                                               ; preds = %59
  br label %63

63:                                               ; preds = %66, %62
  %64 = phi i64 [ %70, %66 ], [ 0, %62 ]
  %65 = icmp slt i64 %64, %11
  br i1 %65, label %66, label %71

66:                                               ; preds = %63
  %67 = mul nuw nsw i64 %60, %11
  %68 = add nuw nsw i64 %67, %64
  %69 = getelementptr inbounds nuw float, ptr %29, i64 %68
  store float 0.000000e+00, ptr %69, align 4
  %70 = add i64 %64, 1
  br label %63

71:                                               ; preds = %63
  %72 = add i64 %60, 1
  br label %59

73:                                               ; preds = %59
  br label %74

74:                                               ; preds = %107, %73
  %75 = phi i64 [ %108, %107 ], [ 0, %73 ]
  %76 = icmp slt i64 %75, %57
  br i1 %76, label %77, label %109

77:                                               ; preds = %74
  br label %78

78:                                               ; preds = %105, %77
  %79 = phi i64 [ %106, %105 ], [ 0, %77 ]
  %80 = icmp slt i64 %79, %11
  br i1 %80, label %81, label %107

81:                                               ; preds = %78
  br label %82

82:                                               ; preds = %85, %81
  %83 = phi i64 [ %104, %85 ], [ 0, %81 ]
  %84 = icmp slt i64 %83, %4
  br i1 %84, label %85, label %105

85:                                               ; preds = %82
  %86 = getelementptr float, ptr %1, i64 %58
  %87 = mul nuw nsw i64 %75, %5
  %88 = add nuw nsw i64 %87, %83
  %89 = getelementptr inbounds nuw float, ptr %86, i64 %88
  %90 = load float, ptr %89, align 4
  %91 = mul nuw nsw i64 %83, %12
  %92 = add nuw nsw i64 %91, %79
  %93 = getelementptr inbounds nuw float, ptr %8, i64 %92
  %94 = load float, ptr %93, align 4
  %95 = mul nuw nsw i64 %75, %11
  %96 = add nuw nsw i64 %95, %79
  %97 = getelementptr inbounds nuw float, ptr %29, i64 %96
  %98 = load float, ptr %97, align 4
  %99 = fmul float %90, %94
  %100 = fadd float %98, %99
  %101 = mul nuw nsw i64 %75, %11
  %102 = add nuw nsw i64 %101, %79
  %103 = getelementptr inbounds nuw float, ptr %29, i64 %102
  store float %100, ptr %103, align 4
  %104 = add i64 %83, 1
  br label %82

105:                                              ; preds = %82
  %106 = add i64 %79, 1
  br label %78

107:                                              ; preds = %78
  %108 = add i64 %75, 1
  br label %74

109:                                              ; preds = %74
  %110 = mul nsw i64 %52, %26
  br label %111

111:                                              ; preds = %145, %109
  %112 = phi i64 [ %146, %145 ], [ 0, %109 ]
  %113 = icmp slt i64 %112, %57
  br i1 %113, label %114, label %147

114:                                              ; preds = %111
  br label %115

115:                                              ; preds = %143, %114
  %116 = phi i64 [ %144, %143 ], [ 0, %114 ]
  %117 = icmp slt i64 %116, %18
  br i1 %117, label %118, label %145

118:                                              ; preds = %115
  br label %119

119:                                              ; preds = %122, %118
  %120 = phi i64 [ %142, %122 ], [ 0, %118 ]
  %121 = icmp slt i64 %120, %11
  br i1 %121, label %122, label %143

122:                                              ; preds = %119
  %123 = mul nuw nsw i64 %112, %11
  %124 = add nuw nsw i64 %123, %120
  %125 = getelementptr inbounds nuw float, ptr %29, i64 %124
  %126 = load float, ptr %125, align 4
  %127 = mul nuw nsw i64 %120, %19
  %128 = add nuw nsw i64 %127, %116
  %129 = getelementptr inbounds nuw float, ptr %15, i64 %128
  %130 = load float, ptr %129, align 4
  %131 = getelementptr float, ptr %22, i64 %110
  %132 = mul nuw nsw i64 %112, %26
  %133 = add nuw nsw i64 %132, %116
  %134 = getelementptr inbounds nuw float, ptr %131, i64 %133
  %135 = load float, ptr %134, align 4
  %136 = fmul float %126, %130
  %137 = fadd float %135, %136
  %138 = getelementptr float, ptr %22, i64 %110
  %139 = mul nuw nsw i64 %112, %26
  %140 = add nuw nsw i64 %139, %116
  %141 = getelementptr inbounds nuw float, ptr %138, i64 %140
  store float %137, ptr %141, align 4
  %142 = add i64 %120, 1
  br label %119

143:                                              ; preds = %119
  %144 = add i64 %116, 1
  br label %115

145:                                              ; preds = %115
  %146 = add i64 %112, 1
  br label %111

147:                                              ; preds = %111
  %148 = add i64 %52, 4
  br label %51

149:                                              ; preds = %51
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.smin.i64(i64, i64) #1

attributes #0 = { "amdgpu-flat-work-group-size"="1,1" "denormal-fp-math"="ieee,ieee" "denormal-fp-math-f32"="ieee,ieee" "target-cpu"="gfx1150" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, i32 1, i32 1}
