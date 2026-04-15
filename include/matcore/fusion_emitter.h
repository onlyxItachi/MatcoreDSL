#pragma once

#include <vector>

#include "matcore/fusion_analysis.h"
#include "matcore/kernel_ir.h"
#include "matcore/mlir_engine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace matcore {

class FusionMlirEmitter {
 public:
  static mlir::OwningOpRef<mlir::ModuleOp> Emit(
      const KernelIR &kernel, const FusedKernelPlan &plan,
      const std::vector<RuntimeTensorView> &tensors,
      const RequestedTargetProfile &target, mlir::MLIRContext &context);

 private:
  static mlir::OwningOpRef<mlir::ModuleOp> emitFamilyA(
      const KernelIR &kernel, const FusedKernelPlan &plan,
      const std::vector<RuntimeTensorView> &tensors,
      const RequestedTargetProfile &target, mlir::MLIRContext &context);

  static mlir::OwningOpRef<mlir::ModuleOp> emitFamilyB(
      const KernelIR &kernel, const FusedKernelPlan &plan,
      const std::vector<RuntimeTensorView> &tensors,
      const RequestedTargetProfile &target, mlir::MLIRContext &context);

  static mlir::OwningOpRef<mlir::ModuleOp> emitFamilyC(
      const KernelIR &kernel, const FusedKernelPlan &plan,
      const std::vector<RuntimeTensorView> &tensors,
      const RequestedTargetProfile &target, mlir::MLIRContext &context);
};

}  // namespace matcore
