#pragma once

#include <vector>

#include "matcore/kernel_ir.h"
#include "matcore/mlir_engine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace matcore {

class RegionMlirEmitter {
 public:
  static mlir::OwningOpRef<mlir::ModuleOp> Emit(
      const KernelIR &kernel, const std::vector<RuntimeTensorView> &tensors,
      const RequestedTargetProfile &target, mlir::MLIRContext &context);

 private:
  static mlir::OwningOpRef<mlir::ModuleOp> emitBlockAttnRes(
      const KernelIR &kernel, const RegionNode &node,
      const std::vector<RuntimeTensorView> &tensors,
      const RequestedTargetProfile &target, mlir::MLIRContext &context);
};

}  // namespace matcore
