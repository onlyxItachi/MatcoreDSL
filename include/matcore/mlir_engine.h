#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "matcore/kernel_ir.h"

namespace matcore {

struct LoweredModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string entry_point;
  std::size_t lhs_tensor_index = 0;
  std::size_t rhs_tensor_index = 1;
  std::size_t out_tensor_index = 2;
};

class MlirEngine {
 public:
  static LoweredModule BuildAndLower(const KernelIR &kernel, TargetKind target,
                                     const std::vector<RuntimeTensorView> &tensors,
                                     mlir::MLIRContext &context);
};

}  // namespace matcore
