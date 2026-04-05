#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "matcore/kernel_ir.h"
#include "matcore/target_registry.h"

namespace matcore {

class ObservabilityContext;

struct LoweredModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string entry_point;
  RequestedTargetProfile target_profile;
  ExecutionRequirements execution_requirements;
  std::string route_description;
  bool executable = true;
  std::size_t lhs_tensor_index = 0;
  std::size_t rhs_tensor_index = 1;
  std::size_t out_tensor_index = 2;
};

class MlirEngine {
 public:
  static LoweredModule BuildAndLower(
      const KernelIR &kernel, const RequestedTargetProfile &target_profile,
      const std::vector<RuntimeTensorView> &tensors, mlir::MLIRContext &context,
      ObservabilityContext *obs = nullptr);
};

}  // namespace matcore
