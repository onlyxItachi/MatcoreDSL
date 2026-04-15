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

struct KernelArgumentDesc {
  std::string symbol;
  TensorDType dtype = TensorDType::kFloat32;
  int rank = 2;
  bool is_input = true;
  bool is_output = false;
};

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
  std::size_t tensor_count = 3;
  bool needs_output_zeroing = true;
  std::vector<KernelArgumentDesc> arguments;
  std::vector<std::size_t> output_tensor_indices;
  int actual_reg_count = 0;
  bool reg_budget_exceeded = false;
};

class MlirEngine {
 public:
  static LoweredModule BuildAndLower(
      const KernelIR &kernel, const RequestedTargetProfile &target_profile,
      const std::vector<RuntimeTensorView> &tensors, mlir::MLIRContext &context,
      ObservabilityContext *obs = nullptr);
};

}  // namespace matcore
