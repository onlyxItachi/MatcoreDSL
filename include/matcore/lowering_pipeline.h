#pragma once

#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "matcore/kernel_ir.h"

namespace matcore {

enum class LoweringRoute {
  kCpuVector,
  kNvidiaNvptx,
  kAmdRocdl,
  kAmdNpuScaffold,
};

struct MatmulLoweringSignature {
  TensorDType lhs_dtype = TensorDType::kFloat32;
  TensorDType rhs_dtype = TensorDType::kFloat32;
  TensorDType out_dtype = TensorDType::kFloat32;
  bool quantized_i8 = false;
};

struct LoweringPlan {
  LoweringRoute route = LoweringRoute::kCpuVector;
  std::string route_name;
  std::string route_description;
  bool executable = true;
};

LoweringPlan selectLoweringPlan(TargetKind target);

const char *routeName(LoweringRoute route);
std::string routeDescription(LoweringRoute route);

MatmulLoweringSignature decodeMatmulSignatureFromModule(mlir::ModuleOp module);

void configureLoweringPipeline(mlir::PassManager &pm, const LoweringPlan &plan,
                               const MatmulLoweringSignature &signature,
                               llvm::StringRef nvidia_chip = "sm_80");

void runLoweringPipeline(mlir::ModuleOp module, const LoweringPlan &plan,
                         const MatmulLoweringSignature &signature,
                         llvm::StringRef nvidia_chip = "sm_80");

}  // namespace matcore

