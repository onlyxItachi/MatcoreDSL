#pragma once

#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "matcore/kernel_ir.h"

namespace matcore {

class ObservabilityContext;

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
  TargetKind target_kind = TargetKind::kX86Auto;
  int nvidia_sm_major = 0;
  int nvidia_sm_minor = 0;
  int matmul_m = -1;
  int matmul_n = -1;
  int matmul_k = -1;
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
                               llvm::StringRef nvidia_chip = "sm_80",
                               llvm::StringRef amd_chip = "gfx90a",
                               mlir::ModuleOp module = mlir::ModuleOp(),
                               ObservabilityContext *obs = nullptr);

void runLoweringPipeline(mlir::ModuleOp module, const LoweringPlan &plan,
                         const MatmulLoweringSignature &signature,
                         llvm::StringRef nvidia_chip = "sm_80",
                         llvm::StringRef amd_chip = "gfx90a",
                         ObservabilityContext *obs = nullptr);

}  // namespace matcore
