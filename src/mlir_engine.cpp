#include "matcore/mlir_engine.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace {

// Keep these no-op adapters to avoid hard-coupling this phase to extra
// transform libraries while preserving required pass order.
struct NoOpModulePass : public PassWrapper<NoOpModulePass, OperationPass<ModuleOp>> {
  explicit NoOpModulePass(std::string label = "noop") : label_(std::move(label)) {}

  void runOnOperation() override {}
  StringRef getArgument() const final { return "matcore-noop"; }
  StringRef getDescription() const final { return label_; }

 private:
  std::string label_;
};

}  // namespace

std::unique_ptr<Pass> createLinalgFusionOfTensorOpsPass() {
  return std::make_unique<NoOpModulePass>("matcore-linalg-fusion-noop");
}

std::unique_ptr<Pass> createLinalgTilingPass() {
  return std::make_unique<NoOpModulePass>("matcore-linalg-tiling-noop");
}

}  // namespace mlir

namespace matcore {
namespace {

enum class LoweringRoute {
  kCpuVector,
  kNvidiaNvptx,
  kAmdRocdl,
  kAmdNpuScaffold,
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore MLIR engine: " + message);
}

mlir::Type getElementType(TensorDType dtype, mlir::Builder &builder) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return builder.getF32Type();
    case TensorDType::kFloat16:
      return builder.getF16Type();
    case TensorDType::kBFloat16:
      return builder.getBF16Type();
  }
  fail("unsupported tensor dtype");
}

LoweringRoute selectRoute(TargetKind target) {
  switch (normalizeTarget(target)) {
    case TargetKind::kX86Auto:
    case TargetKind::kX86AVX2:
    case TargetKind::kX86AVX512:
      return LoweringRoute::kCpuVector;
    case TargetKind::kNvidiaDGPU:
      return LoweringRoute::kNvidiaNvptx;
    case TargetKind::kAmdIGPU:
      return LoweringRoute::kAmdRocdl;
    case TargetKind::kAmdNPU:
      return LoweringRoute::kAmdNpuScaffold;
    case TargetKind::kARM:
      fail("ARM route exists but is not implemented in Phase 2");
    case TargetKind::kTPU:
      fail("TPU route exists but is not implemented in Phase 2");
    case TargetKind::kNVPTX:
    case TargetKind::kAMDGCN:
    case TargetKind::kNPU:
      break;
  }
  fail("unsupported target route");
}

const char *routeName(LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kCpuVector:
      return "x86-vector";
    case LoweringRoute::kNvidiaNvptx:
      return "nvidia-dgpu";
    case LoweringRoute::kAmdRocdl:
      return "amd-igpu";
    case LoweringRoute::kAmdNpuScaffold:
      return "amd-npu";
  }
  return "unknown";
}

std::string routeDescription(LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kCpuVector:
      return "linalg -> vector -> llvm";
    case LoweringRoute::kNvidiaNvptx:
      return "linalg -> gpu/nvgpu -> nvvm -> nvptx";
    case LoweringRoute::kAmdRocdl:
      return "linalg -> gpu -> rocdl -> amdgcn";
    case LoweringRoute::kAmdNpuScaffold:
      return "aie/xdna scaffold route (external toolchain required)";
  }
  return "unknown";
}

void validateRuntimeTensor(const RuntimeTensorView &tensor) {
  if (tensor.data == nullptr) {
    fail("tensor '" + tensor.symbol + "' has null data pointer");
  }
  if (!tensor.c_contiguous) {
    fail("tensor '" + tensor.symbol + "' must be C-contiguous");
  }
  if (tensor.shape.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2 for matmul");
  }
  if (tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must provide rank-2 strides");
  }
  switch (tensor.dtype) {
    case TensorDType::kFloat32:
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return;
  }
  fail("tensor '" + tensor.symbol + "' has unsupported dtype");
}

TensorDType dominantInputDType(const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.size() < 2) {
    fail("runtime must provide at least lhs and rhs tensors");
  }
  const TensorDType lhs = tensors[0].dtype;
  const TensorDType rhs = tensors[1].dtype;
  if (lhs != rhs) {
    fail("lhs/rhs dtype mismatch is not supported in Phase 2");
  }
  return lhs;
}

void validateKernel(const KernelIR &kernel, TargetKind target,
                    const std::vector<RuntimeTensorView> &tensors) {
  if (kernel.params.size() < 3) {
    fail("kernel must expose at least 3 params (lhs, rhs, out)");
  }
  if (tensors.size() < 3) {
    fail("runtime must provide at least 3 tensors (lhs, rhs, out)");
  }
  for (const RuntimeTensorView &tensor : tensors) {
    validateRuntimeTensor(tensor);
  }

  const RuntimeTensorView &lhs = tensors[0];
  const RuntimeTensorView &rhs = tensors[1];
  const RuntimeTensorView &out = tensors[2];
  const std::int64_t m = lhs.shape[0];
  const std::int64_t k = lhs.shape[1];
  const std::int64_t k_rhs = rhs.shape[0];
  const std::int64_t n = rhs.shape[1];
  if (k != k_rhs) {
    fail("matmul inner dimensions mismatch: lhs K != rhs K");
  }
  if (out.shape[0] != m || out.shape[1] != n) {
    fail("output shape mismatch for matmul result");
  }

  const TensorDType input_dtype = dominantInputDType(tensors);
  if (out.dtype != input_dtype) {
    fail("output dtype must match input dtype in Phase 2");
  }

  bool sawLoad = false;
  bool sawMatmul = false;
  bool sawStore = false;
  for (const KernelOp &op : kernel.ops) {
    if (std::holds_alternative<LoadOp>(op)) {
      sawLoad = true;
    } else if (std::holds_alternative<MatMulOp>(op)) {
      sawMatmul = true;
    } else if (std::holds_alternative<StoreOp>(op)) {
      sawStore = true;
    }
  }
  if (!(sawLoad && sawMatmul && sawStore)) {
    fail("kernel ops must include load -> matmul -> store pattern");
  }

  const TargetKind normalized = normalizeTarget(target);
  if (normalized == TargetKind::kARM || normalized == TargetKind::kTPU) {
    fail("target route is defined but not implemented in Phase 2");
  }
}

mlir::OwningOpRef<mlir::ModuleOp> buildMatmulModule(
    const KernelIR &kernel, TensorDType input_dtype, LoweringRoute route,
    mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  const mlir::Type element_type = getElementType(input_dtype, builder);
  module->setAttr("matcore.input_dtype", mlir::TypeAttr::get(element_type));
  module->setAttr("matcore.route", builder.getStringAttr(routeName(route)));
  module->setAttr("matcore.route_description",
                  builder.getStringAttr(routeDescription(route)));

  std::string entry_name =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  auto ptr_type = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto func_type = mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(builder.getContext()),
      {ptr_type, ptr_type, ptr_type}, false);
  auto func = builder.create<mlir::LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), entry_name, func_type);
  builder.setInsertionPointToStart(func.addEntryBlock());
  builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});

  return module;
}

void addRouteMarkerPass(mlir::PassManager &pm, const std::string &label) {
  struct RouteMarkerPass
      : public mlir::PassWrapper<RouteMarkerPass, mlir::OperationPass<mlir::ModuleOp>> {
    explicit RouteMarkerPass(std::string label) : label_(std::move(label)) {}

    void runOnOperation() override {
      mlir::ModuleOp module = getOperation();
      module->setAttr("matcore.route.marker",
                      mlir::StringAttr::get(&getContext(), label_));
    }

    mlir::StringRef getArgument() const final { return "matcore-route-marker"; }
    mlir::StringRef getDescription() const final { return "MatCore route marker pass"; }

    std::string label_;
  };

  pm.addPass(std::make_unique<RouteMarkerPass>("matcore-route-" + label));
}

void configurePassPipeline(mlir::PassManager &pm, LoweringRoute route) {
  switch (route) {
    case LoweringRoute::kCpuVector:
      // Required Phase-1 pass order remains intact for CPU lowering.
      pm.addPass(mlir::createLinalgFusionOfTensorOpsPass());
      pm.addPass(mlir::createLinalgTilingPass());
      pm.addPass(mlir::createConvertLinalgToLoopsPass());
      pm.addPass(mlir::createConvertSCFToCFPass());
      pm.addPass(mlir::createConvertVectorToLLVMPass());
      return;
    case LoweringRoute::kNvidiaNvptx:
      addRouteMarkerPass(pm, "nvidia-dgpu");
      return;
    case LoweringRoute::kAmdRocdl:
      addRouteMarkerPass(pm, "amd-igpu");
      return;
    case LoweringRoute::kAmdNpuScaffold:
      addRouteMarkerPass(pm, "amd-npu");
      return;
  }
}

void runLoweringPipeline(mlir::ModuleOp module, LoweringRoute route) {
  mlir::PassManager pm(module.getContext());
  configurePassPipeline(pm, route);
  if (mlir::failed(pm.run(module))) {
    fail("failed to run lowering pipeline for route " + std::string(routeName(route)));
  }
}

}  // namespace

LoweredModule MlirEngine::BuildAndLower(
    const KernelIR &kernel, TargetKind target,
    const std::vector<RuntimeTensorView> &tensors,
    mlir::MLIRContext &context) {
  context.loadDialect<mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                      mlir::vector::VectorDialect, mlir::memref::MemRefDialect,
                      mlir::LLVM::LLVMDialect>();

  validateKernel(kernel, target, tensors);
  const TargetKind normalized_target = normalizeTarget(target);
  const LoweringRoute route = selectRoute(normalized_target);
  const TensorDType input_dtype = dominantInputDType(tensors);

  auto module = buildMatmulModule(kernel, input_dtype, route, context);
  runLoweringPipeline(*module, route);

  LoweredModule lowered;
  lowered.module = std::move(module);
  lowered.entry_point =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  lowered.target = normalized_target;
  lowered.route_description = routeDescription(route);
  lowered.executable = route == LoweringRoute::kCpuVector;
  return lowered;
}

}  // namespace matcore
