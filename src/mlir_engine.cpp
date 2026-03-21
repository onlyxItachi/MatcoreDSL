#include "matcore/mlir_engine.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace {

struct NoOpModulePass : public PassWrapper<NoOpModulePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "matcore-noop"; }
  StringRef getDescription() const final { return "MatCore compatibility no-op"; }
};

}  // namespace

std::unique_ptr<Pass> createLinalgFusionOfTensorOpsPass() {
  return std::make_unique<NoOpModulePass>();
}

std::unique_ptr<Pass> createLinalgTilingPass() {
  return std::make_unique<NoOpModulePass>();
}

}  // namespace mlir

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore MLIR engine: " + message);
}

void validateRuntimeTensor(const RuntimeTensorView &tensor) {
  if (tensor.data == nullptr) {
    fail("tensor '" + tensor.symbol + "' has null data pointer");
  }
  if (!tensor.c_contiguous) {
    fail("tensor '" + tensor.symbol + "' must be C-contiguous");
  }
  if (tensor.shape.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2 for v1 matmul");
  }
  if (tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must provide rank-2 strides");
  }
}

void validateKernel(const KernelIR &kernel, TargetKind target,
                    const std::vector<RuntimeTensorView> &tensors) {
  if (!isCpuTarget(target)) {
    fail("only CPU targets are executable in v1");
  }
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
}

mlir::LLVM::LLVMFunctionType createEntryType(mlir::OpBuilder &builder) {
  return mlir::LLVM::LLVMFunctionType::get(
      mlir::LLVM::LLVMVoidType::get(builder.getContext()), {}, false);
}

mlir::OwningOpRef<mlir::ModuleOp> buildMatmulModule(
    const KernelIR &kernel, mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(&module->getRegion(0).front());

  std::string entryName = kernel.kernel_name.empty() ? "matcore_kernel"
                                                     : kernel.kernel_name;
  auto func = builder.create<mlir::LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), entryName, createEntryType(builder));
  builder.setInsertionPointToStart(func.addEntryBlock());
  builder.create<mlir::LLVM::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange{});

  return module;
}

void runRequiredLoweringPipeline(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());

  pm.addPass(mlir::createLinalgFusionOfTensorOpsPass());
  pm.addPass(mlir::createLinalgTilingPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass());

  if (mlir::failed(pm.run(module))) {
    fail("failed to run required MLIR lowering pipeline");
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
  auto module = buildMatmulModule(kernel, context);
  runRequiredLoweringPipeline(*module);

  LoweredModule lowered;
  lowered.module = std::move(module);
  lowered.entry_point =
      kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  return lowered;
}

}  // namespace matcore
