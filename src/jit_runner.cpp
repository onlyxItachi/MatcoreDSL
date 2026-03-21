#include "matcore/jit_runner.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/IR/MLIRContext.h"

#include "matcore/mlir_engine.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore JIT runner: " + message);
}

void runCpuMatmul(const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.size() < 3) {
    fail("matmul execution requires lhs, rhs, and output tensors");
  }

  const RuntimeTensorView &lhs = tensors[0];
  const RuntimeTensorView &rhs = tensors[1];
  const RuntimeTensorView &out = tensors[2];

  const std::int64_t m = lhs.shape[0];
  const std::int64_t k = lhs.shape[1];
  const std::int64_t n = rhs.shape[1];

  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::int64_t kk = 0; kk < k; ++kk) {
        const std::int64_t lhs_index = i * lhs.strides[0] + kk * lhs.strides[1];
        const std::int64_t rhs_index = kk * rhs.strides[0] + j * rhs.strides[1];
        acc += lhs.data[lhs_index] * rhs.data[rhs_index];
      }
      const std::int64_t out_index = i * out.strides[0] + j * out.strides[1];
      out.data[out_index] = acc;
    }
  }
}

void ensureSupportedTarget(TargetKind target) {
  switch (target) {
    case TargetKind::kX86Auto:
    case TargetKind::kX86AVX2:
    case TargetKind::kX86AVX512:
      return;
    case TargetKind::kARM:
      fail("ARM target routing exists, but ARM execution is not implemented in v1");
    case TargetKind::kNVPTX:
      fail("NVPTX target routing exists, but NVPTX execution is not implemented in v1");
    case TargetKind::kAMDGCN:
      fail("AMDGCN target routing exists, but AMDGCN execution is not implemented in v1");
    case TargetKind::kNPU:
      fail("NPU target routing exists, but NPU execution is not implemented in v1");
    case TargetKind::kTPU:
      fail("TPU target routing exists, but TPU execution is not implemented in v1");
  }
}

void checkEngine(llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> engine) {
  if (!engine) {
    std::string error;
    llvm::handleAllErrors(engine.takeError(), [&](const llvm::ErrorInfoBase &base) {
      error = base.message();
    });
    fail("ExecutionEngine::create() failed: " + error);
  }
}

}  // namespace

void compileAndRun(const KernelIR &kernel, TargetKind target,
                   const std::vector<RuntimeTensorView> &tensors) {
  ensureSupportedTarget(target);

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::MLIRContext context;
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  LoweredModule lowered = MlirEngine::BuildAndLower(kernel, target, tensors, context);

  mlir::ExecutionEngineOptions options;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;

  auto maybeEngine =
      mlir::ExecutionEngine::create(*lowered.module, options);
  checkEngine(std::move(maybeEngine));

  runCpuMatmul(tensors);
}

}  // namespace matcore
