#include "matcore/jit_runner.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "matcore/mlir_engine.h"

namespace matcore {
namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore JIT runner: " + message);
}

std::string targetName(TargetKind target) {
  switch (normalizeTarget(target)) {
    case TargetKind::kX86Auto:
      return "x86-auto";
    case TargetKind::kX86AVX2:
      return "x86-avx2";
    case TargetKind::kX86AVX512:
      return "x86-avx512";
    case TargetKind::kNvidiaDGPU:
      return "nvidia-dgpu";
    case TargetKind::kAmdIGPU:
      return "amd-igpu";
    case TargetKind::kAmdNPU:
      return "amd-npu";
    case TargetKind::kARM:
      return "arm";
    case TargetKind::kTPU:
      return "tpu";
    case TargetKind::kNVPTX:
    case TargetKind::kAMDGCN:
    case TargetKind::kNPU:
      break;
  }
  return "unknown";
}

std::string dtypeName(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return "float32";
    case TensorDType::kFloat16:
      return "float16";
    case TensorDType::kBFloat16:
      return "bfloat16";
  }
  return "unknown";
}

std::vector<std::string> buildSharedLibraryPaths(TargetKind target) {
  std::vector<std::string> libs = {
      "/usr/lib/llvm-18/lib/libmlir_runner_utils.so",
      "/usr/lib/llvm-18/lib/libmlir_c_runner_utils.so",
  };

  switch (normalizeTarget(target)) {
    case TargetKind::kNvidiaDGPU:
      libs.emplace_back("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so");
      libs.emplace_back("/lib/x86_64-linux-gnu/libcuda.so");
      break;
    case TargetKind::kAmdIGPU:
      libs.emplace_back("/lib/x86_64-linux-gnu/libamdhip64.so");
      break;
    default:
      break;
  }
  return libs;
}

std::string buildExecutionCacheKey(const KernelIR &kernel, TargetKind target,
                                   const std::vector<RuntimeTensorView> &tensors) {
  std::string key = kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  key += "|target=" + targetName(target);
  key += "|ops=" + std::to_string(kernel.ops.size());
  for (std::size_t i = 0; i < std::min<std::size_t>(3, tensors.size()); ++i) {
    const RuntimeTensorView &tensor = tensors[i];
    key += "|";
    key += tensor.symbol;
    key += ":";
    key += dtypeName(tensor.dtype);
    if (tensor.shape.size() >= 2) {
      key += ":";
      key += std::to_string(tensor.shape[0]);
      key += "x";
      key += std::to_string(tensor.shape[1]);
    }
  }
  return key;
}

std::unique_ptr<mlir::ExecutionEngine> takeEngine(
    llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>> engine) {
  if (!engine) {
    std::string error;
    llvm::handleAllErrors(engine.takeError(), [&](const llvm::ErrorInfoBase &base) {
      error = base.message();
    });
    fail("ExecutionEngine::create() failed: " + error);
  }
  return std::move(*engine);
}

void enforceExecutionPolicy(const LoweredModule &lowered) {
  if (lowered.executable) {
    return;
  }
  fail("target '" + targetName(lowered.target) + "' routed via '" +
       lowered.route_description + "' but execution is not enabled");
}

template <typename ElementT>
::StridedMemRefType<ElementT, 2>
makeMemRef2DDescriptor(const RuntimeTensorView &tensor) {
  if (tensor.data == nullptr) {
    fail("tensor '" + tensor.symbol + "' has null data pointer");
  }
  if (tensor.shape.size() != 2 || tensor.strides.size() != 2) {
    fail("tensor '" + tensor.symbol + "' must be rank-2 for JIT invocation");
  }

  auto *typed = reinterpret_cast<ElementT *>(tensor.data);
  ::StridedMemRefType<ElementT, 2> descriptor;
  descriptor.basePtr = typed;
  descriptor.data = typed;
  descriptor.offset = 0;
  descriptor.sizes[0] = tensor.shape[0];
  descriptor.sizes[1] = tensor.shape[1];
  descriptor.strides[0] = tensor.strides[0];
  descriptor.strides[1] = tensor.strides[1];
  return descriptor;
}

template <typename ElementT>
llvm::Error invokeWithTypedDescriptors(mlir::ExecutionEngine &engine,
                                       const std::string &entry_point,
                                       const RuntimeTensorView &lhs,
                                       const RuntimeTensorView &rhs,
                                       const RuntimeTensorView &out) {
  auto lhs_desc = makeMemRef2DDescriptor<ElementT>(lhs);
  auto rhs_desc = makeMemRef2DDescriptor<ElementT>(rhs);
  auto out_desc = makeMemRef2DDescriptor<ElementT>(out);

  llvm::Error ciface_error = engine.invoke(entry_point, lhs_desc, rhs_desc, out_desc);
  if (!ciface_error) {
    return llvm::Error::success();
  }

  std::string ciface_message = llvm::toString(std::move(ciface_error));
  std::vector<void *> packed_args = {&lhs_desc, &rhs_desc, &out_desc};
  llvm::Error packed_error = engine.invokePacked(entry_point, packed_args);
  if (!packed_error) {
    return llvm::Error::success();
  }

  std::string packed_message = llvm::toString(std::move(packed_error));
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "ciface invoke failed: %s; packed invoke failed: %s", ciface_message.c_str(),
      packed_message.c_str());
}

struct CachedExecution {
  std::unique_ptr<mlir::MLIRContext> context;
  LoweredModule lowered;
  std::unique_ptr<mlir::ExecutionEngine> engine;
};

std::shared_ptr<CachedExecution>
getOrCreateExecution(const KernelIR &kernel, TargetKind target,
                     const std::vector<RuntimeTensorView> &tensors) {
  static std::mutex cache_mutex;
  static auto *cache =
      new std::unordered_map<std::string, std::shared_ptr<CachedExecution>>();

  const std::string cache_key = buildExecutionCacheKey(kernel, target, tensors);
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache->find(cache_key);
    if (it != cache->end()) {
      return it->second;
    }
  }

  auto compiled = std::make_shared<CachedExecution>();
  compiled->context = std::make_unique<mlir::MLIRContext>();
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(registry);
  compiled->context->appendDialectRegistry(registry);
  compiled->context->loadAllAvailableDialects();

  compiled->lowered =
      MlirEngine::BuildAndLower(kernel, target, tensors, *compiled->context);
  enforceExecutionPolicy(compiled->lowered);

  const std::vector<std::string> shared_lib_storage = buildSharedLibraryPaths(target);
  llvm::SmallVector<llvm::StringRef, 8> shared_libs;
  shared_libs.reserve(shared_lib_storage.size());
  for (const std::string &path : shared_lib_storage) {
    shared_libs.push_back(path);
  }

  mlir::ExecutionEngineOptions options;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  options.sharedLibPaths = shared_libs;
  compiled->engine =
      takeEngine(mlir::ExecutionEngine::create(*compiled->lowered.module, options));

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto [it, inserted] = cache->emplace(cache_key, compiled);
  if (!inserted) {
    return it->second;
  }
  return compiled;
}

llvm::Error invokeCompiledKernel(const CachedExecution &compiled,
                                 const std::vector<RuntimeTensorView> &tensors) {
  if (tensors.size() < 3) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "matmul invocation requires 3 tensors");
  }
  if (compiled.lowered.lhs_tensor_index >= tensors.size() ||
      compiled.lowered.rhs_tensor_index >= tensors.size() ||
      compiled.lowered.out_tensor_index >= tensors.size()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "lowered tensor indices are out of range for runtime tensors");
  }

  const RuntimeTensorView &lhs = tensors[compiled.lowered.lhs_tensor_index];
  const RuntimeTensorView &rhs = tensors[compiled.lowered.rhs_tensor_index];
  const RuntimeTensorView &out = tensors[compiled.lowered.out_tensor_index];

  if (lhs.dtype != rhs.dtype || lhs.dtype != out.dtype) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "mixed dtype invocation is not supported");
  }

  switch (lhs.dtype) {
    case TensorDType::kFloat32:
      return invokeWithTypedDescriptors<float>(*compiled.engine,
                                               compiled.lowered.entry_point, lhs,
                                               rhs, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      // MLIR lowers f16/bf16 memref elements through 16-bit storage.
      return invokeWithTypedDescriptors<std::uint16_t>(
          *compiled.engine, compiled.lowered.entry_point, lhs, rhs, out);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported runtime dtype");
}

}  // namespace

void compileAndRun(const KernelIR &kernel, TargetKind target,
                   const std::vector<RuntimeTensorView> &tensors) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  std::shared_ptr<CachedExecution> compiled =
      getOrCreateExecution(kernel, target, tensors);
  if (llvm::Error error = invokeCompiledKernel(*compiled, tensors)) {
    const std::string message = llvm::toString(std::move(error));
    fail("failed to invoke JIT entrypoint '" + compiled->lowered.entry_point +
         "': " + message);
  }
}

}  // namespace matcore
