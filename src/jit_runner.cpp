#include "matcore/jit_runner.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <dlfcn.h>
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "matcore/gpu_runtime_symbols.h"
#include "matcore/mlir_engine.h"
#include "matcore/runtime_capabilities.h"

#if defined(__GNUC__) || defined(__clang__)
#define MATCORE_JIT_RUNTIME_EXPORT __attribute__((visibility("default")))
#else
#define MATCORE_JIT_RUNTIME_EXPORT
#endif

extern "C" MATCORE_JIT_RUNTIME_EXPORT void *_mlir_malloc(std::size_t size) {
  return std::malloc(size);
}

extern "C" MATCORE_JIT_RUNTIME_EXPORT void _mlir_free(void *ptr) {
  std::free(ptr);
}

namespace matcore {
namespace {

namespace fs = std::filesystem;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore JIT runner: " + message);
}

std::string targetName(const RequestedTargetProfile &target_profile) {
  if (!target_profile.canonical.empty()) {
    return target_profile.canonical;
  }
  return CanonicalTargetString(target_profile);
}

std::string dtypeName(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kFloat32:
      return "float32";
    case TensorDType::kFloat16:
      return "float16";
    case TensorDType::kBFloat16:
      return "bfloat16";
    case TensorDType::kInt8:
      return "int8";
    case TensorDType::kInt32:
      return "int32";
    case TensorDType::kFloat8E4M3FN:
      return "float8_e4m3fn";
  }
  return "unknown";
}

bool isX86Target(const RequestedTargetProfile &target_profile) {
  const TargetKind normalized = normalizeTarget(target_profile.kind);
  return normalized == TargetKind::kX86Auto ||
         normalized == TargetKind::kX86AVX2 ||
         normalized == TargetKind::kX86AVX512;
}

struct X86TargetProfile {
  std::string cpu;
  std::string features;
  std::string cache_tag;
};

struct DiskCacheArtifacts {
  fs::path root_dir;
  fs::path artifact_dir;
  fs::path shared_object_path;
  fs::path object_path;
};

enum class ExecutionBackend {
  kExecutionEngine,
  kSharedLibrary,
};

constexpr std::string_view kDiskCacheVersion =
    "matcore-phase4-cache-v4-nvidia-shared-mma-hotfix";

void addFeature(llvm::SubtargetFeatures &subtarget, llvm::StringRef feature_name,
                bool enabled = true) {
  subtarget.AddFeature(feature_name, enabled);
}

bool runtimeHasFeature(const RuntimeCapabilities &runtime, llvm::StringRef name) {
  return SupportsX86Feature(runtime, name.str());
}

std::optional<X86TargetProfile> resolveX86TargetProfile(
    const RequestedTargetProfile &target_profile,
    const RuntimeCapabilities &runtime) {
  if (!isX86Target(target_profile)) {
    return std::nullopt;
  }

  const TargetKind normalized = normalizeTarget(target_profile.kind);
  X86TargetProfile profile;
  llvm::SubtargetFeatures subtarget;

  if (normalized == TargetKind::kX86AVX512 ||
      (normalized == TargetKind::kX86Auto &&
       runtimeHasFeature(runtime, "avx512f"))) {
    profile.cpu = "generic";
    for (llvm::StringRef feature :
         {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512cd", "avx2",
          "fma", "f16c"}) {
      addFeature(subtarget, feature);
    }
    for (llvm::StringRef feature :
         {"avx512bf16", "avx512fp16", "avx512vnni", "avxvnni", "avxvnniint8",
          "amx-bf16", "amx-int8", "amx-tile"}) {
      if (runtimeHasFeature(runtime, feature)) {
        addFeature(subtarget, feature);
      }
    }
    profile.cache_tag = "x86-tier=avx512";
  } else if (normalized == TargetKind::kX86AVX2 ||
             (normalized == TargetKind::kX86Auto &&
              runtimeHasFeature(runtime, "avx2"))) {
    profile.cpu = "generic";
    addFeature(subtarget, "avx2");
    for (llvm::StringRef feature : {"fma", "f16c"}) {
      addFeature(subtarget, feature);
    }
    for (llvm::StringRef feature : {"avx512f", "avx512bw", "avx512vl",
                                    "avx512dq", "avx512cd", "avx512bf16",
                                    "avx512fp16"}) {
      addFeature(subtarget, feature, /*enabled=*/false);
    }
    profile.cache_tag = "x86-tier=avx2";
  } else {
    profile.cpu = llvm::sys::getHostCPUName().str();
    if (profile.cpu.empty()) {
      profile.cpu = "generic";
    }
    profile.cache_tag = "x86-tier=baseline";
  }

  profile.features = subtarget.getString();
  profile.cache_tag += "|cpu=" + profile.cpu;
  profile.cache_tag += "|mattr=" + profile.features;
  return profile;
}

std::unique_ptr<llvm::TargetMachine>
createTargetMachine(const std::optional<X86TargetProfile> &x86_profile) {
  if (!x86_profile.has_value()) {
    return nullptr;
  }

  const std::string triple = llvm::sys::getProcessTriple();
  std::string error;
  const llvm::Target *target_info =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (target_info == nullptr) {
    fail("failed to lookup target for triple '" + triple + "': " + error);
  }

  llvm::TargetOptions options;
  llvm::TargetMachine *raw_tm = target_info->createTargetMachine(
      triple, x86_profile->cpu, x86_profile->features, options,
      /*RM=*/std::nullopt, /*CM=*/std::nullopt,
      /*OL=*/llvm::CodeGenOptLevel::Default, /*JIT=*/true);
  if (raw_tm == nullptr) {
    fail("failed to create TargetMachine for x86 profile '" +
         x86_profile->cache_tag + "'");
  }

  return std::unique_ptr<llvm::TargetMachine>(raw_tm);
}

std::vector<std::string> buildSharedLibraryPaths(
    const RequestedTargetProfile &target_profile) {
  std::vector<std::string> libs = {
      "/usr/lib/llvm-18/lib/libmlir_runner_utils.so",
      "/usr/lib/llvm-18/lib/libmlir_c_runner_utils.so",
  };

  switch (normalizeTarget(target_profile.kind)) {
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

std::string buildExecutionCacheKey(
    const KernelIR &kernel, const RequestedTargetProfile &target_profile,
    const std::vector<RuntimeTensorView> &tensors,
    const std::optional<X86TargetProfile> &x86_profile) {
  std::string key = kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
  key += "|target=" + targetName(target_profile);
  if (x86_profile.has_value()) {
    key += "|";
    key += x86_profile->cache_tag;
  }
  key += "|ops=" + std::to_string(kernel.ops.size());
  if (kernel.global_quantization.enabled) {
    key += "|gq=" + std::to_string(kernel.global_quantization.scale);
    key += ":";
    key += std::to_string(kernel.global_quantization.zero_point);
  }
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
    if (tensor.quantization.enabled) {
      key += ":q=";
      key += std::to_string(tensor.quantization.scale);
      key += ":";
      key += std::to_string(tensor.quantization.zero_point);
    }
  }
  return key;
}

std::string entryPointName(const KernelIR &kernel) {
  return kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
}

std::string buildStableCacheHash(const std::string &key) {
  llvm::MD5 hasher;
  hasher.update(llvm::StringRef(kDiskCacheVersion.data(), kDiskCacheVersion.size()));
  hasher.update(key);
  llvm::MD5::MD5Result result;
  hasher.final(result);
  return result.digest().str().str();
}

fs::path currentExtensionPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(&registerGpuRuntimeSymbols), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  std::error_code ec;
  fs::path path = fs::weakly_canonical(fs::path(info.dli_fname), ec);
  if (ec) {
    return fs::path(info.dli_fname);
  }
  return path;
}

fs::path cacheRootPath() {
  if (const char *override_dir = std::getenv("MATCORE_CACHE_DIR")) {
    if (*override_dir != '\0') {
      return fs::path(override_dir);
    }
  }
  const fs::path extension_path = currentExtensionPath();
  if (!extension_path.empty()) {
    return extension_path.parent_path().parent_path() / ".matcore_cache";
  }
  return fs::current_path() / ".matcore_cache";
}

DiskCacheArtifacts buildDiskCacheArtifacts(const std::string &cache_key) {
  DiskCacheArtifacts artifacts;
  artifacts.root_dir = cacheRootPath();
  artifacts.artifact_dir = artifacts.root_dir / buildStableCacheHash(cache_key);
  artifacts.shared_object_path = artifacts.artifact_dir / "kernel.so";
  artifacts.object_path = artifacts.artifact_dir / "kernel.o";
  return artifacts;
}

bool isDiskCacheSupported(const RequestedTargetProfile &target_profile) {
  switch (normalizeTarget(target_profile.kind)) {
    case TargetKind::kX86Auto:
    case TargetKind::kX86AVX2:
    case TargetKind::kX86AVX512:
    case TargetKind::kNvidiaDGPU:
      return true;
    default:
      return false;
  }
}

void ensureCacheDirectory(const fs::path &path) {
  std::error_code ec;
  fs::create_directories(path, ec);
  if (ec) {
    fail("failed to create cache directory '" + path.string() + "': " +
         ec.message());
  }
}

void removeArtifactIfExists(const fs::path &path) {
  std::error_code ec;
  fs::remove(path, ec);
}

void removeDiskCacheArtifacts(const DiskCacheArtifacts &artifacts) {
  removeArtifactIfExists(artifacts.shared_object_path);
  removeArtifactIfExists(artifacts.object_path);
}

bool fileExists(const fs::path &path) {
  std::error_code ec;
  return fs::exists(path, ec) && !ec;
}

bool keepDiskCacheObjectFile() {
  const char *raw = std::getenv("MATCORE_KEEP_CACHE_OBJECT");
  return raw != nullptr && std::string(raw) == "1";
}

std::vector<std::string> buildSharedLibraryLinkInputs(
    const RequestedTargetProfile &target_profile) {
  std::vector<std::string> inputs = {
      "/usr/lib/llvm-18/lib/libmlir_runner_utils.so",
      "/usr/lib/llvm-18/lib/libmlir_c_runner_utils.so",
  };
  if (normalizeTarget(target_profile.kind) == TargetKind::kNvidiaDGPU) {
    const fs::path extension_path = currentExtensionPath();
    if (extension_path.empty()) {
      fail("unable to resolve the current MatCore extension path for GPU cache "
           "linking");
    }
    inputs.push_back(extension_path.string());
  }
  return inputs;
}

std::vector<std::string> buildSharedLibraryRPaths(
    const RequestedTargetProfile &target_profile) {
  std::unordered_set<std::string> unique_rpaths;
  unique_rpaths.insert("/usr/lib/llvm-18/lib");
  if (normalizeTarget(target_profile.kind) == TargetKind::kNvidiaDGPU) {
    const fs::path extension_path = currentExtensionPath();
    if (!extension_path.empty()) {
      unique_rpaths.insert(extension_path.parent_path().string());
    }
  }

  std::vector<std::string> rpaths(unique_rpaths.begin(), unique_rpaths.end());
  std::sort(rpaths.begin(), rpaths.end());
  return rpaths;
}

void linkObjectFileToSharedLibrary(const DiskCacheArtifacts &artifacts,
                                   const RequestedTargetProfile &target_profile) {
  std::vector<std::string> owned_args = {
      "/usr/bin/clang++",
      "-shared",
      "-fPIC",
      "-o",
      artifacts.shared_object_path.string(),
      artifacts.object_path.string(),
  };
  for (const std::string &rpath : buildSharedLibraryRPaths(target_profile)) {
    owned_args.push_back("-Wl,-rpath," + rpath);
  }
  for (const std::string &input : buildSharedLibraryLinkInputs(target_profile)) {
    owned_args.push_back(input);
  }

  llvm::SmallVector<llvm::StringRef, 16> args;
  args.reserve(owned_args.size());
  for (const std::string &arg : owned_args) {
    args.push_back(arg);
  }

  std::string err_msg;
  bool execution_failed = false;
  const int result = llvm::sys::ExecuteAndWait(
      owned_args.front(), args, std::nullopt, {}, /*SecondsToWait=*/0,
      /*MemoryLimit=*/0, &err_msg, &execution_failed);
  if (result != 0 || execution_failed) {
    removeArtifactIfExists(artifacts.shared_object_path);
    fail("failed to link cached shared object '" +
         artifacts.shared_object_path.string() + "' (result=" +
         std::to_string(result) + "): " +
         (err_msg.empty() ? std::string("no linker error message") : err_msg));
  }
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

void registerExecutionRuntimeSymbols(mlir::ExecutionEngine &engine) {
  engine.registerSymbols([](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    auto add_symbol = [&](llvm::StringRef name, auto *fn) {
      symbols[interner(name)] = llvm::orc::ExecutorSymbolDef(
          llvm::orc::ExecutorAddr::fromPtr(fn), llvm::JITSymbolFlags::Exported);
    };

    add_symbol("_mlir_malloc", &_mlir_malloc);
    add_symbol("_mlir_free", &_mlir_free);
    return symbols;
  });
}

void enforceExecutionPolicy(const LoweredModule &lowered) {
  if (lowered.executable) {
    return;
  }
  fail("target '" + targetName(lowered.target_profile) + "' routed via '" +
       lowered.route_description + "' but execution is not enabled");
}

struct CachedExecution {
  ~CachedExecution() {
    if (shared_library_handle != nullptr) {
      dlclose(shared_library_handle);
    }
  }

  ExecutionBackend backend = ExecutionBackend::kExecutionEngine;
  std::unique_ptr<mlir::MLIRContext> context;
  LoweredModule lowered;
  std::unique_ptr<mlir::ExecutionEngine> engine;
  void *shared_library_handle = nullptr;
  void (*ciface_entrypoint)(void *, void *, void *) = nullptr;
};

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

template <typename LhsElementT, typename RhsElementT, typename OutElementT>
llvm::Error invokeWithTypedDescriptors(const CachedExecution &compiled,
                                       const std::string &entry_point,
                                       const RuntimeTensorView &lhs,
                                       const RuntimeTensorView &rhs,
                                       const RuntimeTensorView &out) {
  auto lhs_desc = makeMemRef2DDescriptor<LhsElementT>(lhs);
  auto rhs_desc = makeMemRef2DDescriptor<RhsElementT>(rhs);
  auto out_desc = makeMemRef2DDescriptor<OutElementT>(out);

  const std::string adapter_name = std::string("_mlir_ciface_") + entry_point;
  void *lhs_arg = &lhs_desc;
  void *rhs_arg = &rhs_desc;
  void *out_arg = &out_desc;
  // The C-interface wrapper takes pointer-valued arguments, so the packed call
  // expects addresses of those pointer values, not the descriptor storage itself.
  std::vector<void *> packed_args = {&lhs_arg, &rhs_arg, &out_arg};
  if (compiled.backend == ExecutionBackend::kSharedLibrary) {
    if (compiled.ciface_entrypoint == nullptr) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cached shared library entrypoint missing");
    }
    compiled.ciface_entrypoint(&lhs_desc, &rhs_desc, &out_desc);
    return llvm::Error::success();
  }

  if (compiled.engine == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "execution engine is not initialized");
  }

  llvm::Error packed_error = compiled.engine->invokePacked(adapter_name, packed_args);
  if (!packed_error) {
    return llvm::Error::success();
  }

  std::string packed_message = llvm::toString(std::move(packed_error));
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(), "packed invoke failed: %s",
      packed_message.c_str());
}

template <typename LhsElementT, typename RhsElementT>
llvm::Error invokeWithOutputType(const CachedExecution &compiled,
                                 const std::string &entry_point,
                                 const RuntimeTensorView &lhs,
                                 const RuntimeTensorView &rhs,
                                 const RuntimeTensorView &out) {
  switch (out.dtype) {
    case TensorDType::kFloat32:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, float>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, std::uint16_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kInt8:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, std::int8_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kInt32:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, std::int32_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kFloat8E4M3FN:
      return invokeWithTypedDescriptors<LhsElementT, RhsElementT, std::uint8_t>(
          compiled, entry_point, lhs, rhs, out);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported output dtype");
}

template <typename LhsElementT>
llvm::Error invokeWithRhsType(const CachedExecution &compiled,
                              const std::string &entry_point,
                              const RuntimeTensorView &lhs,
                              const RuntimeTensorView &rhs,
                              const RuntimeTensorView &out) {
  switch (rhs.dtype) {
    case TensorDType::kFloat32:
      return invokeWithOutputType<LhsElementT, float>(compiled, entry_point, lhs,
                                                      rhs, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return invokeWithOutputType<LhsElementT, std::uint16_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kInt8:
      return invokeWithOutputType<LhsElementT, std::int8_t>(compiled, entry_point,
                                                            lhs, rhs, out);
    case TensorDType::kInt32:
      return invokeWithOutputType<LhsElementT, std::int32_t>(
          compiled, entry_point, lhs, rhs, out);
    case TensorDType::kFloat8E4M3FN:
      return invokeWithOutputType<LhsElementT, std::uint8_t>(
          compiled, entry_point, lhs, rhs, out);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported rhs dtype");
}

const mlir::DialectRegistry &sharedDialectRegistry() {
  static const mlir::DialectRegistry registry = [] {
    mlir::DialectRegistry dialect_registry;
    mlir::registerAllToLLVMIRTranslations(dialect_registry);
    mlir::NVVM::registerNVVMTargetInterfaceExternalModels(dialect_registry);
    mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(dialect_registry);
    return dialect_registry;
  }();
  return registry;
}

const RuntimeCapabilities &cachedRuntimeCapabilities() {
  static const RuntimeCapabilities runtime = DetectRuntimeCapabilities();
  return runtime;
}

RequestedTargetProfile resolveCompilationTargetProfile(
    const RequestedTargetProfile &target_profile,
    const RuntimeCapabilities &runtime) {
  RequestedTargetProfile resolved = target_profile;
  if (normalizeTarget(resolved.kind) == TargetKind::kNvidiaDGPU &&
      !resolved.nvidia_sm_major.has_value() &&
      !resolved.nvidia_sm_minor.has_value() && runtime.nvidia.device_present &&
      runtime.nvidia.compute_major > 0) {
    resolved.nvidia_sm_major = runtime.nvidia.compute_major;
    resolved.nvidia_sm_minor = runtime.nvidia.compute_minor;
    resolved.canonical = CanonicalTargetString(resolved);
  }
  return resolved;
}

std::shared_ptr<CachedExecution> tryLoadDiskCachedExecution(
    const KernelIR &kernel, const RequestedTargetProfile &compile_target,
    const ExecutionRequirements &execution_requirements,
    const DiskCacheArtifacts &artifacts) {
  if (!isDiskCacheSupported(compile_target) ||
      !fileExists(artifacts.shared_object_path)) {
    return nullptr;
  }

  void *handle = dlopen(artifacts.shared_object_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    removeDiskCacheArtifacts(artifacts);
    return nullptr;
  }

  const std::string entry_point = entryPointName(kernel);
  const std::string symbol_name = std::string("_mlir_ciface_") + entry_point;
  void *symbol = dlsym(handle, symbol_name.c_str());
  if (symbol == nullptr) {
    dlclose(handle);
    removeDiskCacheArtifacts(artifacts);
    return nullptr;
  }

  auto compiled = std::make_shared<CachedExecution>();
  compiled->backend = ExecutionBackend::kSharedLibrary;
  compiled->shared_library_handle = handle;
  compiled->ciface_entrypoint =
      reinterpret_cast<void (*)(void *, void *, void *)>(symbol);
  compiled->lowered.entry_point = entry_point;
  compiled->lowered.target_profile = compile_target;
  compiled->lowered.execution_requirements = execution_requirements;
  compiled->lowered.route_description = "disk-cached shared object";
  compiled->lowered.executable = true;
  return compiled;
}

void persistExecutionToDiskCache(const CachedExecution &compiled,
                                 const RequestedTargetProfile &compile_target,
                                 const DiskCacheArtifacts &artifacts) {
  if (compiled.engine == nullptr || !isDiskCacheSupported(compile_target)) {
    return;
  }
  if (fileExists(artifacts.shared_object_path)) {
    return;
  }

  ensureCacheDirectory(artifacts.artifact_dir);
  removeArtifactIfExists(artifacts.object_path);
  compiled.engine->dumpToObjectFile(artifacts.object_path.string());
  if (!fileExists(artifacts.object_path)) {
    fail("object dump for cache artifact '" + artifacts.object_path.string() +
         "' did not produce a file");
  }
  linkObjectFileToSharedLibrary(artifacts, compile_target);
  if (!keepDiskCacheObjectFile()) {
    removeArtifactIfExists(artifacts.object_path);
  }
}

std::shared_ptr<CachedExecution>
getOrCreateExecution(const KernelIR &kernel,
                     const RequestedTargetProfile &target_profile,
                     const std::vector<RuntimeTensorView> &tensors,
                     const RuntimeCapabilities &runtime) {
  static std::mutex cache_mutex;
  static auto *cache =
      new std::unordered_map<std::string, std::shared_ptr<CachedExecution>>();

  const RequestedTargetProfile compile_target =
      resolveCompilationTargetProfile(target_profile, runtime);
  const ExecutionRequirements requested_requirements =
      BuildExecutionRequirements(target_profile);
  std::string upfront_denial_reason;
  if (!CanExecuteOnHost(runtime, requested_requirements, &upfront_denial_reason)) {
    fail(FormatExecutionDeniedMessage(target_profile, upfront_denial_reason));
  }
  const std::optional<X86TargetProfile> x86_profile =
      resolveX86TargetProfile(compile_target, runtime);
  const std::string cache_key =
      buildExecutionCacheKey(kernel, compile_target, tensors, x86_profile);
  const DiskCacheArtifacts disk_artifacts = buildDiskCacheArtifacts(cache_key);
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache->find(cache_key);
    if (it != cache->end()) {
      return it->second;
    }
  }
  if (std::shared_ptr<CachedExecution> disk_cached = tryLoadDiskCachedExecution(
          kernel, compile_target, requested_requirements, disk_artifacts)) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto [it, inserted] = cache->emplace(cache_key, disk_cached);
    if (!inserted) {
      return it->second;
    }
    return disk_cached;
  }

  auto compiled = std::make_shared<CachedExecution>();
  compiled->context = std::make_unique<mlir::MLIRContext>();
  compiled->context->appendDialectRegistry(sharedDialectRegistry());
  compiled->context->loadAllAvailableDialects();

  compiled->lowered = MlirEngine::BuildAndLower(kernel, compile_target, tensors,
                                                *compiled->context);
  enforceExecutionPolicy(compiled->lowered);
  std::string denial_reason;
  if (!CanExecuteOnHost(runtime, compiled->lowered.execution_requirements,
                        &denial_reason)) {
    fail(FormatExecutionDeniedMessage(compiled->lowered.target_profile,
                                      denial_reason));
  }

  const std::vector<std::string> shared_lib_storage =
      buildSharedLibraryPaths(compile_target);
  llvm::SmallVector<llvm::StringRef, 8> shared_libs;
  shared_libs.reserve(shared_lib_storage.size());
  for (const std::string &path : shared_lib_storage) {
    shared_libs.push_back(path);
  }

  mlir::ExecutionEngineOptions options;
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  options.sharedLibPaths = shared_libs;
  options.enableObjectDump = isDiskCacheSupported(compile_target);
  std::unique_ptr<llvm::TargetMachine> target_machine =
      createTargetMachine(x86_profile);
  compiled->engine = takeEngine(mlir::ExecutionEngine::create(
      *compiled->lowered.module, options, std::move(target_machine)));
  registerExecutionRuntimeSymbols(*compiled->engine);
  registerGpuRuntimeSymbols(*compiled->engine,
                            compiled->lowered.target_profile.kind);
  persistExecutionToDiskCache(*compiled, compile_target, disk_artifacts);

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

  switch (lhs.dtype) {
    case TensorDType::kFloat32:
      return invokeWithRhsType<float>(compiled, compiled.lowered.entry_point, lhs,
                                      rhs, out);
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return invokeWithRhsType<std::uint16_t>(compiled,
                                              compiled.lowered.entry_point, lhs,
                                              rhs, out);
    case TensorDType::kInt8:
      return invokeWithRhsType<std::int8_t>(compiled,
                                            compiled.lowered.entry_point, lhs, rhs,
                                            out);
    case TensorDType::kInt32:
      return invokeWithRhsType<std::int32_t>(compiled,
                                             compiled.lowered.entry_point, lhs, rhs,
                                             out);
    case TensorDType::kFloat8E4M3FN:
      return invokeWithRhsType<std::uint8_t>(compiled,
                                             compiled.lowered.entry_point, lhs, rhs,
                                             out);
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported runtime dtype");
}

}  // namespace

void compileAndRun(const KernelIR &kernel,
                   const RequestedTargetProfile &target_profile,
                   const std::vector<RuntimeTensorView> &tensors) {
  static std::once_flag native_target_once;
  std::call_once(native_target_once, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
  const RuntimeCapabilities &runtime = cachedRuntimeCapabilities();

  std::shared_ptr<CachedExecution> compiled =
      getOrCreateExecution(kernel, target_profile, tensors, runtime);
  if (llvm::Error error = invokeCompiledKernel(*compiled, tensors)) {
    const std::string message = llvm::toString(std::move(error));
    fail("failed to invoke JIT entrypoint '" + compiled->lowered.entry_point +
         "': " + message);
  }
}

}  // namespace matcore
