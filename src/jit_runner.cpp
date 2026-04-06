#include "matcore/jit_runner.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "cache_manager.h"
#include "executor.h"
#include "matcore/device_buffer.h"
#include "matcore/gpu_runtime_symbols.h"
#include "matcore/mlir_engine.h"
#include "matcore/observability.h"
#include "matcore/plan.h"
#include "matcore/runtime_capabilities.h"
#include "jit_runner_internal.h"

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

bool isX86Target(const RequestedTargetProfile &target_profile) {
  const TargetKind normalized = normalizeTarget(target_profile.kind);
  return normalized == TargetKind::kX86Auto ||
         normalized == TargetKind::kX86AVX2 ||
         normalized == TargetKind::kX86AVX512;
}

bool isGpuTarget(TargetKind target) {
  switch (target) {
    case TargetKind::kNvidiaDGPU:
    case TargetKind::kAmdIGPU:
    case TargetKind::kAMDGCN:
    case TargetKind::kNVPTX:
      return true;
    default:
      return false;
  }
}

struct X86TargetProfile {
  std::string cpu;
  std::string features;
  std::string cache_tag;
};

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

std::string entryPointName(const KernelIR &kernel) {
  return kernel.kernel_name.empty() ? "matcore_kernel" : kernel.kernel_name;
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

}  // anonymous namespace

RuntimeCapabilities &cachedRuntimeCapabilities() {
  static RuntimeCapabilities runtime = DetectRuntimeCapabilities();
  return runtime;
}

RequestedTargetProfile resolveCompilationTargetProfile(
    const RequestedTargetProfile &target_profile,
    RuntimeCapabilities &runtime) {
  // Ensure GPU capabilities are probed before reading compute_major.
  // Without this, the lazy probe hasn't fired yet and we fall back to sm_80.
  if (normalizeTarget(target_profile.kind) == TargetKind::kNvidiaDGPU) {
    probeNvidiaIfNeeded(runtime);
  }
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

namespace {

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
  removeArtifactIfExists(artifacts.object_path);
  try {
    auto metadata_path = artifacts.artifact_dir / "metadata.json";
    llvm::json::Object metadata_obj{
        {"matcore_cache_version", std::string(kDiskCacheVersion)},
        {"target", CanonicalTargetString(compile_target)},
        {"entry_point", compiled.lowered.entry_point},
        {"route_description", compiled.lowered.route_description},
    };
    std::ofstream meta_out(metadata_path);
    if (meta_out.is_open()) {
      meta_out << llvm::formatv("{0:2}\n",
                                llvm::json::Value(std::move(metadata_obj)))
                      .str();
    }
  } catch (...) {
  }
}

std::shared_ptr<CachedExecution>
getOrCreateExecution(const KernelIR &kernel,
                     const RequestedTargetProfile &target_profile,
                     const std::vector<RuntimeTensorView> &tensors,
                     RuntimeCapabilities &runtime,
                     ObservabilityContext *obs) {
  static std::mutex cache_mutex;
  static auto *cache =
      new std::unordered_map<std::string, std::shared_ptr<CachedExecution>>();

  const RequestedTargetProfile compile_target =
      resolveCompilationTargetProfile(target_profile, runtime);
  const ExecutionRequirements requested_requirements =
      BuildExecutionRequirements(target_profile);
  auto validateExecutionEligibility =
      [&](const RequestedTargetProfile &profile,
          const ExecutionRequirements &requirements) {
        if (isGpuTarget(profile.kind)) {
          const GpuPreflightResult preflight = gpuPreflightCheck(profile.kind);
          if (obs) {
            obs->traceEvent(TraceEventKind::kGpuPreflight, "gpu_preflight",
                            preflight.diagnostic);
          }
          if (preflight.status == GpuPreflightStatus::kFail) {
            fail(FormatExecutionDeniedMessage(
                profile, "GPU preflight failed: " + preflight.diagnostic));
          }
        }

        std::string denial_reason;
        if (!CanExecuteOnHost(runtime, requirements, &denial_reason)) {
          fail(FormatExecutionDeniedMessage(profile, denial_reason));
        }
      };
  const std::optional<X86TargetProfile> x86_profile =
      resolveX86TargetProfile(compile_target, runtime);
  const std::optional<std::string_view> x86_cache_tag =
      x86_profile.has_value()
          ? std::optional<std::string_view>(x86_profile->cache_tag)
          : std::nullopt;
  const std::string cache_key =
      buildExecutionCacheKey(kernel, compile_target, tensors, x86_cache_tag);
  const DiskCacheArtifacts disk_artifacts = buildDiskCacheArtifacts(cache_key);
  const bool skip_cache_lookup = obs && obs->forceRecompile();
  if (!skip_cache_lookup) {
    std::shared_ptr<CachedExecution> memory_cached;
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      auto it = cache->find(cache_key);
      if (it != cache->end()) {
        if (obs) {
          obs->recordCacheHit(cache_key);
          obs->traceEvent(TraceEventKind::kCacheHit, cache_key);
        }
        memory_cached = it->second;
      }
    }
    if (memory_cached) {
      validateExecutionEligibility(compile_target, requested_requirements);
      return memory_cached;
    }
  }
  if (!skip_cache_lookup) {
    if (std::shared_ptr<CachedExecution> disk_cached = tryLoadDiskCachedExecution(
            kernel, compile_target, requested_requirements, disk_artifacts)) {
      std::shared_ptr<CachedExecution> selected;
      {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto [it, inserted] = cache->emplace(cache_key, disk_cached);
        if (obs) {
          obs->recordCacheHit(cache_key);
          obs->traceEvent(TraceEventKind::kCacheHit, cache_key);
        }
        selected = inserted ? disk_cached : it->second;
      }
      validateExecutionEligibility(compile_target, requested_requirements);
      return selected;
    }
  }
  if (obs) {
    obs->traceEvent(TraceEventKind::kCacheMiss, cache_key);
  }

  auto compiled = std::make_shared<CachedExecution>();
  compiled->context = std::make_unique<mlir::MLIRContext>();
  compiled->context->appendDialectRegistry(sharedDialectRegistry());
  compiled->context->loadAllAvailableDialects();

  compiled->lowered = MlirEngine::BuildAndLower(kernel, compile_target, tensors,
                                                *compiled->context, obs);
  enforceExecutionPolicy(compiled->lowered);
  validateExecutionEligibility(compiled->lowered.target_profile,
                               compiled->lowered.execution_requirements);

  const std::vector<std::string> shared_lib_storage =
      buildSharedLibraryPaths(compile_target);
  llvm::SmallVector<llvm::StringRef, 8> shared_libs;
  shared_libs.reserve(shared_lib_storage.size());
  for (const std::string &path : shared_lib_storage) {
    shared_libs.push_back(path);
  }

  // Dump final MLIR IR for debug (temporary).
  if (std::getenv("MATCORE_DUMP_FINAL_IR")) {
    llvm::errs() << "=== FINAL MLIR IR BEFORE JIT ===\n";
    compiled->lowered.module->print(llvm::errs());
    llvm::errs() << "=== END FINAL MLIR IR ===\n";
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
  if (skip_cache_lookup) {
    (*cache)[cache_key] = compiled;
    return compiled;
  }
  auto [it, inserted] = cache->emplace(cache_key, compiled);
  if (!inserted) {
    return it->second;
  }
  return compiled;
}

}  // namespace

CachedExecution::~CachedExecution() {
  if (shared_library_handle != nullptr) {
    dlclose(shared_library_handle);
  }
}

void compileAndRun(const KernelIR &kernel,
                   const RequestedTargetProfile &target_profile,
                   const std::vector<RuntimeTensorView> &tensors,
                   ObservabilityContext *obs) {
  static std::once_flag native_target_once;
  std::call_once(native_target_once, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
  RuntimeCapabilities &runtime = cachedRuntimeCapabilities();
  auto run = [&]() {
    const bool trace_timing = std::getenv("MATCORE_TRACE_TIMING") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    std::shared_ptr<CachedExecution> compiled;
    if (obs) {
      auto compile_scope = obs->scopedTrace(TraceEventKind::kCompileStart,
                                            TraceEventKind::kCompileEnd,
                                            "compileAndRun");
      compiled = getOrCreateExecution(kernel, target_profile, tensors, runtime, obs);
    } else {
      compiled = getOrCreateExecution(kernel, target_profile, tensors, runtime, obs);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    setGpuRuntimeObservabilityContext(obs);
    std::unique_ptr<ObservabilityContext::TraceScope> execute_scope;
    if (obs) {
      execute_scope = std::make_unique<ObservabilityContext::TraceScope>(
          *obs, TraceEventKind::kPassStageStart, TraceEventKind::kPassStageEnd,
          "execute");
    }
    if (llvm::Error error = invokeCompiledKernel(*compiled, tensors)) {
      setGpuRuntimeObservabilityContext(nullptr);
      const std::string message = llvm::toString(std::move(error));
      fail("failed to invoke JIT entrypoint '" + compiled->lowered.entry_point +
           "': " + message);
    }
    setGpuRuntimeObservabilityContext(nullptr);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (trace_timing) {
      auto cache_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      auto exec_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
      std::fprintf(stderr, "[MATCORE_TIMING] cache_lookup=%lld us  execute=%lld us  total=%lld us\n",
                   (long long)cache_us, (long long)exec_us, (long long)total_us);
    }
  };
  if (obs) {
    try {
      run();
      obs->finalize();
      return;
    } catch (...) {
      setGpuRuntimeObservabilityContext(nullptr);
      obs->finalize();
      throw;
    }
  }
  run();
}

// -------------------------------------------------------------------------
// V2: MatcorePlan — pre-compiled execution plan for near-zero-overhead dispatch
// -------------------------------------------------------------------------

static std::atomic<uint64_t> g_plan_generation_counter{1};

uint64_t MatcorePlan::nextGenerationId() {
  return g_plan_generation_counter.fetch_add(1, std::memory_order_relaxed);
}

MatcorePlan::~MatcorePlan() {
  // V2: Destroy CUDA graph resources before releasing execution bundle
  if (graph_exec_) {
    matcore_graph_exec_destroy(graph_exec_);
    graph_exec_ = nullptr;
  }
  if (graph_stream_) {
    matcore_graph_stream_destroy(graph_stream_);
    graph_stream_ = nullptr;
  }
}
MatcorePlan::MatcorePlan(MatcorePlan &&other) noexcept
    : generation_id_(other.generation_id_),
      execution_(std::move(other.execution_)),
      frozen_meta_(std::move(other.frozen_meta_)),
      has_device_tensors_(other.has_device_tensors_),
      cache_key_(std::move(other.cache_key_)),
      graph_mode_(other.graph_mode_),
      graph_stream_(other.graph_stream_),
      graph_exec_(other.graph_exec_),
      graph_captured_(other.graph_captured_),
      captured_ptrs_(std::move(other.captured_ptrs_)) {
  // Null the source to prevent double-destroy of CUDA resources
  other.graph_stream_ = nullptr;
  other.graph_exec_ = nullptr;
  other.graph_captured_ = false;
}

MatcorePlan &MatcorePlan::operator=(MatcorePlan &&other) noexcept {
  if (this != &other) {
    // Destroy our own graph resources first
    if (graph_exec_) matcore_graph_exec_destroy(graph_exec_);
    if (graph_stream_) matcore_graph_stream_destroy(graph_stream_);

    generation_id_ = other.generation_id_;
    execution_ = std::move(other.execution_);
    frozen_meta_ = std::move(other.frozen_meta_);
    has_device_tensors_ = other.has_device_tensors_;
    cache_key_ = std::move(other.cache_key_);
    graph_mode_ = other.graph_mode_;
    graph_stream_ = other.graph_stream_;
    graph_exec_ = other.graph_exec_;
    graph_captured_ = other.graph_captured_;
    captured_ptrs_ = std::move(other.captured_ptrs_);

    other.graph_stream_ = nullptr;
    other.graph_exec_ = nullptr;
    other.graph_captured_ = false;
  }
  return *this;
}

std::unique_ptr<MatcorePlan>
MatcorePlan::create(const KernelIR &kernel,
                    const std::vector<RuntimeTensorView> &template_tensors,
                    const std::string &target_str,
                    ObservabilityContext *obs,
                    bool graph_mode) {
  static std::once_flag native_target_once;
  std::call_once(native_target_once, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });

  // Parse target string to RequestedTargetProfile
  const RequestedTargetProfile target_profile = ParseRequestedTargetProfile(target_str);
  RuntimeCapabilities &runtime = cachedRuntimeCapabilities();

  // Full JIT compilation — this is the expensive part (done once)
  std::shared_ptr<CachedExecution> compiled =
      getOrCreateExecution(kernel, target_profile, template_tensors, runtime, obs);

  // Build the plan
  auto plan = std::unique_ptr<MatcorePlan>(new MatcorePlan());
  plan->generation_id_ = nextGenerationId();
  plan->execution_ = std::move(compiled);

  // Freeze tensor metadata
  plan->frozen_meta_.reserve(template_tensors.size());
  for (size_t i = 0; i < template_tensors.size(); ++i) {
    const auto &t = template_tensors[i];
    FrozenTensorMeta meta;
    meta.symbol = t.symbol;
    meta.dtype = t.dtype;
    meta.rank = static_cast<int64_t>(t.shape.size());
    meta.shape = t.shape;
    meta.strides = t.strides;
    meta.is_device_resident = t.is_device_resident;
    // Output tensor is the last one (index 2 for matmul)
    meta.is_output = (i == plan->execution_->lowered.out_tensor_index);
    plan->frozen_meta_.push_back(std::move(meta));
    if (t.is_device_resident) {
      plan->has_device_tensors_ = true;
    }
  }

  // V2 Pillar 2: Create dedicated graph stream if graph_mode requested
  plan->graph_mode_ = graph_mode;
  if (graph_mode) {
    plan->graph_stream_ = matcore_graph_stream_create();
  }

  return plan;
}

bool MatcorePlan::validateTensors(const std::vector<RuntimeTensorView> &tensors,
                                  std::string *error_msg) const {
  if (tensors.size() != frozen_meta_.size()) {
    if (error_msg) {
      *error_msg = "expected " + std::to_string(frozen_meta_.size()) +
                   " tensors, got " + std::to_string(tensors.size());
    }
    return false;
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &t = tensors[i];
    const auto &m = frozen_meta_[i];

    if (t.dtype != m.dtype) {
      if (error_msg) {
        *error_msg = "tensor[" + std::to_string(i) + "] dtype mismatch";
      }
      return false;
    }
    if (static_cast<int64_t>(t.shape.size()) != m.rank) {
      if (error_msg) {
        *error_msg = "tensor[" + std::to_string(i) + "] rank mismatch";
      }
      return false;
    }
    if (t.shape != m.shape) {
      if (error_msg) {
        *error_msg = "tensor[" + std::to_string(i) + "] shape mismatch — "
                     "plans are shape-locked, create a new plan for different shapes";
      }
      return false;
    }
    if (t.strides != m.strides) {
      if (error_msg) {
        *error_msg = "tensor[" + std::to_string(i) + "] strides mismatch";
      }
      return false;
    }
    if (t.is_device_resident != m.is_device_resident) {
      if (error_msg) {
        *error_msg = "tensor[" + std::to_string(i) + "] residency mismatch — "
                     "cannot mix host/device tensors with a device-resident plan";
      }
      return false;
    }
  }

  return true;
}

void MatcorePlan::zeroOutputs(const std::vector<RuntimeTensorView> &tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (frozen_meta_[i].is_output && tensors[i].is_device_resident &&
        tensors[i].data != nullptr) {
      uint64_t size_bytes = 1;
      for (auto dim : frozen_meta_[i].shape) size_bytes *= dim;
      switch (frozen_meta_[i].dtype) {
        case TensorDType::kFloat16:
        case TensorDType::kBFloat16: size_bytes *= 2; break;
        case TensorDType::kFloat32:
        case TensorDType::kInt32: size_bytes *= 4; break;
        case TensorDType::kInt8:
        case TensorDType::kFloat8E4M3FN: break;
      }
      matcore_device_zero_raw(tensors[i].data, size_bytes);
    }
  }
}

void MatcorePlan::zeroOutputsOnStream(
    const std::vector<RuntimeTensorView> &tensors, void *stream) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (frozen_meta_[i].is_output && tensors[i].is_device_resident &&
        tensors[i].data != nullptr) {
      uint64_t size_bytes = 1;
      for (auto dim : frozen_meta_[i].shape) size_bytes *= dim;
      switch (frozen_meta_[i].dtype) {
        case TensorDType::kFloat16:
        case TensorDType::kBFloat16: size_bytes *= 2; break;
        case TensorDType::kFloat32:
        case TensorDType::kInt32: size_bytes *= 4; break;
        case TensorDType::kInt8:
        case TensorDType::kFloat8E4M3FN: break;
      }
      matcore_device_zero_raw_on_stream(tensors[i].data, size_bytes, stream);
    }
  }
}

void MatcorePlan::execute(const std::vector<RuntimeTensorView> &tensors) {
  // Step 1: Validate tensors match frozen plan
  std::string error_msg;
  if (!validateTensors(tensors, &error_msg)) {
    fail("execute_plan: " + error_msg);
  }

  // ---- Graph replay fast path ----
  if (graph_mode_ && graph_captured_) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (tensors[i].data != captured_ptrs_[i]) {
        fail("execute_plan: tensor[" + std::to_string(i) +
             "] data pointer changed since graph capture. "
             "CUDA graphs bake addresses — use the same DeviceTensors, "
             "or create a new plan to re-capture.");
      }
    }
    matcore_graph_launch(graph_exec_, graph_stream_);
    matcore_stream_synchronize(graph_stream_);
    return;
  }

  // ---- First call with graph_mode: warm-up then capture ----
  if (graph_mode_ && !graph_captured_) {
    // Phase 1: Warm-up invocation — caches module/function, skips unload.
    matcore_set_graph_warmup(true);
    zeroOutputs(tensors);
    setGpuRuntimeObservabilityContext(nullptr);
    if (llvm::Error error = invokeCompiledKernel(*execution_, tensors)) {
      matcore_set_graph_warmup(false);
      const std::string message = llvm::toString(std::move(error));
      fail("execute_plan: warm-up failed: " + message);
    }
    matcore_set_graph_warmup(false);

    // Phase 2: Capture. Stream override routes all GPU ops to capture stream.
    matcore_set_capture_stream_override(graph_stream_);
    matcore_graph_begin_capture(graph_stream_);

    zeroOutputsOnStream(tensors, graph_stream_);

    if (llvm::Error error = invokeCompiledKernel(*execution_, tensors)) {
      matcore_set_capture_stream_override(nullptr);
      try {
        void *exec = matcore_graph_end_capture(graph_stream_);
        if (exec) matcore_graph_exec_destroy(exec);
      } catch (...) {}
      const std::string message = llvm::toString(std::move(error));
      fail("execute_plan: capture failed: " + message);
    }

    matcore_set_capture_stream_override(nullptr);
    graph_exec_ = matcore_graph_end_capture(graph_stream_);
    graph_captured_ = true;
    captured_ptrs_.clear();
    captured_ptrs_.reserve(tensors.size());
    for (const auto &t : tensors) {
      captured_ptrs_.push_back(t.data);
    }
    matcore_stream_synchronize(graph_stream_);
    return;
  }

  // ---- Normal (non-graph) execution ----
  zeroOutputs(tensors);
  setGpuRuntimeObservabilityContext(nullptr);
  if (llvm::Error error = invokeCompiledKernel(*execution_, tensors)) {
    const std::string message = llvm::toString(std::move(error));
    fail("execute_plan: failed to invoke kernel '" +
         execution_->lowered.entry_point + "': " + message);
  }
}

}  // namespace matcore
