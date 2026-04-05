#include "matcore/gpu_runtime_symbols.h"
#include "matcore/runtime_capabilities.h"

#include "amd_runtime_symbols.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

#include "matcore/observability.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#if __has_include("cuda.h")
#include "cuda.h"
#include <cstdlib>
#include <list>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#endif

namespace matcore {
namespace {

thread_local ObservabilityContext *g_gpu_runtime_observability = nullptr;

void traceGpuRuntimeEvent(TraceEventKind kind, const std::string &name,
                          const std::string &metadata = {}) {
  if (g_gpu_runtime_observability == nullptr) {
    return;
  }
  g_gpu_runtime_observability->traceEvent(kind, name, metadata);
}

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore GPU runtime: " + message);
}

template <typename Fn>
Fn loadSymbol(void *handle, const char *name) {
  void *symbol = dlsym(handle, name);
  if (symbol == nullptr) {
    fail("failed to resolve runtime symbol '" + std::string(name) + "'");
  }
  return reinterpret_cast<Fn>(symbol);
}

#if __has_include("cuda.h")

#define MATCORE_GPU_RUNTIME_EXPORT                                            \
  extern "C" __attribute__((visibility("default")))

struct CudaDriverApi {
  using CuInitFn = CUresult (*)(unsigned int);
  using CuGetErrorNameFn = CUresult (*)(CUresult, const char **);
  using CuDeviceGetFn = CUresult (*)(CUdevice *, int);
  using CuDevicePrimaryCtxRetainFn = CUresult (*)(CUcontext *, CUdevice);
  using CuCtxPushCurrentFn = CUresult (*)(CUcontext);
  using CuCtxPopCurrentFn = CUresult (*)(CUcontext *);
  using CuModuleLoadDataFn = CUresult (*)(CUmodule *, const void *);
  using CuModuleLoadDataExFn =
      CUresult (*)(CUmodule *, const void *, unsigned int, CUjit_option *,
                   void **);
  using CuModuleUnloadFn = CUresult (*)(CUmodule);
  using CuModuleGetFunctionFn =
      CUresult (*)(CUfunction *, CUmodule, const char *);
  using CuLaunchKernelFn =
      CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                   unsigned int, unsigned int, unsigned int, unsigned int,
                   CUstream, void **, void **);
  using CuStreamCreateFn = CUresult (*)(CUstream *, unsigned int);
  using CuStreamDestroyFn = CUresult (*)(CUstream);
  using CuStreamSynchronizeFn = CUresult (*)(CUstream);
  using CuMemAllocFn = CUresult (*)(CUdeviceptr *, size_t);
  using CuMemFreeFn = CUresult (*)(CUdeviceptr);
  using CuMemcpyFn = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
  using CuMemcpyAsyncFn =
      CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  using CuMemsetD32AsyncFn =
      CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);
  using CuMemsetD16AsyncFn =
      CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream);
  using CuMemHostRegisterFn = CUresult (*)(void *, size_t, unsigned int);

  static CudaDriverApi &instance() {
    static CudaDriverApi api = load();
    return api;
  }

  static CudaDriverApi load() {
    void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      fail("unable to dlopen libcuda.so.1");
    }

    CudaDriverApi api;
    api.cuInit = loadSymbol<CuInitFn>(handle, "cuInit");
    api.cuGetErrorName = loadSymbol<CuGetErrorNameFn>(handle, "cuGetErrorName");
    api.cuDeviceGet = loadSymbol<CuDeviceGetFn>(handle, "cuDeviceGet");
    api.cuDevicePrimaryCtxRetain =
        loadSymbol<CuDevicePrimaryCtxRetainFn>(handle, "cuDevicePrimaryCtxRetain");
    api.cuCtxPushCurrent =
        loadSymbol<CuCtxPushCurrentFn>(handle, "cuCtxPushCurrent_v2");
    api.cuCtxPopCurrent =
        loadSymbol<CuCtxPopCurrentFn>(handle, "cuCtxPopCurrent_v2");
    api.cuModuleLoadData =
        loadSymbol<CuModuleLoadDataFn>(handle, "cuModuleLoadData");
    api.cuModuleLoadDataEx =
        loadSymbol<CuModuleLoadDataExFn>(handle, "cuModuleLoadDataEx");
    api.cuModuleUnload = loadSymbol<CuModuleUnloadFn>(handle, "cuModuleUnload");
    api.cuModuleGetFunction =
        loadSymbol<CuModuleGetFunctionFn>(handle, "cuModuleGetFunction");
    api.cuLaunchKernel =
        loadSymbol<CuLaunchKernelFn>(handle, "cuLaunchKernel");
    api.cuStreamCreate =
        loadSymbol<CuStreamCreateFn>(handle, "cuStreamCreate");
    api.cuStreamDestroy =
        loadSymbol<CuStreamDestroyFn>(handle, "cuStreamDestroy_v2");
    api.cuStreamSynchronize = loadSymbol<CuStreamSynchronizeFn>(
        handle, "cuStreamSynchronize");
    api.cuMemAlloc = loadSymbol<CuMemAllocFn>(handle, "cuMemAlloc_v2");
    api.cuMemFree = loadSymbol<CuMemFreeFn>(handle, "cuMemFree_v2");
    api.cuMemcpy = loadSymbol<CuMemcpyFn>(handle, "cuMemcpy");
    api.cuMemcpyAsync =
        loadSymbol<CuMemcpyAsyncFn>(handle, "cuMemcpyAsync");
    api.cuMemsetD32Async =
        loadSymbol<CuMemsetD32AsyncFn>(handle, "cuMemsetD32Async");
    api.cuMemsetD16Async =
        loadSymbol<CuMemsetD16AsyncFn>(handle, "cuMemsetD16Async");
    api.cuMemHostRegister =
        loadSymbol<CuMemHostRegisterFn>(handle, "cuMemHostRegister_v2");
    return api;
  }

  CuInitFn cuInit = nullptr;
  CuGetErrorNameFn cuGetErrorName = nullptr;
  CuDeviceGetFn cuDeviceGet = nullptr;
  CuDevicePrimaryCtxRetainFn cuDevicePrimaryCtxRetain = nullptr;
  CuCtxPushCurrentFn cuCtxPushCurrent = nullptr;
  CuCtxPopCurrentFn cuCtxPopCurrent = nullptr;
  CuModuleLoadDataFn cuModuleLoadData = nullptr;
  CuModuleLoadDataExFn cuModuleLoadDataEx = nullptr;
  CuModuleUnloadFn cuModuleUnload = nullptr;
  CuModuleGetFunctionFn cuModuleGetFunction = nullptr;
  CuLaunchKernelFn cuLaunchKernel = nullptr;
  CuStreamCreateFn cuStreamCreate = nullptr;
  CuStreamDestroyFn cuStreamDestroy = nullptr;
  CuStreamSynchronizeFn cuStreamSynchronize = nullptr;
  CuMemAllocFn cuMemAlloc = nullptr;
  CuMemFreeFn cuMemFree = nullptr;
  CuMemcpyFn cuMemcpy = nullptr;
  CuMemcpyAsyncFn cuMemcpyAsync = nullptr;
  CuMemsetD32AsyncFn cuMemsetD32Async = nullptr;
  CuMemsetD16AsyncFn cuMemsetD16Async = nullptr;
  CuMemHostRegisterFn cuMemHostRegister = nullptr;
};

void checkCuda(CUresult result, const char *expr) {
  if (result == CUDA_SUCCESS) {
    return;
  }

  const char *name = nullptr;
  CudaDriverApi::instance().cuGetErrorName(result, &name);
  fail("'" + std::string(expr) + "' failed with '" +
       (name != nullptr ? std::string(name) : std::string("<unknown>")) + "'");
}

thread_local static int32_t defaultDevice = 0;

CUdevice getDefaultCuDevice() {
  CudaDriverApi &api = CudaDriverApi::instance();
  checkCuda(api.cuInit(0), "cuInit");
  CUdevice device = 0;
  checkCuda(api.cuDeviceGet(&device, defaultDevice), "cuDeviceGet");
  return device;
}

struct ScopedContext {
  ScopedContext() {
    CudaDriverApi &api = CudaDriverApi::instance();
    static thread_local CUcontext context = []() {
      CudaDriverApi &inner_api = CudaDriverApi::instance();
      CUcontext ctx = nullptr;
      CUdevice device = getDefaultCuDevice();
      checkCuda(inner_api.cuDevicePrimaryCtxRetain(&ctx, device),
                "cuDevicePrimaryCtxRetain");
      return ctx;
    }();
    checkCuda(api.cuCtxPushCurrent(context), "cuCtxPushCurrent");
  }

  ~ScopedContext() { CudaDriverApi::instance().cuCtxPopCurrent(nullptr); }
};

class GpuMemoryPool {
public:
  static GpuMemoryPool &instance() {
    static GpuMemoryPool pool;
    return pool;
  }

  void *acquire(uint64_t sizeBytes) {
    if (sizeBytes == 0) {
      return nullptr;
    }

    CUdeviceptr pooled_ptr = 0;
    bool needs_alloc = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = free_lists_.find(sizeBytes);
      if (it != free_lists_.end()) {
        auto &list = it->second;
        while (!list.empty()) {
          CUdeviceptr candidate = list.back();
          list.pop_back();
          if (isValidPooledEntry(candidate, sizeBytes)) {
            pooled_ptr = candidate;
            auto pooled_it = pooled_entries_.find(candidate);
            if (pooled_it != pooled_entries_.end()) {
              pooled_order_.erase(pooled_it->second.order_it);
              pooled_entries_.erase(pooled_it);
            }
            pooled_bytes_ -= sizeBytes;
            break;
          }
        }
      }
      if (pooled_ptr == 0) {
        needs_alloc = true;
      }
    }

    if (needs_alloc) {
      ScopedContext ctx;
      CUdeviceptr ptr = 0;
      checkCuda(CudaDriverApi::instance().cuMemAlloc(&ptr, sizeBytes), "cuMemAlloc");
      {
        std::lock_guard<std::mutex> lock(mutex_);
        allocation_sizes_[ptr] = sizeBytes;
      }
      return reinterpret_cast<void *>(ptr);
    }
    return reinterpret_cast<void *>(pooled_ptr);
  }

  void release(void *ptrVoid, CUstream stream) {
    if (ptrVoid == nullptr) {
      return;
    }

    (void)stream;
    const CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(ptrVoid);
    std::vector<CUdeviceptr> to_free;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto size_it = allocation_sizes_.find(ptr);
      if (size_it == allocation_sizes_.end()) {
        fail("attempted to free unknown CUDA pointer");
      }
      const uint64_t sizeBytes = size_it->second;

      if (pooled_entries_.find(ptr) != pooled_entries_.end()) {
        fail("double free detected in GPU memory pool");
      }
      free_lists_[sizeBytes].push_back(ptr);
      pooled_order_.push_back(ptr);
      PooledEntry pooled_entry{sizeBytes, std::prev(pooled_order_.end())};
      pooled_entries_.emplace(ptr, pooled_entry);
      pooled_bytes_ += sizeBytes;

      while (pooled_bytes_ > max_pooled_bytes_ && !pooled_order_.empty()) {
        const CUdeviceptr victim_ptr = pooled_order_.front();
        pooled_order_.pop_front();
        auto victim_it = pooled_entries_.find(victim_ptr);
        if (victim_it == pooled_entries_.end()) {
          continue;
        }
        const uint64_t victim_size = victim_it->second.size_bytes;
        pooled_entries_.erase(victim_it);
        pooled_bytes_ -= victim_size;
        allocation_sizes_.erase(victim_ptr);
        to_free.push_back(victim_ptr);
      }
    }

    if (!to_free.empty()) {
      ScopedContext ctx;
      for (CUdeviceptr victim_ptr : to_free) {
        checkCuda(CudaDriverApi::instance().cuMemFree(victim_ptr), "cuMemFree");
      }
    }
  }

private:
  GpuMemoryPool() : max_pooled_bytes_(readMaxPoolBytes()) {}

  static uint64_t readMaxPoolBytes() {
    constexpr uint64_t kDefaultPoolMb = 512;
    const char *raw = std::getenv("MATCORE_GPU_POOL_MB");
    if (raw == nullptr || raw[0] == '\0') {
      return kDefaultPoolMb * 1024ULL * 1024ULL;
    }

    char *end = nullptr;
    unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0') || parsed == 0ULL) {
      return kDefaultPoolMb * 1024ULL * 1024ULL;
    }
    return static_cast<uint64_t>(parsed) * 1024ULL * 1024ULL;
  }

  bool isValidPooledEntry(CUdeviceptr ptr, uint64_t expectedSize) const {
    auto pooled_it = pooled_entries_.find(ptr);
    if (pooled_it == pooled_entries_.end()) {
      return false;
    }
    return pooled_it->second.size_bytes == expectedSize;
  }

  struct PooledEntry {
    uint64_t size_bytes = 0;
    std::list<CUdeviceptr>::iterator order_it;
  };

  std::mutex mutex_;
  std::unordered_map<uint64_t, std::vector<CUdeviceptr>> free_lists_;
  std::unordered_map<CUdeviceptr, uint64_t> allocation_sizes_;
  std::unordered_map<CUdeviceptr, PooledEntry> pooled_entries_;
  std::list<CUdeviceptr> pooled_order_;
  uint64_t pooled_bytes_ = 0;
  uint64_t max_pooled_bytes_ = 0;
};

class GpuStreamPool {
public:
  static GpuStreamPool &instance() {
    static GpuStreamPool pool;
    return pool;
  }

  CUstream acquire() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      while (!free_streams_.empty()) {
        CUstream stream = free_streams_.back();
        free_streams_.pop_back();
        if (pooled_streams_.erase(stream) != 0) {
          return stream;
        }
      }
    }

    ScopedContext ctx;
    CUstream stream = nullptr;
    checkCuda(CudaDriverApi::instance().cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
              "cuStreamCreate");
    return stream;
  }

  void release(CUstream stream) {
    if (stream == nullptr) {
      return;
    }

    bool should_destroy = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto inserted = pooled_streams_.insert(stream);
      if (!inserted.second) {
        fail("double destroy detected in GPU stream pool");
      }
      if (free_streams_.size() < kMaxPooledStreams) {
        free_streams_.push_back(stream);
      } else {
        pooled_streams_.erase(stream);
        should_destroy = true;
      }
    }

    if (should_destroy) {
      ScopedContext ctx;
      checkCuda(CudaDriverApi::instance().cuStreamDestroy(stream), "cuStreamDestroy");
    }
  }

private:
  static constexpr size_t kMaxPooledStreams = 8;

  std::mutex mutex_;
  std::vector<CUstream> free_streams_;
  std::unordered_set<CUstream> pooled_streams_;
};

MATCORE_GPU_RUNTIME_EXPORT CUmodule mgpuModuleLoad(void *data,
                                                   size_t /*gpuBlobSize*/) {
  ScopedContext scoped_context;
  CUmodule module = nullptr;
  checkCuda(CudaDriverApi::instance().cuModuleLoadData(&module, data),
            "cuModuleLoadData");
  traceGpuRuntimeEvent(TraceEventKind::kGpuModuleLoad, "mgpuModuleLoad");
  return module;
}

MATCORE_GPU_RUNTIME_EXPORT CUmodule mgpuModuleLoadJIT(void *data, int optLevel) {
  ScopedContext scoped_context;
  CudaDriverApi &api = CudaDriverApi::instance();
  CUmodule module = nullptr;
  char error_log[8192] = {};
  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                            CU_JIT_OPTIMIZATION_LEVEL};
  void *values[] = {
      error_log,
      reinterpret_cast<void *>(static_cast<std::intptr_t>(sizeof(error_log))),
      reinterpret_cast<void *>(static_cast<std::intptr_t>(optLevel)),
  };
  const CUresult result =
      api.cuModuleLoadDataEx(&module, data, 3, options, values);
  if (result != CUDA_SUCCESS) {
    if (error_log[0] != '\0') {
      std::fprintf(stderr, "CUDA JIT compilation failed with: '%s'\n", error_log);
    }
    checkCuda(result, "cuModuleLoadDataEx");
  }
  traceGpuRuntimeEvent(TraceEventKind::kGpuModuleLoad, "mgpuModuleLoadJIT");
  return module;
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuModuleUnload(CUmodule module) {
  ScopedContext scoped_context;
  checkCuda(CudaDriverApi::instance().cuModuleUnload(module), "cuModuleUnload");
}

MATCORE_GPU_RUNTIME_EXPORT CUfunction mgpuModuleGetFunction(CUmodule module,
                                                            const char *name) {
  ScopedContext scoped_context;
  CUfunction function = nullptr;
  checkCuda(CudaDriverApi::instance().cuModuleGetFunction(&function, module, name),
            "cuModuleGetFunction");
  return function;
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuLaunchKernel(
    CUfunction function, std::intptr_t gridX, std::intptr_t gridY,
    std::intptr_t gridZ, std::intptr_t blockX, std::intptr_t blockY,
    std::intptr_t blockZ, int32_t smem, CUstream stream, void **params,
    void **extra, size_t /*paramsCount*/) {
  ScopedContext scoped_context;
  constexpr const char *kKernelTraceName = "mgpuLaunchKernel";
  traceGpuRuntimeEvent(TraceEventKind::kGpuKernelLaunch, kKernelTraceName);
  checkCuda(CudaDriverApi::instance().cuLaunchKernel(
                 function, static_cast<unsigned int>(gridX),
                 static_cast<unsigned int>(gridY),
                 static_cast<unsigned int>(gridZ),
                static_cast<unsigned int>(blockX),
                static_cast<unsigned int>(blockY),
                static_cast<unsigned int>(blockZ),
                static_cast<unsigned int>(smem), stream, params, extra),
            "cuLaunchKernel");
}

MATCORE_GPU_RUNTIME_EXPORT CUstream mgpuStreamCreate() {
  return GpuStreamPool::instance().acquire();
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuStreamDestroy(CUstream stream) {
  GpuStreamPool::instance().release(stream);
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuStreamSynchronize(CUstream stream) {
  ScopedContext scoped_context;
  constexpr const char *kKernelTraceName = "mgpuLaunchKernel";
  checkCuda(CudaDriverApi::instance().cuStreamSynchronize(stream),
            "cuStreamSynchronize");
  traceGpuRuntimeEvent(TraceEventKind::kGpuKernelComplete, kKernelTraceName);
}

MATCORE_GPU_RUNTIME_EXPORT void *mgpuMemAlloc(uint64_t sizeBytes,
                                              CUstream /*stream*/,
                                              bool isHostShared) {
  if (isHostShared) {
    fail("managed CUDA allocations are not implemented in MatCore's runtime");
  }
  if (sizeBytes == 0) {
    return nullptr;
  }
  return GpuMemoryPool::instance().acquire(sizeBytes);
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemFree(void *ptr, CUstream stream) {
  GpuMemoryPool::instance().release(ptr, stream);
}

// NOTE: Uses synchronous cuMemcpy (not cuMemcpyAsync). The stream parameter
// is accepted for API compatibility with MLIR's gpu-to-llvm lowering but is
// currently unused. Correctness is maintained because the staging pass inserts
// explicit gpu.wait synchronization barriers before/after all memcpy sequences.
MATCORE_GPU_RUNTIME_EXPORT void mgpuMemcpy(void *dst, void *src,
                                           size_t sizeBytes, CUstream stream) {
  ScopedContext scoped_context;
  checkCuda(CudaDriverApi::instance().cuMemcpy(
                reinterpret_cast<CUdeviceptr>(dst),
                reinterpret_cast<CUdeviceptr>(src), sizeBytes),
            "cuMemcpy");
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemset32(void *dst, unsigned int value,
                                             size_t count, CUstream stream) {
  ScopedContext scoped_context;
  checkCuda(CudaDriverApi::instance().cuMemsetD32Async(
                reinterpret_cast<CUdeviceptr>(dst), value, count, stream),
            "cuMemsetD32Async");
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemset16(void *dst, unsigned short value,
                                             size_t count, CUstream stream) {
  ScopedContext scoped_context;
  checkCuda(CudaDriverApi::instance().cuMemsetD16Async(
                reinterpret_cast<CUdeviceptr>(dst), value, count, stream),
            "cuMemsetD16Async");
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemHostRegister(void *ptr,
                                                    uint64_t sizeBytes) {
  ScopedContext scoped_context;
  checkCuda(CudaDriverApi::instance().cuMemHostRegister(
                ptr, static_cast<size_t>(sizeBytes), 0),
            "cuMemHostRegister");
}

void registerCudaRuntimeSymbols(mlir::ExecutionEngine &engine) {
  engine.registerSymbols([](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    auto add_symbol = [&](llvm::StringRef name, auto *fn) {
      symbols[interner(name)] = llvm::orc::ExecutorSymbolDef(
          llvm::orc::ExecutorAddr::fromPtr(fn), llvm::JITSymbolFlags::Exported);
    };

    // Register under standard names (used by gpu-to-llvm pass lowering).
    add_symbol("mgpuModuleLoad", &mgpuModuleLoad);
    add_symbol("mgpuModuleLoadJIT", &mgpuModuleLoadJIT);
    add_symbol("mgpuModuleUnload", &mgpuModuleUnload);
    add_symbol("mgpuModuleGetFunction", &mgpuModuleGetFunction);
    add_symbol("mgpuLaunchKernel", &mgpuLaunchKernel);
    add_symbol("mgpuStreamCreate", &mgpuStreamCreate);
    add_symbol("mgpuStreamDestroy", &mgpuStreamDestroy);
    add_symbol("mgpuStreamSynchronize", &mgpuStreamSynchronize);
    add_symbol("mgpuMemAlloc", &mgpuMemAlloc);
    add_symbol("mgpuMemFree", &mgpuMemFree);
    add_symbol("mgpuMemcpy", &mgpuMemcpy);
    add_symbol("mgpuMemset32", &mgpuMemset32);
    add_symbol("mgpuMemset16", &mgpuMemset16);
    add_symbol("mgpuMemHostRegister", &mgpuMemHostRegister);

    // Register under _mlir_ prefixed names (used by ExecutionEngine's
    // GPUDialectLLVMIRTranslationInterface for gpu.launch_func + gpu.binary).
    add_symbol("_mlir_ciface_mgpuModuleLoad", &mgpuModuleLoad);
    add_symbol("_mlir_ciface_mgpuModuleLoadJIT", &mgpuModuleLoadJIT);
    add_symbol("_mlir_ciface_mgpuModuleUnload", &mgpuModuleUnload);
    add_symbol("_mlir_ciface_mgpuModuleGetFunction", &mgpuModuleGetFunction);
    add_symbol("_mlir_ciface_mgpuLaunchKernel", &mgpuLaunchKernel);
    add_symbol("_mlir_ciface_mgpuStreamCreate", &mgpuStreamCreate);
    add_symbol("_mlir_ciface_mgpuStreamDestroy", &mgpuStreamDestroy);
    add_symbol("_mlir_ciface_mgpuStreamSynchronize", &mgpuStreamSynchronize);
    add_symbol("_mlir_ciface_mgpuMemAlloc", &mgpuMemAlloc);
    add_symbol("_mlir_ciface_mgpuMemFree", &mgpuMemFree);
    add_symbol("_mlir_ciface_mgpuMemcpy", &mgpuMemcpy);

    // Also register with bare _mlir_ prefix (JIT mangling variant).
    add_symbol("_mlir_mgpuModuleLoad", &mgpuModuleLoad);
    add_symbol("_mlir_mgpuModuleLoadJIT", &mgpuModuleLoadJIT);
    add_symbol("_mlir_mgpuModuleUnload", &mgpuModuleUnload);
    add_symbol("_mlir_mgpuModuleGetFunction", &mgpuModuleGetFunction);
    add_symbol("_mlir_mgpuLaunchKernel", &mgpuLaunchKernel);
    add_symbol("_mlir_mgpuStreamCreate", &mgpuStreamCreate);
    add_symbol("_mlir_mgpuStreamDestroy", &mgpuStreamDestroy);
    add_symbol("_mlir_mgpuStreamSynchronize", &mgpuStreamSynchronize);
    add_symbol("_mlir_mgpuMemAlloc", &mgpuMemAlloc);
    add_symbol("_mlir_mgpuMemFree", &mgpuMemFree);
    add_symbol("_mlir_mgpuMemcpy", &mgpuMemcpy);

    return symbols;
  });
}

#undef MATCORE_GPU_RUNTIME_EXPORT

#endif

}  // namespace

void setGpuRuntimeObservabilityContext(ObservabilityContext *obs) {
  g_gpu_runtime_observability = obs;
}

void registerGpuRuntimeSymbols(mlir::ExecutionEngine &engine, TargetKind target) {
  // SAFETY: Only ONE GPU backend may register symbols per process lifetime.
  // NVIDIA and AMD use the same mgpu* symbol names — registering both
  // overwrites CUDA ptrs with HIP ptrs (or vice-versa), causing GPU MMU
  // faults and potential kernel panics on dual-GPU systems (e.g. RTX 4060
  // + AMD 890M iGPU sharing the same PCIe/memory bus).
  // Uses compare_exchange to prevent TOCTOU races between threads.

  switch (normalizeTarget(target)) {
    case TargetKind::kNvidiaDGPU: {
      std::string denial_reason;
      if (!acquireGpuBackendClaim(TargetKind::kNvidiaDGPU, &denial_reason)) {
        fail(denial_reason);
      }
#if __has_include("cuda.h")
      registerCudaRuntimeSymbols(engine);
#else
      fail("nvidia-dgpu requested but CUDA headers were not available at build time");
#endif
      return;
    }
    case TargetKind::kAmdIGPU: {
      std::string denial_reason;
      if (!acquireGpuBackendClaim(TargetKind::kAmdIGPU, &denial_reason)) {
        fail(denial_reason);
      }
      if (!registerAmdRuntimeSymbols(engine)) {
        // Registration failed (no HIP runtime) — release the claim
        releaseGpuBackendClaim(TargetKind::kAmdIGPU);
      }
      return;
    }
    case TargetKind::kX86Auto:
    case TargetKind::kX86AVX2:
    case TargetKind::kX86AVX512:
    case TargetKind::kAmdNPU:
    case TargetKind::kARM:
    case TargetKind::kTPU:
    case TargetKind::kNVPTX:
    case TargetKind::kAMDGCN:
    case TargetKind::kNPU:
      return;
  }
}

}  // namespace matcore
