#include "matcore/gpu_runtime_symbols.h"
#include "matcore/device_buffer.h"
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

// V2 Pillar 2: Thread-local capture stream override.
// When set, mgpuStreamCreate() returns this stream instead of creating one,
// and mgpuStreamDestroy() is a no-op. This ensures the kernel executes on
// the capture stream so cuStreamBeginCapture captures all GPU ops.
thread_local CUstream g_capture_stream_override = nullptr;

// V2: Module/function cache during graph capture.
// MLIR-generated code loads the module on every invokePacked. During capture,
// we cache the module+function from the first (pre-capture) call and reuse them.
struct CaptureModuleCache {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  bool active = false;  // true during capture — reuse cached values
  bool warmup = false;  // true during warm-up — cache values, skip unload
};
thread_local CaptureModuleCache g_capture_module_cache;

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
  using CuMemcpyHtoDFn = CUresult (*)(CUdeviceptr, const void *, size_t);
  using CuMemcpyDtoHFn = CUresult (*)(void *, CUdeviceptr, size_t);
  using CuCtxSynchronizeFn = CUresult (*)();
  using CuMemcpyAsyncFn =
      CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  using CuMemsetD32AsyncFn =
      CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);
  using CuMemsetD16AsyncFn =
      CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream);
  using CuMemHostRegisterFn = CUresult (*)(void *, size_t, unsigned int);
  // V2: Event + stream capture symbols for stream-correct execution
  using CuEventCreateFn = CUresult (*)(CUevent *, unsigned int);
  using CuEventRecordFn = CUresult (*)(CUevent, CUstream);
  using CuEventQueryFn = CUresult (*)(CUevent);
  using CuEventDestroyFn = CUresult (*)(CUevent);
  using CuStreamIsCapturingFn = CUresult (*)(CUstream, CUstreamCaptureStatus *);
  // V2 Pillar 2: CUDA Graph symbols
  using CuStreamBeginCaptureFn = CUresult (*)(CUstream, CUstreamCaptureMode);
  using CuStreamEndCaptureFn = CUresult (*)(CUstream, CUgraph *);
  using CuGraphInstantiateFn = CUresult (*)(CUgraphExec *, CUgraph,
                                            unsigned long long);
  using CuGraphLaunchFn = CUresult (*)(CUgraphExec, CUstream);
  using CuGraphExecDestroyFn = CUresult (*)(CUgraphExec);
  using CuGraphDestroyFn = CUresult (*)(CUgraph);

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
    api.cuMemcpyHtoD = loadSymbol<CuMemcpyHtoDFn>(handle, "cuMemcpyHtoD_v2");
    api.cuMemcpyDtoH = loadSymbol<CuMemcpyDtoHFn>(handle, "cuMemcpyDtoH_v2");
    api.cuCtxSynchronize =
        loadSymbol<CuCtxSynchronizeFn>(handle, "cuCtxSynchronize");
    api.cuMemcpyAsync =
        loadSymbol<CuMemcpyAsyncFn>(handle, "cuMemcpyAsync");
    api.cuMemsetD32Async =
        loadSymbol<CuMemsetD32AsyncFn>(handle, "cuMemsetD32Async");
    api.cuMemsetD16Async =
        loadSymbol<CuMemsetD16AsyncFn>(handle, "cuMemsetD16Async");
    api.cuMemHostRegister =
        loadSymbol<CuMemHostRegisterFn>(handle, "cuMemHostRegister_v2");
    // V2: Event + stream capture symbols
    api.cuEventCreate =
        loadSymbol<CuEventCreateFn>(handle, "cuEventCreate");
    api.cuEventRecord =
        loadSymbol<CuEventRecordFn>(handle, "cuEventRecord");
    api.cuEventQuery =
        loadSymbol<CuEventQueryFn>(handle, "cuEventQuery");
    api.cuEventDestroy =
        loadSymbol<CuEventDestroyFn>(handle, "cuEventDestroy_v2");
    api.cuStreamIsCapturing =
        loadSymbol<CuStreamIsCapturingFn>(handle, "cuStreamIsCapturing");
    // V2 Pillar 2: CUDA Graph symbols
    api.cuStreamBeginCapture =
        loadSymbol<CuStreamBeginCaptureFn>(handle, "cuStreamBeginCapture");
    api.cuStreamEndCapture =
        loadSymbol<CuStreamEndCaptureFn>(handle, "cuStreamEndCapture");
    api.cuGraphInstantiate =
        loadSymbol<CuGraphInstantiateFn>(handle, "cuGraphInstantiateWithFlags");
    api.cuGraphLaunch =
        loadSymbol<CuGraphLaunchFn>(handle, "cuGraphLaunch");
    api.cuGraphExecDestroy =
        loadSymbol<CuGraphExecDestroyFn>(handle, "cuGraphExecDestroy");
    api.cuGraphDestroy =
        loadSymbol<CuGraphDestroyFn>(handle, "cuGraphDestroy");
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
  CuMemcpyHtoDFn cuMemcpyHtoD = nullptr;
  CuMemcpyDtoHFn cuMemcpyDtoH = nullptr;
  CuCtxSynchronizeFn cuCtxSynchronize = nullptr;
  CuMemcpyAsyncFn cuMemcpyAsync = nullptr;
  CuMemsetD32AsyncFn cuMemsetD32Async = nullptr;
  CuMemsetD16AsyncFn cuMemsetD16Async = nullptr;
  CuMemHostRegisterFn cuMemHostRegister = nullptr;
  // V2: Event + stream capture symbols
  CuEventCreateFn cuEventCreate = nullptr;
  CuEventRecordFn cuEventRecord = nullptr;
  CuEventQueryFn cuEventQuery = nullptr;
  CuEventDestroyFn cuEventDestroy = nullptr;
  CuStreamIsCapturingFn cuStreamIsCapturing = nullptr;
  // V2 Pillar 2: CUDA Graph symbols
  CuStreamBeginCaptureFn cuStreamBeginCapture = nullptr;
  CuStreamEndCaptureFn cuStreamEndCapture = nullptr;
  CuGraphInstantiateFn cuGraphInstantiate = nullptr;
  CuGraphLaunchFn cuGraphLaunch = nullptr;
  CuGraphExecDestroyFn cuGraphExecDestroy = nullptr;
  CuGraphDestroyFn cuGraphDestroy = nullptr;
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

// V2 SAFETY: Global (not thread-local) device context for plan-bound execution.
// Plans store this context at creation time and push it on every execute_plan()
// call, regardless of which thread executes them.
CUcontext getDeviceContext() {
  static CUcontext g_context = []() {
    CudaDriverApi &api = CudaDriverApi::instance();
    checkCuda(api.cuInit(0), "cuInit");
    CUdevice device = 0;
    checkCuda(api.cuDeviceGet(&device, defaultDevice), "cuDeviceGet");
    CUcontext ctx = nullptr;
    checkCuda(api.cuDevicePrimaryCtxRetain(&ctx, device),
              "cuDevicePrimaryCtxRetain");
    return ctx;
  }();
  return g_context;
}

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
    CUevent event_to_destroy = nullptr;
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
            auto pooled_it = pooled_entries_.find(candidate);
            if (pooled_it != pooled_entries_.end()) {
              // V2 SAFETY: Check fence event before reuse — only reuse if
              // GPU work on this buffer has completed.
              CUevent fence = pooled_it->second.fence_event;
              if (fence != nullptr) {
                CUresult status = CudaDriverApi::instance().cuEventQuery(fence);
                if (status == CUDA_ERROR_NOT_READY) {
                  // GPU still using this buffer — skip it, try next candidate
                  list.insert(list.begin(), candidate);
                  continue;
                }
                event_to_destroy = fence;
              }
              pooled_order_.erase(pooled_it->second.order_it);
              pooled_entries_.erase(pooled_it);
            }
            pooled_ptr = candidate;
            pooled_bytes_ -= sizeBytes;
            break;
          }
        }
      }
      if (pooled_ptr == 0) {
        needs_alloc = true;
      }
    }

    if (event_to_destroy != nullptr) {
      CudaDriverApi::instance().cuEventDestroy(event_to_destroy);
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

    const CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(ptrVoid);

    // V2 SAFETY: Record event on the provided stream so we know when GPU
    // work using this buffer completes. Acquired buffers won't be reused
    // until the event signals completion.
    CUevent event = nullptr;
    if (stream != nullptr) {
      CudaDriverApi &api = CudaDriverApi::instance();
      CUresult err = api.cuEventCreate(&event, CU_EVENT_DISABLE_TIMING);
      if (err == CUDA_SUCCESS) {
        err = api.cuEventRecord(event, stream);
        if (err != CUDA_SUCCESS) {
          api.cuEventDestroy(event);
          event = nullptr;
        }
      } else {
        event = nullptr;
      }
    }

    std::vector<CUdeviceptr> to_free;
    std::vector<CUevent> events_to_destroy;
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
      PooledEntry pooled_entry{sizeBytes, std::prev(pooled_order_.end()), event};
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
        if (victim_it->second.fence_event != nullptr) {
          events_to_destroy.push_back(victim_it->second.fence_event);
        }
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
    for (CUevent e : events_to_destroy) {
      CudaDriverApi::instance().cuEventDestroy(e);
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
    CUevent fence_event = nullptr;  // V2: stream-ordering fence
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
  // V2: During graph capture, reuse cached module (cuModuleLoadDataEx not capturable)
  if (g_capture_stream_override != nullptr && g_capture_module_cache.active &&
      g_capture_module_cache.module != nullptr) {
    return g_capture_module_cache.module;
  }
  ScopedContext scoped_context;
  CudaDriverApi &api = CudaDriverApi::instance();
  CUmodule module = nullptr;  // V2 SAFETY: null-init before load attempt
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
    // V2 SAFETY: Ensure module handle is null on failure (driver may leave
    // it in partial state). Clean up before throwing.
    if (module != nullptr) {
      api.cuModuleUnload(module);
      module = nullptr;
    }
    if (error_log[0] != '\0') {
      std::fprintf(stderr, "CUDA JIT compilation failed with: '%s'\n", error_log);
    }
    checkCuda(result, "cuModuleLoadDataEx");
  }
  // Cache for graph capture reuse
  g_capture_module_cache.module = module;
  traceGpuRuntimeEvent(TraceEventKind::kGpuModuleLoad, "mgpuModuleLoadJIT");
  return module;
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuModuleUnload(CUmodule module) {
  // V2: During graph capture or warm-up, don't unload — module must stay valid
  if (g_capture_module_cache.active || g_capture_module_cache.warmup) {
    return;
  }
  ScopedContext scoped_context;
  checkCuda(CudaDriverApi::instance().cuModuleUnload(module), "cuModuleUnload");
}

MATCORE_GPU_RUNTIME_EXPORT CUfunction mgpuModuleGetFunction(CUmodule module,
                                                            const char *name) {
  // V2: During graph capture, reuse cached function
  if (g_capture_stream_override != nullptr && g_capture_module_cache.active &&
      g_capture_module_cache.function != nullptr) {
    return g_capture_module_cache.function;
  }
  ScopedContext scoped_context;
  CUfunction function = nullptr;
  checkCuda(CudaDriverApi::instance().cuModuleGetFunction(&function, module, name),
            "cuModuleGetFunction");
  // Cache for graph capture reuse
  g_capture_module_cache.function = function;
  return function;
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuLaunchKernel(
    CUfunction function, std::intptr_t gridX, std::intptr_t gridY,
    std::intptr_t gridZ, std::intptr_t blockX, std::intptr_t blockY,
    std::intptr_t blockZ, int32_t smem, CUstream stream, void **params,
    void **extra, size_t /*paramsCount*/) {
  ScopedContext scoped_context;

  // V2 SAFETY: Param validation gate — dry-run mode to catch bad params
  // before they reach the GPU and potentially cause kernel panics.
  static const bool validate_params =
      std::getenv("MATCORE_VALIDATE_KERNEL_PARAMS") != nullptr;
  if (validate_params) {
    if (function == nullptr) {
      fail("MATCORE_VALIDATE_KERNEL_PARAMS: kernel function is null");
    }
    if (gridX <= 0 || gridY <= 0 || gridZ <= 0) {
      fail("MATCORE_VALIDATE_KERNEL_PARAMS: invalid grid dimensions (" +
           std::to_string(gridX) + "," + std::to_string(gridY) + "," +
           std::to_string(gridZ) + ")");
    }
    if (blockX <= 0 || blockY <= 0 || blockZ <= 0) {
      fail("MATCORE_VALIDATE_KERNEL_PARAMS: invalid block dimensions (" +
           std::to_string(blockX) + "," + std::to_string(blockY) + "," +
           std::to_string(blockZ) + ")");
    }
    if (smem < 0) {
      fail("MATCORE_VALIDATE_KERNEL_PARAMS: negative shared memory (" +
           std::to_string(smem) + ")");
    }
  }

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
  // V2: During graph capture, return the capture stream so all ops are captured.
  if (g_capture_stream_override != nullptr) {
    return g_capture_stream_override;
  }
  return GpuStreamPool::instance().acquire();
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuStreamDestroy(CUstream stream) {
  // V2: If using capture override, don't destroy the shared capture stream.
  if (g_capture_stream_override != nullptr && stream == g_capture_stream_override) {
    return;
  }
  GpuStreamPool::instance().release(stream);
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuStreamSynchronize(CUstream stream) {
  ScopedContext scoped_context;
  // V2 SAFETY: cuStreamSynchronize is ILLEGAL during CUDA graph capture.
  // If the stream is currently being captured, skip the sync — the graph
  // captures ordering dependencies implicitly.
  if (stream != nullptr) {
    CUstreamCaptureStatus capture_status;
    CUresult qr = CudaDriverApi::instance().cuStreamIsCapturing(
        stream, &capture_status);
    if (qr == CUDA_SUCCESS &&
        capture_status == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
      return;  // Skip sync during capture
    }
  }
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
  // V2: cuMemAlloc is not capturable. If we're inside graph capture,
  // staging buffers indicate a non-device-resident path — reject.
  if (g_capture_stream_override != nullptr) {
    fail("mgpuMemAlloc called during CUDA graph capture. "
         "graph_mode requires all tensors to be device-resident (no staging).");
  }
  return GpuMemoryPool::instance().acquire(sizeBytes);
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemFree(void *ptr, CUstream stream) {
  // V2: Skip free during graph capture (matches alloc guard above)
  if (g_capture_stream_override != nullptr) {
    return;
  }
  GpuMemoryPool::instance().release(ptr, stream);
}

// V2 SAFETY: Uses cuMemcpyAsync with the execution stream. This is required
// for CUDA graph stream capture — synchronous cuMemcpy cannot be captured.
// The staging pass inserts explicit gpu.wait barriers, so async is safe here.
MATCORE_GPU_RUNTIME_EXPORT void mgpuMemcpy(void *dst, void *src,
                                           size_t sizeBytes, CUstream stream) {
  ScopedContext scoped_context;
  checkCuda(CudaDriverApi::instance().cuMemcpyAsync(
                reinterpret_cast<CUdeviceptr>(dst),
                reinterpret_cast<CUdeviceptr>(src), sizeBytes, stream),
            "cuMemcpyAsync");
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

// ---------------------------------------------------------------------------
// Device Buffer Manager — ownership-tracked device allocations for Python
// DeviceTensor objects. Uses GpuMemoryPool for allocation/free. Each handle
// carries a monotonic alloc_id to prevent use-after-free / stale handle bugs.
// ---------------------------------------------------------------------------
class DeviceBufferManager {
public:
  static DeviceBufferManager &instance() {
    static DeviceBufferManager mgr;
    return mgr;
  }

  DeviceBufferHandle allocate(uint64_t size_bytes) {
    if (size_bytes == 0) {
      fail("matcore_device_alloc: size_bytes must be > 0");
    }
    void *raw = GpuMemoryPool::instance().acquire(size_bytes);
    const CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(raw);
    DeviceBufferHandle handle;
    handle.ptr = static_cast<uint64_t>(ptr);
    handle.size_bytes = size_bytes;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      handle.alloc_id = next_alloc_id_++;
      live_allocations_[handle.alloc_id] = {ptr, size_bytes};
    }
    return handle;
  }

  void free(DeviceBufferHandle handle) {
    CUdeviceptr ptr = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = live_allocations_.find(handle.alloc_id);
      if (it == live_allocations_.end()) {
        fail("matcore_device_free: invalid or already-freed handle "
             "(alloc_id=" + std::to_string(handle.alloc_id) + ")");
      }
      if (it->second.ptr != static_cast<CUdeviceptr>(handle.ptr) ||
          it->second.size_bytes != handle.size_bytes) {
        fail("matcore_device_free: handle fields don't match live allocation");
      }
      ptr = it->second.ptr;
      live_allocations_.erase(it);
    }
    {
      ScopedContext ctx;
      checkCuda(CudaDriverApi::instance().cuCtxSynchronize(), "cuCtxSynchronize");
    }
    GpuMemoryPool::instance().release(reinterpret_cast<void *>(ptr), nullptr);
  }

  void upload(DeviceBufferHandle dst, const void *host_src, uint64_t size_bytes) {
    validateLive(dst, "matcore_device_upload");
    if (size_bytes > dst.size_bytes) {
      fail("matcore_device_upload: size_bytes (" + std::to_string(size_bytes) +
           ") exceeds buffer capacity (" + std::to_string(dst.size_bytes) + ")");
    }
    if (host_src == nullptr) {
      fail("matcore_device_upload: host_src is null");
    }
    ScopedContext ctx;
    checkCuda(CudaDriverApi::instance().cuMemcpyHtoD(
                  static_cast<CUdeviceptr>(dst.ptr), host_src, size_bytes),
              "cuMemcpyHtoD(H→D)");
  }

  void download(void *host_dst, DeviceBufferHandle src, uint64_t size_bytes) {
    validateLive(src, "matcore_device_download");
    if (size_bytes > src.size_bytes) {
      fail("matcore_device_download: size_bytes (" + std::to_string(size_bytes) +
           ") exceeds buffer capacity (" + std::to_string(src.size_bytes) + ")");
    }
    if (host_dst == nullptr) {
      fail("matcore_device_download: host_dst is null");
    }
    ScopedContext ctx;
    checkCuda(CudaDriverApi::instance().cuMemcpyDtoH(
                  host_dst, static_cast<CUdeviceptr>(src.ptr), size_bytes),
              "cuMemcpyDtoH(D→H)");
  }

  bool isValid(DeviceBufferHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_allocations_.find(handle.alloc_id);
    if (it == live_allocations_.end()) {
      return false;
    }
    return it->second.ptr == static_cast<CUdeviceptr>(handle.ptr) &&
           it->second.size_bytes == handle.size_bytes;
  }

private:
  DeviceBufferManager() = default;

  void validateLive(DeviceBufferHandle handle, const char *caller) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_allocations_.find(handle.alloc_id);
    if (it == live_allocations_.end()) {
      fail(std::string(caller) + ": invalid or freed handle (alloc_id=" +
           std::to_string(handle.alloc_id) + ")");
    }
    if (it->second.ptr != static_cast<CUdeviceptr>(handle.ptr)) {
      fail(std::string(caller) + ": handle pointer mismatch (stale handle?)");
    }
  }

  struct LiveAllocation {
    CUdeviceptr ptr = 0;
    uint64_t size_bytes = 0;
  };

  std::mutex mutex_;
  uint64_t next_alloc_id_ = 1;
  std::unordered_map<uint64_t, LiveAllocation> live_allocations_;
};


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

#if __has_include("cuda.h")

DeviceBufferHandle matcore_device_alloc(std::uint64_t size_bytes) {
  return DeviceBufferManager::instance().allocate(size_bytes);
}

void matcore_device_free(DeviceBufferHandle handle) {
  DeviceBufferManager::instance().free(handle);
}

void matcore_device_upload(DeviceBufferHandle dst, const void *host_src,
                           std::uint64_t size_bytes) {
  DeviceBufferManager::instance().upload(dst, host_src, size_bytes);
}

void matcore_device_download(void *host_dst, DeviceBufferHandle src,
                             std::uint64_t size_bytes) {
  DeviceBufferManager::instance().download(host_dst, src, size_bytes);
}

bool matcore_device_is_valid(DeviceBufferHandle handle) {
  return DeviceBufferManager::instance().isValid(handle);
}

void matcore_device_zero(DeviceBufferHandle handle) {
  auto &mgr = DeviceBufferManager::instance();
  if (!mgr.isValid(handle))
    throw std::runtime_error("matcore_device_zero: invalid or stale handle");
  ScopedContext ctx;
  CUdeviceptr dptr = static_cast<CUdeviceptr>(handle.ptr);
  if (handle.size_bytes % 4 == 0) {
    CUresult err = CudaDriverApi::instance().cuMemsetD32Async(dptr, 0, handle.size_bytes / 4,
                                          /*stream=*/nullptr);
    if (err != CUDA_SUCCESS)
      throw std::runtime_error("matcore_device_zero: cuMemsetD32Async failed, err=" +
                               std::to_string(err));
  } else if (handle.size_bytes % 2 == 0) {
    CUresult err = CudaDriverApi::instance().cuMemsetD16Async(dptr, 0, handle.size_bytes / 2,
                                          /*stream=*/nullptr);
    if (err != CUDA_SUCCESS)
      throw std::runtime_error("matcore_device_zero: cuMemsetD16Async failed, err=" +
                               std::to_string(err));
  } else {
    throw std::runtime_error(
        "matcore_device_zero: buffer size not 2-aligned (" +
        std::to_string(handle.size_bytes) + " bytes)");
  }
}

void matcore_device_zero_raw(void *device_ptr, uint64_t size_bytes) {
  if (device_ptr == nullptr || size_bytes == 0) return;
  ScopedContext ctx;
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  if (size_bytes % 4 == 0) {
    checkCuda(CudaDriverApi::instance().cuMemsetD32Async(
                  dptr, 0, size_bytes / 4, /*stream=*/nullptr),
              "cuMemsetD32Async(zero_raw)");
  } else if (size_bytes % 2 == 0) {
    checkCuda(CudaDriverApi::instance().cuMemsetD16Async(
                  dptr, 0, size_bytes / 2, /*stream=*/nullptr),
              "cuMemsetD16Async(zero_raw)");
  } else {
    throw std::runtime_error(
        "matcore_device_zero_raw: buffer size not 2-aligned (" +
        std::to_string(size_bytes) + " bytes)");
  }
}

void matcore_device_zero_raw_on_stream(void *device_ptr, uint64_t size_bytes,
                                       void *stream) {
  if (device_ptr == nullptr || size_bytes == 0) return;
  ScopedContext ctx;
  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  CUstream cu_stream = reinterpret_cast<CUstream>(stream);
  if (size_bytes % 4 == 0) {
    checkCuda(CudaDriverApi::instance().cuMemsetD32Async(
                  dptr, 0, size_bytes / 4, cu_stream),
              "cuMemsetD32Async(zero_raw_on_stream)");
  } else if (size_bytes % 2 == 0) {
    checkCuda(CudaDriverApi::instance().cuMemsetD16Async(
                  dptr, 0, size_bytes / 2, cu_stream),
              "cuMemsetD16Async(zero_raw_on_stream)");
  } else {
    throw std::runtime_error(
        "matcore_device_zero_raw_on_stream: buffer size not 2-aligned (" +
        std::to_string(size_bytes) + " bytes)");
  }
}

// -------------------------------------------------------------------------
// V2 Pillar 2: CUDA Graph capture/replay
// -------------------------------------------------------------------------

void *matcore_graph_stream_create() {
  ScopedContext ctx;
  CUstream stream = nullptr;
  checkCuda(CudaDriverApi::instance().cuStreamCreate(
                &stream, CU_STREAM_NON_BLOCKING),
            "cuStreamCreate(graph)");
  return reinterpret_cast<void *>(stream);
}

void matcore_graph_begin_capture(void *stream) {
  ScopedContext ctx;
  CUstream cu_stream = reinterpret_cast<CUstream>(stream);
  if (cu_stream == nullptr) {
    fail("matcore_graph_begin_capture: stream is null");
  }
  checkCuda(CudaDriverApi::instance().cuStreamBeginCapture(
                cu_stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
            "cuStreamBeginCapture");
}

void *matcore_graph_end_capture(void *stream) {
  ScopedContext ctx;
  CUstream cu_stream = reinterpret_cast<CUstream>(stream);
  CUgraph graph = nullptr;
  checkCuda(CudaDriverApi::instance().cuStreamEndCapture(cu_stream, &graph),
            "cuStreamEndCapture");
  if (graph == nullptr) {
    fail("matcore_graph_end_capture: capture produced null graph");
  }
  CUgraphExec graph_exec = nullptr;
  checkCuda(CudaDriverApi::instance().cuGraphInstantiate(
                &graph_exec, graph, 0),
            "cuGraphInstantiateWithFlags");
  // Destroy the intermediate graph object — we only need the exec
  CudaDriverApi::instance().cuGraphDestroy(graph);
  return reinterpret_cast<void *>(graph_exec);
}

void matcore_graph_launch(void *graph_exec, void *stream) {
  ScopedContext ctx;
  CUgraphExec cu_exec = reinterpret_cast<CUgraphExec>(graph_exec);
  CUstream cu_stream = reinterpret_cast<CUstream>(stream);
  if (cu_exec == nullptr) {
    fail("matcore_graph_launch: graph_exec is null");
  }
  checkCuda(CudaDriverApi::instance().cuGraphLaunch(cu_exec, cu_stream),
            "cuGraphLaunch");
}

void matcore_graph_exec_destroy(void *graph_exec) {
  if (graph_exec == nullptr) return;
  ScopedContext ctx;
  CUgraphExec cu_exec = reinterpret_cast<CUgraphExec>(graph_exec);
  checkCuda(CudaDriverApi::instance().cuGraphExecDestroy(cu_exec),
            "cuGraphExecDestroy");
}

void matcore_graph_stream_destroy(void *stream) {
  if (stream == nullptr) return;
  ScopedContext ctx;
  CUstream cu_stream = reinterpret_cast<CUstream>(stream);
  checkCuda(CudaDriverApi::instance().cuStreamDestroy(cu_stream),
            "cuStreamDestroy(graph)");
}

void matcore_set_capture_stream_override(void *stream) {
  g_capture_stream_override = reinterpret_cast<CUstream>(stream);
  if (stream != nullptr) {
    g_capture_module_cache.active = true;
  } else {
    g_capture_module_cache.active = false;
  }
}

void matcore_set_graph_warmup(bool enable) {
  g_capture_module_cache.warmup = enable;
  if (enable) {
    // Clear stale cache before warm-up
    g_capture_module_cache.module = nullptr;
    g_capture_module_cache.function = nullptr;
  }
}

void matcore_stream_synchronize(void *stream) {
  ScopedContext ctx;
  CUstream cu_stream = reinterpret_cast<CUstream>(stream);
  checkCuda(CudaDriverApi::instance().cuStreamSynchronize(cu_stream),
            "cuStreamSynchronize(graph)");
}

#else

DeviceBufferHandle matcore_device_alloc(std::uint64_t) {
  throw std::runtime_error("matcore_device_alloc: CUDA not available at build time");
}
void matcore_device_free(DeviceBufferHandle) {
  throw std::runtime_error("matcore_device_free: CUDA not available at build time");
}
void matcore_device_upload(DeviceBufferHandle, const void *, std::uint64_t) {
  throw std::runtime_error("matcore_device_upload: CUDA not available at build time");
}
void matcore_device_download(void *, DeviceBufferHandle, std::uint64_t) {
  throw std::runtime_error("matcore_device_download: CUDA not available at build time");
}
bool matcore_device_is_valid(DeviceBufferHandle) { return false; }
void matcore_device_zero(DeviceBufferHandle) {
  throw std::runtime_error("matcore_device_zero: CUDA not available at build time");
}
void matcore_device_zero_raw(void *, uint64_t) {
  throw std::runtime_error("matcore_device_zero_raw: CUDA not available at build time");
}
void matcore_device_zero_raw_on_stream(void *, uint64_t, void *) {
  throw std::runtime_error("matcore_device_zero_raw_on_stream: CUDA not available at build time");
}
void *matcore_graph_stream_create() {
  throw std::runtime_error("CUDA not available at build time");
}
void matcore_graph_begin_capture(void *) {
  throw std::runtime_error("CUDA not available at build time");
}
void *matcore_graph_end_capture(void *) {
  throw std::runtime_error("CUDA not available at build time");
}
void matcore_graph_launch(void *, void *) {
  throw std::runtime_error("CUDA not available at build time");
}
void matcore_graph_exec_destroy(void *) {}
void matcore_graph_stream_destroy(void *) {}
void matcore_set_capture_stream_override(void *) {}
void matcore_set_graph_warmup(bool) {}
void matcore_stream_synchronize(void *) {}

#endif

}  // namespace matcore
