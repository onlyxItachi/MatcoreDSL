#include "matcore/gpu_runtime_symbols.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

#if __has_include("cuda.h")
#include "cuda.h"
#endif

namespace matcore {
namespace {

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
        loadSymbol<CuCtxPushCurrentFn>(handle, "cuCtxPushCurrent");
    api.cuCtxPopCurrent =
        loadSymbol<CuCtxPopCurrentFn>(handle, "cuCtxPopCurrent");
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
        loadSymbol<CuStreamDestroyFn>(handle, "cuStreamDestroy");
    api.cuStreamSynchronize = loadSymbol<CuStreamSynchronizeFn>(
        handle, "cuStreamSynchronize");
    api.cuMemAlloc = loadSymbol<CuMemAllocFn>(handle, "cuMemAlloc");
    api.cuMemFree = loadSymbol<CuMemFreeFn>(handle, "cuMemFree");
    api.cuMemcpyAsync =
        loadSymbol<CuMemcpyAsyncFn>(handle, "cuMemcpyAsync");
    api.cuMemsetD32Async =
        loadSymbol<CuMemsetD32AsyncFn>(handle, "cuMemsetD32Async");
    api.cuMemsetD16Async =
        loadSymbol<CuMemsetD16AsyncFn>(handle, "cuMemsetD16Async");
    api.cuMemHostRegister =
        loadSymbol<CuMemHostRegisterFn>(handle, "cuMemHostRegister");
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

MATCORE_GPU_RUNTIME_EXPORT CUmodule mgpuModuleLoad(void *data,
                                                   size_t /*gpuBlobSize*/) {
  ScopedContext scoped_context;
  CUmodule module = nullptr;
  checkCuda(CudaDriverApi::instance().cuModuleLoadData(&module, data),
            "cuModuleLoadData");
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
  return module;
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuModuleUnload(CUmodule module) {
  checkCuda(CudaDriverApi::instance().cuModuleUnload(module), "cuModuleUnload");
}

MATCORE_GPU_RUNTIME_EXPORT CUfunction mgpuModuleGetFunction(CUmodule module,
                                                            const char *name) {
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
  ScopedContext scoped_context;
  CUstream stream = nullptr;
  checkCuda(CudaDriverApi::instance().cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
            "cuStreamCreate");
  return stream;
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuStreamDestroy(CUstream stream) {
  checkCuda(CudaDriverApi::instance().cuStreamDestroy(stream), "cuStreamDestroy");
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuStreamSynchronize(CUstream stream) {
  checkCuda(CudaDriverApi::instance().cuStreamSynchronize(stream),
            "cuStreamSynchronize");
}

MATCORE_GPU_RUNTIME_EXPORT void *mgpuMemAlloc(uint64_t sizeBytes,
                                              CUstream /*stream*/,
                                              bool isHostShared) {
  ScopedContext scoped_context;
  if (isHostShared) {
    fail("managed CUDA allocations are not implemented in MatCore's runtime");
  }
  CUdeviceptr ptr = 0;
  if (sizeBytes == 0) {
    return reinterpret_cast<void *>(ptr);
  }
  checkCuda(CudaDriverApi::instance().cuMemAlloc(&ptr, sizeBytes), "cuMemAlloc");
  return reinterpret_cast<void *>(ptr);
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemFree(void *ptr, CUstream /*stream*/) {
  if (ptr == nullptr) {
    return;
  }
  checkCuda(CudaDriverApi::instance().cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)),
            "cuMemFree");
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemcpy(void *dst, void *src,
                                           size_t sizeBytes, CUstream stream) {
  checkCuda(CudaDriverApi::instance().cuMemcpyAsync(
                reinterpret_cast<CUdeviceptr>(dst),
                reinterpret_cast<CUdeviceptr>(src), sizeBytes, stream),
            "cuMemcpyAsync");
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemset32(void *dst, unsigned int value,
                                             size_t count, CUstream stream) {
  checkCuda(CudaDriverApi::instance().cuMemsetD32Async(
                reinterpret_cast<CUdeviceptr>(dst), value, count, stream),
            "cuMemsetD32Async");
}

MATCORE_GPU_RUNTIME_EXPORT void mgpuMemset16(void *dst, unsigned short value,
                                             size_t count, CUstream stream) {
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
    return symbols;
  });
}

#undef MATCORE_GPU_RUNTIME_EXPORT

#endif

}  // namespace

void registerGpuRuntimeSymbols(mlir::ExecutionEngine &engine, TargetKind target) {
  if (normalizeTarget(target) != TargetKind::kNvidiaDGPU) {
    return;
  }

#if __has_include("cuda.h")
  registerCudaRuntimeSymbols(engine);
#else
  fail("nvidia-dgpu requested but CUDA headers were not available at build time");
#endif
}

}  // namespace matcore
