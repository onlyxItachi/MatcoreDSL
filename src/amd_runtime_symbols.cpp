#include "amd_runtime_symbols.h"

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

namespace matcore {
namespace {

using hipError_t = int;
using hipModule_t = void *;
using hipFunction_t = void *;
using hipStream_t = void *;
using hipDeviceptr_t = std::uint64_t;
using hipMemcpyKind = int;

constexpr hipError_t kHipSuccess = 0;
constexpr unsigned int kHipStreamNonBlocking = 1;
constexpr hipMemcpyKind kHipMemcpyDeviceToDevice = 3;
constexpr hipMemcpyKind kHipMemcpyDefault = 4;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error("MatCore AMD runtime: " + message);
}

struct HipApi {
  using HipInitFn = hipError_t (*)(unsigned int);
  using HipModuleLoadDataFn = hipError_t (*)(hipModule_t *, const void *);
  using HipModuleUnloadFn = hipError_t (*)(hipModule_t);
  using HipModuleGetFunctionFn =
      hipError_t (*)(hipFunction_t *, hipModule_t, const char *);
  using HipModuleLaunchKernelFn =
      hipError_t (*)(hipFunction_t, unsigned int, unsigned int, unsigned int,
                     unsigned int, unsigned int, unsigned int, unsigned int,
                     hipStream_t, void **, void **);
  using HipStreamCreateWithFlagsFn = hipError_t (*)(hipStream_t *, unsigned int);
  using HipStreamDestroyFn = hipError_t (*)(hipStream_t);
  using HipStreamSynchronizeFn = hipError_t (*)(hipStream_t);
  using HipMallocFn = hipError_t (*)(void **, std::size_t);
  using HipFreeFn = hipError_t (*)(void *);
  using HipMemcpyAsyncFn =
      hipError_t (*)(void *, const void *, std::size_t, hipMemcpyKind, hipStream_t);
  using HipMemsetD32AsyncFn =
      hipError_t (*)(hipDeviceptr_t, std::int32_t, std::size_t, hipStream_t);
  using HipMemsetD16AsyncFn =
      hipError_t (*)(hipDeviceptr_t, std::uint16_t, std::size_t, hipStream_t);
  using HipHostRegisterFn = hipError_t (*)(void *, std::size_t, unsigned int);
  using HipGetErrorNameFn = const char *(*)(hipError_t);

  void *handle = nullptr;
  HipInitFn hipInit = nullptr;
  HipModuleLoadDataFn hipModuleLoadData = nullptr;
  HipModuleUnloadFn hipModuleUnload = nullptr;
  HipModuleGetFunctionFn hipModuleGetFunction = nullptr;
  HipModuleLaunchKernelFn hipModuleLaunchKernel = nullptr;
  HipStreamCreateWithFlagsFn hipStreamCreateWithFlags = nullptr;
  HipStreamDestroyFn hipStreamDestroy = nullptr;
  HipStreamSynchronizeFn hipStreamSynchronize = nullptr;
  HipMallocFn hipMalloc = nullptr;
  HipFreeFn hipFree = nullptr;
  HipMemcpyAsyncFn hipMemcpyAsync = nullptr;
  HipMemsetD32AsyncFn hipMemsetD32Async = nullptr;
  HipMemsetD16AsyncFn hipMemsetD16Async = nullptr;
  HipHostRegisterFn hipHostRegister = nullptr;
  HipGetErrorNameFn hipGetErrorName = nullptr;
};

template <typename Fn>
Fn loadSymbol(void *handle, const char *name) {
  void *symbol = dlsym(handle, name);
  if (symbol == nullptr) {
    fail("failed to resolve runtime symbol '" + std::string(name) + "'");
  }
  return reinterpret_cast<Fn>(symbol);
}

HipApi *tryLoadHipApi() {
  static std::once_flag init_flag;
  static HipApi api;
  static bool init_ok = false;

  std::call_once(init_flag, []() {
    api.handle = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
    if (api.handle == nullptr) {
      api.handle = dlopen("/lib/x86_64-linux-gnu/libamdhip64.so",
                          RTLD_NOW | RTLD_LOCAL);
    }
    if (api.handle == nullptr) {
      return;
    }

    try {
      api.hipInit = loadSymbol<HipApi::HipInitFn>(api.handle, "hipInit");
      api.hipModuleLoadData =
          loadSymbol<HipApi::HipModuleLoadDataFn>(api.handle, "hipModuleLoadData");
      api.hipModuleUnload =
          loadSymbol<HipApi::HipModuleUnloadFn>(api.handle, "hipModuleUnload");
      api.hipModuleGetFunction = loadSymbol<HipApi::HipModuleGetFunctionFn>(
          api.handle, "hipModuleGetFunction");
      api.hipModuleLaunchKernel = loadSymbol<HipApi::HipModuleLaunchKernelFn>(
          api.handle, "hipModuleLaunchKernel");
      api.hipStreamCreateWithFlags =
          loadSymbol<HipApi::HipStreamCreateWithFlagsFn>(api.handle,
                                                          "hipStreamCreateWithFlags");
      api.hipStreamDestroy =
          loadSymbol<HipApi::HipStreamDestroyFn>(api.handle, "hipStreamDestroy");
      api.hipStreamSynchronize = loadSymbol<HipApi::HipStreamSynchronizeFn>(
          api.handle, "hipStreamSynchronize");
      api.hipMalloc = loadSymbol<HipApi::HipMallocFn>(api.handle, "hipMalloc");
      api.hipFree = loadSymbol<HipApi::HipFreeFn>(api.handle, "hipFree");
      api.hipMemcpyAsync =
          loadSymbol<HipApi::HipMemcpyAsyncFn>(api.handle, "hipMemcpyAsync");
      api.hipMemsetD32Async =
          loadSymbol<HipApi::HipMemsetD32AsyncFn>(api.handle, "hipMemsetD32Async");
      api.hipMemsetD16Async =
          loadSymbol<HipApi::HipMemsetD16AsyncFn>(api.handle, "hipMemsetD16Async");
      api.hipHostRegister =
          loadSymbol<HipApi::HipHostRegisterFn>(api.handle, "hipHostRegister");
      api.hipGetErrorName =
          loadSymbol<HipApi::HipGetErrorNameFn>(api.handle, "hipGetErrorName");
    } catch (const std::exception &) {
      dlclose(api.handle);
      api = HipApi{};
      return;
    }

    if (api.hipInit(0) != kHipSuccess) {
      dlclose(api.handle);
      api = HipApi{};
      return;
    }

    init_ok = true;
  });

  return init_ok ? &api : nullptr;
}

HipApi &requireHipApi() {
  HipApi *api = tryLoadHipApi();
  if (api == nullptr) {
    fail("HIP runtime is unavailable");
  }
  return *api;
}

void checkHip(const HipApi &api, hipError_t result, const char *expr) {
  if (result == kHipSuccess) {
    return;
  }
  const char *name = api.hipGetErrorName != nullptr ? api.hipGetErrorName(result)
                                                     : nullptr;
  fail("'" + std::string(expr) + "' failed with '" +
       (name != nullptr ? std::string(name) : std::string("<unknown>")) + "'");
}

extern "C" __attribute__((visibility("default"))) hipModule_t
mgpuModuleLoadHip(void *data, size_t /*gpuBlobSize*/) {
  HipApi &api = requireHipApi();
  hipModule_t module = nullptr;
  checkHip(api, api.hipModuleLoadData(&module, data), "hipModuleLoadData");
  return module;
}

extern "C" __attribute__((visibility("default"))) hipModule_t
mgpuModuleLoadJITHip(void *data, int /*optLevel*/) {
  return mgpuModuleLoadHip(data, 0);
}

extern "C" __attribute__((visibility("default"))) void
mgpuModuleUnloadHip(hipModule_t module) {
  if (module == nullptr) {
    return;
  }
  HipApi &api = requireHipApi();
  checkHip(api, api.hipModuleUnload(module), "hipModuleUnload");
}

extern "C" __attribute__((visibility("default"))) hipFunction_t
mgpuModuleGetFunctionHip(hipModule_t module, const char *name) {
  HipApi &api = requireHipApi();
  hipFunction_t function = nullptr;
  checkHip(api, api.hipModuleGetFunction(&function, module, name),
           "hipModuleGetFunction");
  return function;
}

extern "C" __attribute__((visibility("default"))) void mgpuLaunchKernelHip(
    hipFunction_t function, std::intptr_t gridX, std::intptr_t gridY,
    std::intptr_t gridZ, std::intptr_t blockX, std::intptr_t blockY,
    std::intptr_t blockZ, int32_t smem, hipStream_t stream, void **params,
    void **extra, size_t /*paramsCount*/) {
  HipApi &api = requireHipApi();
  checkHip(api,
           api.hipModuleLaunchKernel(
               function, static_cast<unsigned int>(gridX),
               static_cast<unsigned int>(gridY), static_cast<unsigned int>(gridZ),
               static_cast<unsigned int>(blockX), static_cast<unsigned int>(blockY),
               static_cast<unsigned int>(blockZ), static_cast<unsigned int>(smem),
               stream, params, extra),
           "hipModuleLaunchKernel");
}

extern "C" __attribute__((visibility("default"))) hipStream_t
mgpuStreamCreateHip() {
  HipApi &api = requireHipApi();
  hipStream_t stream = nullptr;
  checkHip(api, api.hipStreamCreateWithFlags(&stream, kHipStreamNonBlocking),
           "hipStreamCreateWithFlags");
  return stream;
}

extern "C" __attribute__((visibility("default"))) void
mgpuStreamDestroyHip(hipStream_t stream) {
  if (stream == nullptr) {
    return;
  }
  HipApi &api = requireHipApi();
  checkHip(api, api.hipStreamDestroy(stream), "hipStreamDestroy");
}

extern "C" __attribute__((visibility("default"))) void
mgpuStreamSynchronizeHip(hipStream_t stream) {
  if (stream == nullptr) {
    return;
  }
  HipApi &api = requireHipApi();
  checkHip(api, api.hipStreamSynchronize(stream), "hipStreamSynchronize");
}

extern "C" __attribute__((visibility("default"))) void *mgpuMemAllocHip(
    uint64_t sizeBytes, hipStream_t /*stream*/, bool /*isHostShared*/) {
  HipApi &api = requireHipApi();
  if (sizeBytes == 0) {
    return nullptr;
  }
  void *ptr = nullptr;
  checkHip(api, api.hipMalloc(&ptr, static_cast<std::size_t>(sizeBytes)),
           "hipMalloc");
  return ptr;
}

extern "C" __attribute__((visibility("default"))) void
mgpuMemFreeHip(void *ptr, hipStream_t /*stream*/) {
  if (ptr == nullptr) {
    return;
  }
  HipApi &api = requireHipApi();
  checkHip(api, api.hipFree(ptr), "hipFree");
}

extern "C" __attribute__((visibility("default"))) void mgpuMemcpyHip(
    void *dst, void *src, size_t sizeBytes, hipStream_t stream) {
  HipApi &api = requireHipApi();
  checkHip(api,
           api.hipMemcpyAsync(dst, src, sizeBytes, kHipMemcpyDefault, stream),
           "hipMemcpyAsync");
}

extern "C" __attribute__((visibility("default"))) void mgpuMemset32Hip(
    void *dst, unsigned int value, size_t count, hipStream_t stream) {
  HipApi &api = requireHipApi();
  checkHip(api,
           api.hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dst),
                                 static_cast<std::int32_t>(value), count, stream),
           "hipMemsetD32Async");
}

extern "C" __attribute__((visibility("default"))) void mgpuMemset16Hip(
    void *dst, unsigned short value, size_t count, hipStream_t stream) {
  HipApi &api = requireHipApi();
  checkHip(api,
           api.hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(dst), value, count,
                                 stream),
           "hipMemsetD16Async");
}

extern "C" __attribute__((visibility("default"))) void
mgpuMemHostRegisterHip(void *ptr, uint64_t sizeBytes) {
  if (ptr == nullptr || sizeBytes == 0) {
    return;
  }
  HipApi &api = requireHipApi();
  checkHip(api, api.hipHostRegister(ptr, static_cast<std::size_t>(sizeBytes), 0),
           "hipHostRegister");
}

}  // namespace

bool registerAmdRuntimeSymbols(mlir::ExecutionEngine &engine) {
  if (tryLoadHipApi() == nullptr) {
    return false;
  }

  engine.registerSymbols([](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    auto add_symbol = [&](llvm::StringRef name, auto *fn) {
      symbols[interner(name)] = llvm::orc::ExecutorSymbolDef(
          llvm::orc::ExecutorAddr::fromPtr(fn), llvm::JITSymbolFlags::Exported);
    };

    add_symbol("mgpuModuleLoad", &mgpuModuleLoadHip);
    add_symbol("mgpuModuleLoadJIT", &mgpuModuleLoadJITHip);
    add_symbol("mgpuModuleUnload", &mgpuModuleUnloadHip);
    add_symbol("mgpuModuleGetFunction", &mgpuModuleGetFunctionHip);
    add_symbol("mgpuLaunchKernel", &mgpuLaunchKernelHip);
    add_symbol("mgpuStreamCreate", &mgpuStreamCreateHip);
    add_symbol("mgpuStreamDestroy", &mgpuStreamDestroyHip);
    add_symbol("mgpuStreamSynchronize", &mgpuStreamSynchronizeHip);
    add_symbol("mgpuMemAlloc", &mgpuMemAllocHip);
    add_symbol("mgpuMemFree", &mgpuMemFreeHip);
    add_symbol("mgpuMemcpy", &mgpuMemcpyHip);
    add_symbol("mgpuMemset32", &mgpuMemset32Hip);
    add_symbol("mgpuMemset16", &mgpuMemset16Hip);
    add_symbol("mgpuMemHostRegister", &mgpuMemHostRegisterHip);
    return symbols;
  });
  return true;
}

}  // namespace matcore
