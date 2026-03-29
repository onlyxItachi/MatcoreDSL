#include "matcore/runtime_capabilities.h"

#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>

#include "llvm/ADT/StringMap.h"
#include "llvm/TargetParser/Host.h"

namespace matcore {
namespace {

using CUresult = int;
using CUdevice = int;

using CuInitFn = CUresult (*)(unsigned int);
using CuDeviceGetCountFn = CUresult (*)(int *);
using CuDeviceGetFn = CUresult (*)(CUdevice *, int);
using CuDeviceGetAttributeFn = CUresult (*)(int *, int, CUdevice);

constexpr int kCudaSuccess = 0;
constexpr int kCudaComputeCapabilityMajorAttr = 75;
constexpr int kCudaComputeCapabilityMinorAttr = 76;

std::optional<void *> openSharedLibrary(const char *primary,
                                        const char *fallback = nullptr) {
  if (void *handle = dlopen(primary, RTLD_NOW | RTLD_LOCAL)) {
    return handle;
  }
  if (fallback != nullptr) {
    if (void *handle = dlopen(fallback, RTLD_NOW | RTLD_LOCAL)) {
      return handle;
    }
  }
  return std::nullopt;
}

void closeSharedLibrary(std::optional<void *> handle) {
  if (handle.has_value()) {
    dlclose(*handle);
  }
}

NvidiaRuntimeCapabilities detectNvidiaRuntime() {
  NvidiaRuntimeCapabilities out;

  std::optional<void *> libcuda =
      openSharedLibrary("libcuda.so", "/lib/x86_64-linux-gnu/libcuda.so");
  if (!libcuda.has_value()) {
    return out;
  }
  out.driver_available = true;

  auto cuInit = reinterpret_cast<CuInitFn>(dlsym(*libcuda, "cuInit"));
  auto cuDeviceGetCount =
      reinterpret_cast<CuDeviceGetCountFn>(dlsym(*libcuda, "cuDeviceGetCount"));
  auto cuDeviceGet =
      reinterpret_cast<CuDeviceGetFn>(dlsym(*libcuda, "cuDeviceGet"));
  auto cuDeviceGetAttribute =
      reinterpret_cast<CuDeviceGetAttributeFn>(dlsym(*libcuda, "cuDeviceGetAttribute"));

  if (cuInit == nullptr || cuDeviceGetCount == nullptr || cuDeviceGet == nullptr ||
      cuDeviceGetAttribute == nullptr) {
    closeSharedLibrary(libcuda);
    return out;
  }
  if (cuInit(0) != kCudaSuccess) {
    closeSharedLibrary(libcuda);
    return out;
  }

  int device_count = 0;
  if (cuDeviceGetCount(&device_count) != kCudaSuccess || device_count <= 0) {
    closeSharedLibrary(libcuda);
    return out;
  }
  out.device_present = true;

  CUdevice device = 0;
  if (cuDeviceGet(&device, /*ordinal=*/0) != kCudaSuccess) {
    closeSharedLibrary(libcuda);
    return out;
  }

  int major = 0;
  int minor = 0;
  if (cuDeviceGetAttribute(&major, kCudaComputeCapabilityMajorAttr, device) ==
          kCudaSuccess &&
      cuDeviceGetAttribute(&minor, kCudaComputeCapabilityMinorAttr, device) ==
          kCudaSuccess) {
    out.compute_major = major;
    out.compute_minor = minor;
    out.chip = "sm_" + std::to_string(major) + std::to_string(minor);
  }

  closeSharedLibrary(libcuda);
  return out;
}

bool detectRocmRuntime() {
  std::optional<void *> handle =
      openSharedLibrary("libamdhip64.so", "/lib/x86_64-linux-gnu/libamdhip64.so");
  const bool present = handle.has_value();
  closeSharedLibrary(handle);
  return present;
}

bool detectNpuRuntime() {
  for (const char *candidate :
       {"libamdxdna.so", "libxaiengine.so", "libxrt_coreutil.so"}) {
    std::optional<void *> handle = openSharedLibrary(candidate);
    if (handle.has_value()) {
      closeSharedLibrary(handle);
      return true;
    }
  }
  return false;
}

bool compareComputeCapability(const NvidiaRuntimeCapabilities &runtime,
                             int min_major, int min_minor) {
  if (!runtime.device_present) {
    return false;
  }
  if (runtime.compute_major > min_major) {
    return true;
  }
  if (runtime.compute_major == min_major && runtime.compute_minor >= min_minor) {
    return true;
  }
  return false;
}

}  // namespace

RuntimeCapabilities DetectRuntimeCapabilities() {
  RuntimeCapabilities runtime;

  llvm::StringMap<bool> host_features;
  runtime.x86.detected = llvm::sys::getHostCPUFeatures(host_features);
  if (runtime.x86.detected) {
    for (const auto &entry : host_features) {
      if (entry.second) {
        runtime.x86.enabled_features.insert(entry.first().str());
      }
    }
  }

  runtime.nvidia = detectNvidiaRuntime();
  runtime.rocm_runtime_available = detectRocmRuntime();
  runtime.npu_runtime_available = detectNpuRuntime();
  return runtime;
}

bool SupportsX86Feature(const RuntimeCapabilities &runtime,
                        std::string_view feature) {
  if (!runtime.x86.detected) {
    return false;
  }
  return runtime.x86.enabled_features.find(std::string(feature)) !=
         runtime.x86.enabled_features.end();
}

bool CanExecuteOnHost(const RuntimeCapabilities &runtime,
                      const ExecutionRequirements &requirements,
                      std::string *denial_reason) {
  auto deny = [&](const std::string &message) {
    if (denial_reason != nullptr) {
      *denial_reason = message;
    }
    return false;
  };

  for (const std::string &feature : requirements.required_x86_features) {
    if (!SupportsX86Feature(runtime, feature)) {
      return deny("host CPU does not provide required feature '" + feature + "'");
    }
  }

  if (requirements.requires_nvidia_device) {
    if (!runtime.nvidia.driver_available) {
      return deny("NVIDIA driver library is unavailable");
    }
    if (!runtime.nvidia.device_present) {
      return deny("no NVIDIA device is visible to the runtime");
    }
    if (requirements.min_nvidia_sm_major.has_value() &&
        requirements.min_nvidia_sm_minor.has_value() &&
        !compareComputeCapability(runtime.nvidia, *requirements.min_nvidia_sm_major,
                                  *requirements.min_nvidia_sm_minor)) {
      return deny("device capability " + runtime.nvidia.chip +
                  " is below required sm_" +
                  std::to_string(*requirements.min_nvidia_sm_major) +
                  std::to_string(*requirements.min_nvidia_sm_minor));
    }
  }

  if (requirements.requires_rocm_runtime && !runtime.rocm_runtime_available) {
    return deny("ROCm runtime library is unavailable");
  }
  if (requirements.requires_npu_runtime && !runtime.npu_runtime_available) {
    return deny("NPU runtime library is unavailable");
  }

  if (denial_reason != nullptr) {
    denial_reason->clear();
  }
  return true;
}

std::string FormatExecutionDeniedMessage(const RequestedTargetProfile &profile,
                                         const std::string &denial_reason) {
  return "Execution Denied: Hardware limits for target '" + profile.canonical +
         "': " + denial_reason;
}

}  // namespace matcore
