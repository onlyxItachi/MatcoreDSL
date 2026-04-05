#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>

#include "matcore/target_registry.h"

namespace matcore {

struct X86RuntimeCapabilities {
  bool detected = false;
  std::unordered_set<std::string> enabled_features;
};

struct NvidiaRuntimeCapabilities {
  bool driver_available = false;
  bool device_present = false;
  int compute_major = 0;
  int compute_minor = 0;
  std::string chip = "unknown";
};

struct RuntimeCapabilities {
  X86RuntimeCapabilities x86;
  NvidiaRuntimeCapabilities nvidia;
  bool nvidia_probed = false;   // Lazy: true after detectNvidiaRuntime() called
  bool rocm_probed = false;     // Lazy: true after detectRocmRuntime() called
  bool rocm_library_available = false;
  bool rocm_device_present = false;
  bool rocm_runtime_available = false;
  bool npu_runtime_available = false;
};

enum class GpuPreflightStatus { kPass, kFail, kAdvisoryWarning };

struct GpuPreflightResult {
  GpuPreflightStatus status = GpuPreflightStatus::kFail;
  std::string diagnostic;
};

RuntimeCapabilities DetectRuntimeCapabilities();
GpuPreflightResult gpuPreflightCheck(TargetKind target);
bool acquireGpuBackendClaim(TargetKind target, std::string *denial_reason);
void releaseGpuBackendClaim(TargetKind target);
bool SupportsX86Feature(const RuntimeCapabilities &runtime,
                        std::string_view feature);
bool CanExecuteOnHost(const RuntimeCapabilities &runtime,
                      const ExecutionRequirements &requirements,
                      std::string *denial_reason);
std::string FormatExecutionDeniedMessage(const RequestedTargetProfile &profile,
                                         const std::string &denial_reason);

}  // namespace matcore
