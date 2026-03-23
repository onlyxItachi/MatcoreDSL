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
  bool rocm_runtime_available = false;
  bool npu_runtime_available = false;
};

RuntimeCapabilities DetectRuntimeCapabilities();
bool SupportsX86Feature(const RuntimeCapabilities &runtime,
                        std::string_view feature);
bool CanExecuteOnHost(const RuntimeCapabilities &runtime,
                      const ExecutionRequirements &requirements,
                      std::string *denial_reason);
std::string FormatExecutionDeniedMessage(const RequestedTargetProfile &profile,
                                         const std::string &denial_reason);

}  // namespace matcore
