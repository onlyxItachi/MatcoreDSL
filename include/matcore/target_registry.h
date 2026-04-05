#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "matcore/kernel_ir.h"

namespace matcore {

// A requested compile target profile decoupled from host runtime capabilities.
struct RequestedTargetProfile {
  TargetKind kind = TargetKind::kX86Auto;
  std::string requested;
  std::string canonical;
  std::optional<int> nvidia_sm_major;
  std::optional<int> nvidia_sm_minor;
  std::string amd_chip;
};

// Execution-side requirements derived from a requested target profile.
struct ExecutionRequirements {
  TargetKind kind = TargetKind::kX86Auto;
  std::vector<std::string> required_x86_features;
  bool requires_nvidia_device = false;
  std::optional<int> min_nvidia_sm_major;
  std::optional<int> min_nvidia_sm_minor;
  bool requires_rocm_runtime = false;
  bool requires_npu_runtime = false;
};

RequestedTargetProfile ParseRequestedTargetProfile(std::string_view requested_target);
ExecutionRequirements BuildExecutionRequirements(
    const RequestedTargetProfile &profile);
std::string CanonicalTargetString(const RequestedTargetProfile &profile);

}  // namespace matcore
