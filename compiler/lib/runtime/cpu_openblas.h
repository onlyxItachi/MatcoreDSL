#ifndef MATCORE_MDSLC_RUNTIME_CPU_OPENBLAS_H
#define MATCORE_MDSLC_RUNTIME_CPU_OPENBLAS_H

#include "cpu_planner.h"

#include <cstdint>

namespace matcore::mdslc::runtime {

struct OpenBlasProviderInfoV1 {
  bool linked = false;
  const char *package_version = "unavailable";
  const char *runtime_config = "unavailable";
  const char *runtime_core = "unavailable";
  std::int32_t parallel_model = -1;
  std::int32_t maximum_reported_threads = 0;
};

inline constexpr std::uint32_t kOpenBlasConformanceVersionV1 = 1;

// Immutable process-local evidence for the exact linked
// package/configuration/core identity. The probe is allocation-free and runs
// at most once, on the first caller with an authenticated FP environment.
struct OpenBlasConformanceReportV1 {
  std::uint32_t version = kOpenBlasConformanceVersionV1;
  std::uint64_t provider_identity_key = 0;
  bool provider_linked = false;
  bool identity_complete = false;
  bool probe_attempted = false;
  bool caller_environment_compatible = false;
  bool finite_correct = false;
  bool gradual_subnormal_correct = false;
  bool nan_input_propagated = false;
  bool infinity_classes_preserved = false;
  bool inf_times_zero_is_nan = false;
  bool opposite_infinity_sum_is_nan = false;
  bool nonfinite_classes_correct = false;
  bool control_state_preserved = false;
  bool thread_state_restored = false;
  bool conformant = false;
};

enum class OpenBlasExecutionStatusV1 : std::uint8_t {
  success = 0,
  unavailable = 1,
  invalid_problem = 2,
  invalid_thread_count = 3,
  unsupported_fp_environment = 4,
  provider_conformance_failed = 5,
  provider_state_violation = 6,
};

// Pre-call failures preserve output. A provider_state_violation is necessarily
// detected after the opaque CBLAS call, so the caller's output may already
// have been modified even though control and provider-thread restoration was
// attempted and the execution is reported as failed.

OpenBlasProviderInfoV1 openblas_provider_info_v1() noexcept;
OpenBlasConformanceReportV1 openblas_conformance_report_v1() noexcept;

OpenBlasExecutionStatusV1 execute_openblas_gemm_f32_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, std::uint32_t requested_threads,
    std::uint32_t *actual_threads) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
