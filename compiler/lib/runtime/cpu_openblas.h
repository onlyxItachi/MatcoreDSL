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

enum class OpenBlasExecutionStatusV1 : std::uint8_t {
  success = 0,
  unavailable = 1,
  invalid_problem = 2,
  invalid_thread_count = 3,
  unsupported_fp_environment = 4,
};

OpenBlasProviderInfoV1 openblas_provider_info_v1() noexcept;

OpenBlasExecutionStatusV1 execute_openblas_gemm_f32_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, std::uint32_t requested_threads,
    std::uint32_t *actual_threads) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
