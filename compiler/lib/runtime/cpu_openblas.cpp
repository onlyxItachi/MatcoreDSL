#include "cpu_openblas.h"

#include <climits>

#if MATCORE_MDSLC_HAS_OPENBLAS
#include <cblas.h>
#endif

#ifndef MATCORE_MDSLC_OPENBLAS_VERSION
#define MATCORE_MDSLC_OPENBLAS_VERSION "unavailable"
#endif

namespace matcore::mdslc::runtime {

OpenBlasProviderInfoV1 openblas_provider_info_v1() noexcept {
#if MATCORE_MDSLC_HAS_OPENBLAS
  const char *config = openblas_get_config();
  const char *core = openblas_get_corename();
  return {true,
          MATCORE_MDSLC_OPENBLAS_VERSION,
          config != nullptr ? config : "unknown",
          core != nullptr ? core : "unknown",
          openblas_get_parallel(),
          openblas_get_num_procs()};
#else
  return {};
#endif
}

OpenBlasExecutionStatusV1 execute_openblas_gemm_f32_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, std::uint32_t requested_threads,
    std::uint32_t *actual_threads) noexcept {
  if (actual_threads != nullptr) *actual_threads = 0;
#if MATCORE_MDSLC_HAS_OPENBLAS
  if (lhs == nullptr || rhs == nullptr || out == nullptr || problem.m <= 0 ||
      problem.n <= 0 || problem.k <= 0 ||
      problem.element_type != planner::CpuScalarTypeV1::f32 ||
      problem.accumulation_type != planner::CpuScalarTypeV1::f32 ||
      problem.layout != planner::CpuLayoutV1::row_major_contiguous ||
      problem.m > INT_MAX || problem.n > INT_MAX || problem.k > INT_MAX)
    return OpenBlasExecutionStatusV1::invalid_problem;
  if (requested_threads == 0 || requested_threads > INT_MAX)
    return OpenBlasExecutionStatusV1::invalid_thread_count;

  const int previous_threads =
      openblas_set_num_threads_local(static_cast<int>(requested_threads));
  const int provider_threads = openblas_get_num_threads();
  if (provider_threads <= 0 ||
      provider_threads != static_cast<int>(requested_threads)) {
    openblas_set_num_threads_local(previous_threads);
    return OpenBlasExecutionStatusV1::invalid_thread_count;
  }
  if (actual_threads != nullptr)
    *actual_threads = static_cast<std::uint32_t>(provider_threads);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              static_cast<blasint>(problem.m),
              static_cast<blasint>(problem.n),
              static_cast<blasint>(problem.k), 1.0F, lhs,
              static_cast<blasint>(problem.k), rhs,
              static_cast<blasint>(problem.n), 0.0F, out,
              static_cast<blasint>(problem.n));
  openblas_set_num_threads_local(previous_threads);
  return OpenBlasExecutionStatusV1::success;
#else
  (void)problem;
  (void)lhs;
  (void)rhs;
  (void)out;
  (void)requested_threads;
  return OpenBlasExecutionStatusV1::unavailable;
#endif
}

}  // namespace matcore::mdslc::runtime
