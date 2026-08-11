#include "cpu_openblas.h"

#include "fp_environment_v1.h"

#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>

#if MATCORE_MDSLC_HAS_OPENBLAS
#include <cblas.h>
#endif

#ifndef MATCORE_MDSLC_OPENBLAS_VERSION
#define MATCORE_MDSLC_OPENBLAS_VERSION "unavailable"
#endif

namespace matcore::mdslc::runtime {
namespace {

std::uint64_t hash_identity_component(std::uint64_t hash,
                                      const char *text) noexcept {
  constexpr std::uint64_t kPrime = UINT64_C(1099511628211);
  if (text == nullptr) return 0;
  for (const unsigned char *cursor =
           reinterpret_cast<const unsigned char *>(text);
       *cursor != 0; ++cursor) {
    hash ^= *cursor;
    hash *= kPrime;
  }
  hash ^= UINT64_C(0xFF);
  return hash * kPrime;
}

std::uint64_t provider_identity_key(
    const OpenBlasProviderInfoV1 &provider) noexcept {
  if (!provider.linked || provider.package_version == nullptr ||
      provider.runtime_config == nullptr || provider.runtime_core == nullptr)
    return 0;
  std::uint64_t hash = UINT64_C(14695981039346656037);
  hash = hash_identity_component(hash, provider.package_version);
  hash = hash_identity_component(hash, provider.runtime_config);
  hash = hash_identity_component(hash, provider.runtime_core);
  hash ^= static_cast<std::uint32_t>(provider.parallel_model);
  hash *= UINT64_C(1099511628211);
  return hash;
}

bool complete_identity(const OpenBlasProviderInfoV1 &provider) noexcept {
  return provider_identity_key(provider) != 0 &&
         std::strcmp(provider.package_version, "unavailable") != 0 &&
         std::strcmp(provider.runtime_config, "unavailable") != 0 &&
         std::strcmp(provider.runtime_core, "unavailable") != 0 &&
         std::strcmp(provider.runtime_config, "unknown") != 0 &&
         std::strcmp(provider.runtime_core, "unknown") != 0;
}

#if MATCORE_MDSLC_HAS_OPENBLAS
OpenBlasConformanceReportV1 run_openblas_conformance_probe_v1(
    const OpenBlasProviderInfoV1 &provider) noexcept {
  OpenBlasConformanceReportV1 report;
  report.provider_linked = provider.linked;
  report.provider_identity_key = provider_identity_key(provider);
  report.identity_complete = complete_identity(provider);
  report.probe_attempted = true;
  const auto control_before =
      platform::inspect_current_fp_environment_v1();
  report.caller_environment_compatible =
      control_before.explicit_gemm_f32_v1_compatible;
  if (!report.provider_linked || !report.identity_complete ||
      !report.caller_environment_compatible) {
    return report;
  }

  constexpr std::size_t m = 6;
  constexpr std::size_t n = 6;
  constexpr std::size_t k = 2;
  alignas(64) std::array<float, m * k> lhs{
      3.0F, 0.0F,
      std::numeric_limits<float>::denorm_min(), 0.0F,
      std::numeric_limits<float>::quiet_NaN(), 0.0F,
      std::numeric_limits<float>::infinity(), 0.0F,
      -std::numeric_limits<float>::infinity(), 0.0F,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity()};
  // Columns 4 and 5 are the two reduction-sensitive fixtures:
  // Inf*0 -> NaN and (+Inf)+(-Inf) -> NaN respectively.
  alignas(64) std::array<float, k * n> rhs{
      2.0F, 1.0F, -1.0F, 4.0F, 0.0F, 1.0F,
      0.0F, 0.0F,  0.0F, 0.0F, 1.0F, 1.0F};
  alignas(64) std::array<float, m * n> out{};
  out.fill(-17.0F);

  const int threads_before = openblas_get_num_threads();
  const int previous_threads = openblas_set_num_threads_local(1);
  const bool configured_single_thread = openblas_get_num_threads() == 1;
  if (configured_single_thread) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<blasint>(m), static_cast<blasint>(n),
                static_cast<blasint>(k), 1.0F, lhs.data(),
                static_cast<blasint>(k), rhs.data(),
                static_cast<blasint>(n), 0.0F, out.data(),
                static_cast<blasint>(n));
  }
  openblas_set_num_threads_local(previous_threads);
  const int threads_after = openblas_get_num_threads();
  const auto control_after = platform::inspect_current_fp_environment_v1();
  const bool unchanged = platform::fp_environment_control_state_equal_v1(
      control_before, control_after);
  const bool restored =
      platform::restore_fp_environment_control_state_v1(control_before);

  report.thread_state_restored =
      configured_single_thread && threads_before > 0 &&
      threads_after == threads_before;
  report.control_state_preserved = unchanged && restored;
  report.finite_correct = out[0] == 6.0F && out[1] == 3.0F &&
                          out[2] == -3.0F && out[3] == 12.0F &&
                          out[4] == 0.0F && out[5] == 3.0F;
  report.gradual_subnormal_correct =
      out[n] == std::numeric_limits<float>::denorm_min() * 2.0F &&
      out[n + 1] == std::numeric_limits<float>::denorm_min() &&
      out[n + 2] == -std::numeric_limits<float>::denorm_min() &&
      out[n + 3] == std::numeric_limits<float>::denorm_min() * 4.0F &&
      out[n + 5] == std::numeric_limits<float>::denorm_min();
  report.nan_input_propagated = true;
  for (std::size_t column = 0; column < n; ++column) {
    report.nan_input_propagated =
        report.nan_input_propagated && std::isnan(out[2 * n + column]);
  }
  report.infinity_classes_preserved =
      std::isinf(out[3 * n]) &&
      !std::signbit(out[3 * n]) && std::isinf(out[4 * n]) &&
      std::signbit(out[4 * n]);
  report.inf_times_zero_is_nan = std::isnan(out[3 * n + 4]);
  report.opposite_infinity_sum_is_nan = std::isnan(out[5 * n + 5]);
  report.nonfinite_classes_correct =
      report.nan_input_propagated && report.infinity_classes_preserved &&
      report.inf_times_zero_is_nan &&
      report.opposite_infinity_sum_is_nan;
  report.conformant =
      report.provider_linked && report.identity_complete &&
      report.caller_environment_compatible && report.finite_correct &&
      report.gradual_subnormal_correct &&
      report.nonfinite_classes_correct && report.control_state_preserved &&
      report.thread_state_restored;
  return report;
}
#endif

}  // namespace

OpenBlasProviderInfoV1 openblas_provider_info_v1() noexcept {
#if MATCORE_MDSLC_HAS_OPENBLAS
  // OpenBLAS may derive openblas_get_num_procs() from the calling thread's
  // affinity during first compute initialization.  Exact context validation
  // deliberately runs SGEMM on a bound worker, so repeatedly querying that
  // value would incorrectly collapse the provider ceiling to one CPU. Freeze
  // provider identity and its process-level ceiling on the first caller; the
  // planner still applies the stricter topology/context thread ceiling.
  static const OpenBlasProviderInfoV1 provider = []() noexcept {
    const char *config = openblas_get_config();
    const char *core = openblas_get_corename();
    return OpenBlasProviderInfoV1{
        true,
        MATCORE_MDSLC_OPENBLAS_VERSION,
        config != nullptr ? config : "unknown",
        core != nullptr ? core : "unknown",
        openblas_get_parallel(),
        openblas_get_num_procs()};
  }();
  return provider;
#else
  return {};
#endif
}

OpenBlasConformanceReportV1 openblas_conformance_report_v1() noexcept {
  const OpenBlasProviderInfoV1 provider = openblas_provider_info_v1();
#if MATCORE_MDSLC_HAS_OPENBLAS
  static std::once_flag once;
  static OpenBlasConformanceReportV1 report;
  static std::atomic<bool> complete{false};
  if (!complete.load(std::memory_order_acquire) &&
      !platform::inspect_current_fp_environment_v1()
           .explicit_gemm_f32_v1_compatible) {
    OpenBlasConformanceReportV1 deferred;
    deferred.provider_linked = provider.linked;
    deferred.provider_identity_key = provider_identity_key(provider);
    deferred.identity_complete = complete_identity(provider);
    return deferred;
  }
  std::call_once(once, [&]() noexcept {
    report = run_openblas_conformance_probe_v1(provider);
    complete.store(true, std::memory_order_release);
  });
  return report;
#else
  OpenBlasConformanceReportV1 report;
  report.provider_linked = provider.linked;
  return report;
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
  const int maximum_threads =
      openblas_provider_info_v1().maximum_reported_threads;
  if (maximum_threads <= 0 ||
      requested_threads > static_cast<std::uint32_t>(maximum_threads))
    return OpenBlasExecutionStatusV1::invalid_thread_count;
  // Only the calling thread's floating-point environment is observable.
  // OpenBLAS worker state is provider-private, so multi-thread execution
  // cannot satisfy the explicit-gemm-f32-v1 proof obligation yet.
  if (requested_threads != 1)
    return OpenBlasExecutionStatusV1::unsupported_fp_environment;
  const auto control_before = platform::inspect_current_fp_environment_v1();
  if (!control_before.explicit_gemm_f32_v1_compatible)
    return OpenBlasExecutionStatusV1::unsupported_fp_environment;
  const auto conformance = openblas_conformance_report_v1();
  if (!conformance.conformant ||
      conformance.provider_identity_key !=
          provider_identity_key(openblas_provider_info_v1())) {
    return OpenBlasExecutionStatusV1::provider_conformance_failed;
  }

  const int threads_before = openblas_get_num_threads();
  const int previous_threads =
      openblas_set_num_threads_local(static_cast<int>(requested_threads));
  const int provider_threads = openblas_get_num_threads();
  if (provider_threads <= 0 ||
      provider_threads != static_cast<int>(requested_threads)) {
    openblas_set_num_threads_local(previous_threads);
    const int threads_after = openblas_get_num_threads();
    const auto control_after = platform::inspect_current_fp_environment_v1();
    const bool control_preserved =
        platform::fp_environment_control_state_equal_v1(control_before,
                                                         control_after);
    const bool control_restored =
        platform::restore_fp_environment_control_state_v1(control_before);
    return control_preserved && control_restored && threads_before > 0 &&
                   threads_after == threads_before
               ? OpenBlasExecutionStatusV1::invalid_thread_count
               : OpenBlasExecutionStatusV1::provider_state_violation;
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
  const int threads_after = openblas_get_num_threads();
  const auto control_after = platform::inspect_current_fp_environment_v1();
  const bool control_preserved =
      platform::fp_environment_control_state_equal_v1(control_before,
                                                       control_after);
  const bool control_restored =
      platform::restore_fp_environment_control_state_v1(control_before);
  if (!control_preserved || !control_restored ||
      threads_before <= 0 || threads_after != threads_before) {
    if (actual_threads != nullptr) *actual_threads = 0;
    return OpenBlasExecutionStatusV1::provider_state_violation;
  }
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
