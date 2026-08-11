#include "cpu_openblas.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>

#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {

bool close(float actual, float expected) {
  return std::fabs(static_cast<double>(actual) - expected) <= 1.0e-5;
}

#if defined(__linux__) && defined(__x86_64__)
class ScopedMxcsr {
 public:
  ScopedMxcsr() noexcept : saved_(_mm_getcsr()) {}
  ScopedMxcsr(const ScopedMxcsr &) = delete;
  ScopedMxcsr &operator=(const ScopedMxcsr &) = delete;
  ~ScopedMxcsr() { _mm_setcsr(saved_); }

  void enable_flush_to_zero() noexcept {
    _mm_setcsr(saved_ | (1U << 15U));
  }

 private:
  std::uint32_t saved_ = 0;
};
#endif

}  // namespace

int main() {
  namespace planner = matcore::mdslc::planner;
  namespace runtime = matcore::mdslc::runtime;

  const runtime::OpenBlasProviderInfoV1 provider =
      runtime::openblas_provider_info_v1();
#if MATCORE_MDSLC_HAS_OPENBLAS
  if (!provider.linked || provider.package_version == nullptr ||
      provider.runtime_config == nullptr || provider.runtime_core == nullptr ||
      provider.maximum_reported_threads <= 0) {
    std::cerr << "linked OpenBLAS provider metadata is incomplete\n";
    return 1;
  }
  const planner::CpuGemmProblemV1 problem{
      2, 2, 2, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, alignof(float)};
  const std::array<float, 4> lhs{1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> rhs{5.0F, 6.0F, 7.0F, 8.0F};
  std::array<float, 4> out{-9.0F, -9.0F, -9.0F, -9.0F};
  std::uint32_t actual_threads = 0;
  if (runtime::execute_openblas_gemm_f32_v1(
          problem, lhs.data(), rhs.data(), out.data(), 1,
          &actual_threads) != runtime::OpenBlasExecutionStatusV1::success ||
      actual_threads != 1 || !close(out[0], 19.0F) ||
      !close(out[1], 22.0F) || !close(out[2], 43.0F) ||
      !close(out[3], 50.0F)) {
    std::cerr << "single-thread row-major OpenBLAS SGEMM failed\n";
    return 1;
  }

  out.fill(-8.0F);
  const auto two_thread_status = runtime::execute_openblas_gemm_f32_v1(
      problem, lhs.data(), rhs.data(), out.data(), 2, &actual_threads);
  const auto expected_two_thread_status =
      provider.maximum_reported_threads >= 2
          ? runtime::OpenBlasExecutionStatusV1::unsupported_fp_environment
          : runtime::OpenBlasExecutionStatusV1::invalid_thread_count;
  if (two_thread_status != expected_two_thread_status ||
      actual_threads != 0 ||
      out != std::array<float, 4>{-8.0F, -8.0F, -8.0F, -8.0F}) {
    std::cerr << "unauthenticated two-thread OpenBLAS execution was not rejected\n";
    return 1;
  }

#if defined(__linux__) && defined(__x86_64__)
  out.fill(-4.0F);
  {
    ScopedMxcsr scope;
    scope.enable_flush_to_zero();
    if (runtime::execute_openblas_gemm_f32_v1(
            problem, lhs.data(), rhs.data(), out.data(), 1,
            &actual_threads) !=
            runtime::OpenBlasExecutionStatusV1::unsupported_fp_environment ||
        actual_threads != 0 ||
        out != std::array<float, 4>{-4.0F, -4.0F, -4.0F, -4.0F}) {
      std::cerr << "OpenBLAS accepted an unsupported caller FP environment\n";
      return 1;
    }
  }
#endif

  out.fill(-6.0F);
  if (runtime::execute_openblas_gemm_f32_v1(
          problem, lhs.data(), rhs.data(), out.data(), 1,
          &actual_threads) != runtime::OpenBlasExecutionStatusV1::success ||
      actual_threads != 1 || !close(out[0], 19.0F) ||
      !close(out[1], 22.0F) || !close(out[2], 43.0F) ||
      !close(out[3], 50.0F)) {
    std::cerr << "OpenBLAS local thread policy was not reusable\n";
    return 1;
  }

  out.fill(-7.0F);
  if (runtime::execute_openblas_gemm_f32_v1(
          problem, lhs.data(), rhs.data(), out.data(), 0,
          &actual_threads) !=
          runtime::OpenBlasExecutionStatusV1::invalid_thread_count ||
      out != std::array<float, 4>{-7.0F, -7.0F, -7.0F, -7.0F}) {
    std::cerr << "invalid thread request mutated output or was accepted\n";
    return 1;
  }

  out.fill(-5.0F);
  actual_threads = 99;
  const auto excessive_threads =
      static_cast<std::uint32_t>(provider.maximum_reported_threads) + 1U;
  if (runtime::execute_openblas_gemm_f32_v1(
          problem, lhs.data(), rhs.data(), out.data(), excessive_threads,
          &actual_threads) !=
          runtime::OpenBlasExecutionStatusV1::invalid_thread_count ||
      actual_threads != 0 ||
      out != std::array<float, 4>{-5.0F, -5.0F, -5.0F, -5.0F}) {
    std::cerr << "provider-excessive thread request mutated output or executed\n";
    return 1;
  }
#else
  if (provider.linked) {
    std::cerr << "unlinked build advertised OpenBLAS\n";
    return 1;
  }
#endif
  return 0;
}
