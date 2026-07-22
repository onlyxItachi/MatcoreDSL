#include "cpu_runtime_validation.h"

#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"
#include "cpu_packed_avx512.h"
#include "cpu_parallel_gemm.h"
#include "cpu_planner.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace matcore::mdslc::runtime {
namespace {

enum class SerialValidationVariantV1 : std::uint8_t {
  reference,
  tiled,
  compiler_vectorized,
  openblas,
  packed_avx2,
  packed_avx512,
};

struct SerialValidationTaskV1 {
  SerialValidationVariantV1 variant = SerialValidationVariantV1::reference;
  planner::CpuGemmProblemV1 problem;
  const float *lhs = nullptr;
  const float *rhs = nullptr;
  float *out = nullptr;
  void *workspace = nullptr;
  std::size_t workspace_bytes = 0;
  bool executed = false;
};

CpuExecutionStatusV1 execute_serial_validation_v1(
    std::size_t task_index, std::size_t worker_index,
    void *user_data) noexcept {
  if (task_index != 0 || worker_index != 0 || user_data == nullptr)
    return CpuExecutionStatusV1::invalid_configuration;
  auto &task = *static_cast<SerialValidationTaskV1 *>(user_data);
  using Request = planner::CpuGemmRequestV1;
  switch (task.variant) {
    case SerialValidationVariantV1::reference:
    case SerialValidationVariantV1::tiled:
    case SerialValidationVariantV1::compiler_vectorized: {
      const Request request =
          task.variant == SerialValidationVariantV1::reference
              ? Request::force_reference
              : task.variant == SerialValidationVariantV1::tiled
                    ? Request::force_tiled
                    : Request::force_compiler_vectorized;
      const auto plan = planner::plan_cpu_gemm_v1(
          task.problem, planner::discover_cpu_capabilities_v1(), request);
      task.executed = planner::execute_cpu_gemm_plan_v1(
          plan, task.lhs, task.rhs, task.out);
      break;
    }
    case SerialValidationVariantV1::openblas: {
      std::uint32_t actual_threads = 0;
      task.executed =
          execute_openblas_gemm_f32_v1(task.problem, task.lhs, task.rhs,
                                       task.out, 1, &actual_threads) ==
              OpenBlasExecutionStatusV1::success &&
          actual_threads == 1;
      break;
    }
    case SerialValidationVariantV1::packed_avx2:
      task.executed = cpu_execute_packed_avx2_v1(
                          task.problem, task.lhs, task.rhs, task.out,
                          task.workspace, task.workspace_bytes) ==
                      CpuPackedGemmStatusV1::success;
      break;
    case SerialValidationVariantV1::packed_avx512:
      task.executed = cpu_execute_packed_avx512_v1(
                          task.problem, task.lhs, task.rhs, task.out,
                          task.workspace, task.workspace_bytes) ==
                      CpuPackedGemmStatusV1::success;
      break;
  }
  return task.executed ? CpuExecutionStatusV1::success
                       : CpuExecutionStatusV1::callback_failed;
}

template <std::size_t M, std::size_t N, std::size_t K>
void initialize_validation_problem_v1(std::array<float, M * K> &lhs,
                                      std::array<float, K * N> &rhs,
                                      std::array<float, M * N> &out) noexcept {
  for (std::size_t row = 0; row < M; ++row) {
    for (std::size_t depth = 0; depth < K; ++depth)
      lhs[row * K + depth] =
          static_cast<float>((row + 2U * depth) % 5U + 1U);
  }
  for (std::size_t depth = 0; depth < K; ++depth) {
    for (std::size_t column = 0; column < N; ++column)
      rhs[depth * N + column] =
          static_cast<float>((3U * depth + column) % 7U + 1U);
  }
  out.fill(0.0F);
}

template <std::size_t M, std::size_t N, std::size_t K>
bool validation_output_matches_v1(const std::array<float, M * K> &lhs,
                                  const std::array<float, K * N> &rhs,
                                  const std::array<float, M * N> &out) noexcept {
  for (std::size_t row = 0; row < M; ++row) {
    for (std::size_t column = 0; column < N; ++column) {
      float expected = 0.0F;
      for (std::size_t depth = 0; depth < K; ++depth)
        expected += lhs[row * K + depth] * rhs[depth * N + column];
      if (out[row * N + column] != expected) return false;
    }
  }
  return true;
}

}  // namespace

CpuRuntimeValidationEvidenceV1 validate_cpu_runtime_variants_v1(
    CpuExecutionContextV1 &context) noexcept {
  CpuRuntimeValidationEvidenceV1 evidence;
  if (!context.info().accepting_work ||
      context.info().actual_worker_count == 0)
    return evidence;

  constexpr std::size_t m = 4;
  constexpr std::size_t n = 16;
  constexpr std::size_t k = 3;
  alignas(64) std::array<float, m * k> lhs{};
  alignas(64) std::array<float, k * n> rhs{};
  alignas(64) std::array<float, m * n> out{};
  alignas(64) std::array<std::byte, 512U * 1024U> serial_workspace{};
  initialize_validation_problem_v1<m, n, k>(lhs, rhs, out);
  const planner::CpuGemmProblemV1 problem{
      static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
      static_cast<std::int64_t>(k), planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};

  const auto validate_serial = [&](SerialValidationVariantV1 variant) noexcept {
    out.fill(0.0F);
    SerialValidationTaskV1 task{variant,
                                problem,
                                lhs.data(),
                                rhs.data(),
                                out.data(),
                                serial_workspace.data(),
                                serial_workspace.size(),
                                false};
    const auto nesting_policy =
        variant == SerialValidationVariantV1::openblas
            ? CpuProviderNestingPolicyV1::external_provider_active
            : CpuProviderNestingPolicyV1::native_only;
    const auto status = context.run_tasks(
        1, 1, nesting_policy, execute_serial_validation_v1, &task);
    return status == CpuExecutionStatusV1::success && task.executed &&
           validation_output_matches_v1<m, n, k>(lhs, rhs, out);
  };

  evidence.reference_f32_runtime_validated =
      validate_serial(SerialValidationVariantV1::reference);
  evidence.tiled_f32_runtime_validated =
      validate_serial(SerialValidationVariantV1::tiled);
  evidence.compiler_vectorized_f32_runtime_validated =
      validate_serial(SerialValidationVariantV1::compiler_vectorized);
  evidence.external_openblas_f32_runtime_validated =
      openblas_provider_info_v1().linked &&
      validate_serial(SerialValidationVariantV1::openblas);
  evidence.packed_avx2_f32_runtime_validated =
      validate_serial(SerialValidationVariantV1::packed_avx2);
  evidence.packed_avx512_f32_runtime_validated =
      validate_serial(SerialValidationVariantV1::packed_avx512);

  if (context.info().actual_worker_count < 2) return evidence;
  constexpr std::size_t parallel_m = 256;
  alignas(64) std::array<float, parallel_m * k> parallel_lhs{};
  alignas(64) std::array<float, k * n> parallel_rhs{};
  alignas(64) std::array<float, parallel_m * n> parallel_out{};
  alignas(64) std::array<std::byte, 1024U * 1024U> parallel_workspace{};
  initialize_validation_problem_v1<parallel_m, n, k>(
      parallel_lhs, parallel_rhs, parallel_out);
  const planner::CpuGemmProblemV1 parallel_problem{
      static_cast<std::int64_t>(parallel_m), static_cast<std::int64_t>(n),
      static_cast<std::int64_t>(k), planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  CpuParallelGemmReportV1 parallel_report;

  evidence.parallel_avx2_f32_runtime_validated =
      cpu_execute_parallel_packed_avx2_v1(
          context, parallel_problem, parallel_lhs.data(), parallel_rhs.data(),
          parallel_out.data(), parallel_workspace.data(),
          parallel_workspace.size(), 2,
          CpuProviderNestingPolicyV1::native_only, &parallel_report) ==
          CpuParallelGemmStatusV1::success &&
      parallel_report.actual_threads == 2 &&
      validation_output_matches_v1<parallel_m, n, k>(
          parallel_lhs, parallel_rhs, parallel_out);

  parallel_out.fill(0.0F);
  evidence.parallel_avx512_f32_runtime_validated =
      cpu_execute_parallel_packed_avx512_v1(
          context, parallel_problem, parallel_lhs.data(), parallel_rhs.data(),
          parallel_out.data(), parallel_workspace.data(),
          parallel_workspace.size(), 2,
          CpuProviderNestingPolicyV1::native_only, &parallel_report) ==
          CpuParallelGemmStatusV1::success &&
      parallel_report.actual_threads == 2 &&
      validation_output_matches_v1<parallel_m, n, k>(
          parallel_lhs, parallel_rhs, parallel_out);
  return evidence;
}

}  // namespace matcore::mdslc::runtime
