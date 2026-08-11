#include "cpu_runtime_validation.h"

#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"
#include "cpu_packed_avx512.h"
#include "cpu_parallel_gemm.h"
#include "cpu_planner.h"
#include "fp_environment_v1.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace matcore::mdslc::runtime {
namespace {

enum class ValidationVariantV1 : std::uint8_t {
  reference,
  tiled,
  compiler_vectorized,
  openblas,
  packed_avx2,
  packed_avx512,
  parallel_avx2,
  parallel_avx512,
};

class AlignedValidationWorkspaceV1 {
 public:
  explicit AlignedValidationWorkspaceV1(std::size_t bytes) noexcept
      : bytes_(bytes),
        data_(bytes == 0
                  ? nullptr
                  : static_cast<std::byte *>(::operator new(
                        bytes, std::align_val_t(kAlignment), std::nothrow))) {}

  AlignedValidationWorkspaceV1(const AlignedValidationWorkspaceV1 &) = delete;
  AlignedValidationWorkspaceV1 &operator=(
      const AlignedValidationWorkspaceV1 &) = delete;

  ~AlignedValidationWorkspaceV1() {
    if (data_ != nullptr)
      ::operator delete(data_, std::align_val_t(kAlignment));
  }

  explicit operator bool() const noexcept {
    return bytes_ == 0 || data_ != nullptr;
  }
  std::byte *data() noexcept { return data_; }
  std::size_t size() const noexcept { return bytes_; }

 private:
  static constexpr std::size_t kAlignment = 64;
  std::size_t bytes_ = 0;
  std::byte *data_ = nullptr;
};

bool variant_available_v1(ValidationVariantV1 variant) noexcept {
  switch (variant) {
    case ValidationVariantV1::reference:
    case ValidationVariantV1::tiled:
      return true;
    case ValidationVariantV1::compiler_vectorized:
      return planner::cpu_compiler_vectorization_build_available_v1();
    case ValidationVariantV1::openblas:
      return openblas_conformance_report_v1().conformant;
    case ValidationVariantV1::packed_avx2:
    case ValidationVariantV1::parallel_avx2:
      return cpu_packed_avx2_runtime_usable_v1();
    case ValidationVariantV1::packed_avx512:
    case ValidationVariantV1::parallel_avx512:
      return cpu_packed_avx512_runtime_usable_v1();
  }
  return false;
}

bool execute_serial_variant_v1(
    ValidationVariantV1 variant, const planner::CpuGemmProblemV1 &problem,
    const float *lhs, const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes) noexcept {
  using Request = planner::CpuGemmRequestV1;
  switch (variant) {
    case ValidationVariantV1::reference:
    case ValidationVariantV1::tiled:
    case ValidationVariantV1::compiler_vectorized: {
      const Request request =
          variant == ValidationVariantV1::reference
              ? Request::force_reference
              : variant == ValidationVariantV1::tiled
                    ? Request::force_tiled
                    : Request::force_compiler_vectorized;
      const auto plan = planner::plan_cpu_gemm_v1(
          problem, planner::discover_cpu_capabilities_v1(), request);
      return planner::execute_cpu_gemm_plan_v1(plan, lhs, rhs, out);
    }
    case ValidationVariantV1::openblas: {
      std::uint32_t actual_threads = 0;
      return execute_openblas_gemm_f32_v1(problem, lhs, rhs, out, 1,
                                          &actual_threads) ==
                 OpenBlasExecutionStatusV1::success &&
             actual_threads == 1;
    }
    case ValidationVariantV1::packed_avx2:
      return cpu_execute_packed_avx2_v1(problem, lhs, rhs, out, workspace,
                                        workspace_bytes) ==
             CpuPackedGemmStatusV1::success;
    case ValidationVariantV1::packed_avx512:
      return cpu_execute_packed_avx512_v1(problem, lhs, rhs, out, workspace,
                                          workspace_bytes) ==
             CpuPackedGemmStatusV1::success;
    case ValidationVariantV1::parallel_avx2:
    case ValidationVariantV1::parallel_avx512:
      return false;
  }
  return false;
}

template <std::size_t M, std::size_t N, std::size_t K>
void initialize_finite_problem_v1(std::array<float, M * K> &lhs,
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
bool finite_output_matches_v1(const std::array<float, M * K> &lhs,
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

template <std::size_t M, std::size_t N>
void initialize_special_problem_v1(std::array<float, M * 2> &lhs,
                                   std::array<float, 2 * N> &rhs,
                                   std::array<float, M * N> &out) noexcept {
  static_assert(M >= 6 && N >= 6);
  lhs.fill(0.0F);
  for (std::size_t row = 0; row < M; ++row) {
    switch (row % 6U) {
      case 0:
        lhs[row * 2] = 3.0F;
        break;
      case 1:
        lhs[row * 2] = std::numeric_limits<float>::denorm_min();
        break;
      case 2:
        lhs[row * 2] = std::numeric_limits<float>::quiet_NaN();
        break;
      case 3:
        lhs[row * 2] = std::numeric_limits<float>::infinity();
        break;
      case 4:
        lhs[row * 2] = -std::numeric_limits<float>::infinity();
        break;
      case 5:
        lhs[row * 2] = std::numeric_limits<float>::infinity();
        lhs[row * 2 + 1] = -std::numeric_limits<float>::infinity();
        break;
    }
  }
  constexpr std::array<float, 4> factors{2.0F, 1.0F, -1.0F, 4.0F};
  rhs.fill(0.0F);
  for (std::size_t column = 0; column < N; ++column) {
    rhs[column] = factors[column % factors.size()];
  }
  rhs[4] = 0.0F;
  rhs[N + 4] = 1.0F;
  rhs[5] = 1.0F;
  rhs[N + 5] = 1.0F;
  out.fill(-17.0F);
}

template <std::size_t M, std::size_t N>
void classify_special_output_v1(
    const std::array<float, M * 2> &lhs,
    const std::array<float, 2 * N> &rhs,
    const std::array<float, M * N> &out, bool *finite_correct,
    bool *subnormal_correct, bool *nan_input_propagated,
    bool *infinity_classes_preserved, bool *inf_times_zero_is_nan,
    bool *opposite_infinity_sum_is_nan) noexcept {
  *finite_correct = true;
  *subnormal_correct = true;
  *nan_input_propagated = true;
  *infinity_classes_preserved = true;
  *inf_times_zero_is_nan = true;
  *opposite_infinity_sum_is_nan = true;
  for (std::size_t row = 0; row < M; ++row) {
    for (std::size_t column = 0; column < N; ++column) {
      const float actual = out[row * N + column];
      switch (row % 6U) {
        case 0:
          *finite_correct =
              *finite_correct && actual ==
                  lhs[row * 2] * rhs[column] +
                      lhs[row * 2 + 1] * rhs[N + column];
          break;
        case 1:
          if (column != 4) {
            *subnormal_correct =
                *subnormal_correct && actual ==
                    lhs[row * 2] * rhs[column] &&
                std::fpclassify(actual) == FP_SUBNORMAL;
          }
          break;
        case 2:
          *nan_input_propagated =
              *nan_input_propagated && std::isnan(actual);
          break;
        case 3:
        case 4:
          if (column == 0) {
            *infinity_classes_preserved =
                *infinity_classes_preserved && std::isinf(actual) &&
                std::signbit(actual) ==
                    (std::signbit(lhs[row * 2]) !=
                     std::signbit(rhs[column]));
          }
          if (row % 6U == 3 && column == 4)
            *inf_times_zero_is_nan =
                *inf_times_zero_is_nan && std::isnan(actual);
          break;
        case 5:
          if (column == 5)
            *opposite_infinity_sum_is_nan =
                *opposite_infinity_sum_is_nan && std::isnan(actual);
          break;
      }
    }
  }
}

struct SerialValidationTaskV1 {
  ValidationVariantV1 variant = ValidationVariantV1::reference;
  void *workspace = nullptr;
  std::size_t workspace_bytes = 0;
  CpuVariantConformanceReportV1 report;
};

CpuExecutionStatusV1 validation_fp_preflight_v1(
    std::size_t, void *) noexcept {
  return platform::inspect_current_fp_environment_v1()
                 .explicit_gemm_f32_v1_compatible
             ? CpuExecutionStatusV1::success
             : CpuExecutionStatusV1::callback_failed;
}

CpuExecutionStatusV1 execute_serial_validation_v1(
    std::size_t task_index, std::size_t worker_index,
    void *user_data) noexcept {
  if (task_index != 0 || worker_index != 0 || user_data == nullptr)
    return CpuExecutionStatusV1::invalid_configuration;
  auto &task = *static_cast<SerialValidationTaskV1 *>(user_data);
  task.report.available = variant_available_v1(task.variant);
  if (!task.report.available) return CpuExecutionStatusV1::success;

  const auto control_before = platform::inspect_current_fp_environment_v1();
  task.report.environment_compatible =
      control_before.explicit_gemm_f32_v1_compatible;
  constexpr std::size_t finite_m = 4;
  constexpr std::size_t finite_n = 16;
  constexpr std::size_t finite_k = 3;
  alignas(64) std::array<float, finite_m * finite_k> finite_lhs{};
  alignas(64) std::array<float, finite_k * finite_n> finite_rhs{};
  alignas(64) std::array<float, finite_m * finite_n> finite_out{};
  initialize_finite_problem_v1<finite_m, finite_n, finite_k>(
      finite_lhs, finite_rhs, finite_out);
  const planner::CpuGemmProblemV1 finite_problem{
      finite_m, finite_n, finite_k, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  const bool finite_executed = execute_serial_variant_v1(
      task.variant, finite_problem, finite_lhs.data(), finite_rhs.data(),
      finite_out.data(), task.workspace, task.workspace_bytes);
  task.report.finite_correct =
      finite_executed && finite_output_matches_v1<finite_m, finite_n, finite_k>(
                             finite_lhs, finite_rhs, finite_out);

  constexpr std::size_t special_m = 6;
  constexpr std::size_t special_n = 16;
  alignas(64) std::array<float, special_m * 2> special_lhs{};
  alignas(64) std::array<float, 2 * special_n> special_rhs{};
  alignas(64) std::array<float, special_m * special_n> special_out{};
  initialize_special_problem_v1<special_m, special_n>(
      special_lhs, special_rhs, special_out);
  const planner::CpuGemmProblemV1 special_problem{
      special_m, special_n, 2, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  const bool special_executed = execute_serial_variant_v1(
      task.variant, special_problem, special_lhs.data(), special_rhs.data(),
      special_out.data(), task.workspace, task.workspace_bytes);
  bool special_finite = false;
  classify_special_output_v1<special_m, special_n>(
      special_lhs, special_rhs, special_out, &special_finite,
      &task.report.gradual_subnormal_correct,
      &task.report.nan_input_propagated,
      &task.report.infinity_classes_preserved,
      &task.report.inf_times_zero_is_nan,
      &task.report.opposite_infinity_sum_is_nan);
  task.report.nonfinite_classes_correct =
      task.report.nan_input_propagated &&
      task.report.infinity_classes_preserved &&
      task.report.inf_times_zero_is_nan &&
      task.report.opposite_infinity_sum_is_nan;
  task.report.finite_correct =
      task.report.finite_correct && special_executed && special_finite;

  const auto control_after = platform::inspect_current_fp_environment_v1();
  const bool unchanged = platform::fp_environment_control_state_equal_v1(
      control_before, control_after);
  const bool restored =
      platform::restore_fp_environment_control_state_v1(control_before);
  task.report.control_state_preserved = unchanged && restored;
  task.report.authenticated_threads = 1;
  task.report.conformant =
      task.report.available && task.report.environment_compatible &&
      task.report.finite_correct && task.report.gradual_subnormal_correct &&
      task.report.nonfinite_classes_correct &&
      task.report.control_state_preserved;
  return task.report.conformant ? CpuExecutionStatusV1::success
                                : CpuExecutionStatusV1::callback_failed;
}

CpuVariantConformanceReportV1 validate_serial_variant_v1(
    CpuExecutionContextV1 &context, ValidationVariantV1 variant,
    AlignedValidationWorkspaceV1 &workspace) noexcept {
  SerialValidationTaskV1 task;
  task.variant = variant;
  task.workspace = workspace.data();
  task.workspace_bytes = workspace.size();
  const auto nesting =
      variant == ValidationVariantV1::openblas
          ? CpuProviderNestingPolicyV1::external_provider_active
          : CpuProviderNestingPolicyV1::native_only;
  (void)context.run_tasks_with_preflight(
      1, 1, nesting, validation_fp_preflight_v1,
      execute_serial_validation_v1, &task);
  return task.report;
}

struct WorkerControlStateV1 {
  std::array<platform::FpEnvironmentReportV1, 2> before{};
  std::array<bool, 2> preserved{false, false};
};

CpuExecutionStatusV1 capture_worker_control_v1(
    std::size_t, std::size_t worker_index, void *user_data) noexcept {
  if (user_data == nullptr || worker_index >= 2)
    return CpuExecutionStatusV1::invalid_configuration;
  auto &state = *static_cast<WorkerControlStateV1 *>(user_data);
  state.before[worker_index] = platform::inspect_current_fp_environment_v1();
  return state.before[worker_index].explicit_gemm_f32_v1_compatible
             ? CpuExecutionStatusV1::success
             : CpuExecutionStatusV1::callback_failed;
}

CpuExecutionStatusV1 verify_worker_control_v1(
    std::size_t, std::size_t worker_index, void *user_data) noexcept {
  if (user_data == nullptr || worker_index >= 2)
    return CpuExecutionStatusV1::invalid_configuration;
  auto &state = *static_cast<WorkerControlStateV1 *>(user_data);
  const auto after = platform::inspect_current_fp_environment_v1();
  const bool unchanged = platform::fp_environment_control_state_equal_v1(
      state.before[worker_index], after);
  const bool restored = platform::restore_fp_environment_control_state_v1(
      state.before[worker_index]);
  state.preserved[worker_index] = unchanged && restored;
  return state.preserved[worker_index] ? CpuExecutionStatusV1::success
                                       : CpuExecutionStatusV1::callback_failed;
}

CpuVariantConformanceReportV1 validate_parallel_variant_v1(
    CpuExecutionContextV1 &context, ValidationVariantV1 variant,
    AlignedValidationWorkspaceV1 &workspace) noexcept {
  CpuVariantConformanceReportV1 report;
  report.available = context.info().actual_worker_count >= 2 &&
                     variant_available_v1(variant);
  if (!report.available) return report;

  WorkerControlStateV1 control;
  report.environment_compatible =
      context.run_tasks(2, 2, CpuProviderNestingPolicyV1::native_only,
                        capture_worker_control_v1, &control) ==
      CpuExecutionStatusV1::success;

  constexpr std::size_t finite_m = 256;
  constexpr std::size_t n = 16;
  constexpr std::size_t finite_k = 3;
  alignas(64) std::array<float, finite_m * finite_k> finite_lhs{};
  alignas(64) std::array<float, finite_k * n> finite_rhs{};
  alignas(64) std::array<float, finite_m * n> finite_out{};
  initialize_finite_problem_v1<finite_m, n, finite_k>(
      finite_lhs, finite_rhs, finite_out);
  const planner::CpuGemmProblemV1 finite_problem{
      finite_m, n, finite_k, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  CpuParallelGemmReportV1 execution_report;
  const auto finite_status =
      variant == ValidationVariantV1::parallel_avx2
          ? cpu_execute_parallel_packed_avx2_v1(
                context, finite_problem, finite_lhs.data(), finite_rhs.data(),
                finite_out.data(), workspace.data(), workspace.size(), 2,
                CpuProviderNestingPolicyV1::native_only, &execution_report)
          : cpu_execute_parallel_packed_avx512_v1(
                context, finite_problem, finite_lhs.data(), finite_rhs.data(),
                finite_out.data(), workspace.data(), workspace.size(), 2,
                CpuProviderNestingPolicyV1::native_only, &execution_report);
  report.finite_correct =
      finite_status == CpuParallelGemmStatusV1::success &&
      execution_report.actual_threads == 2 &&
      finite_output_matches_v1<finite_m, n, finite_k>(
          finite_lhs, finite_rhs, finite_out);

  constexpr std::size_t special_m = 256;
  alignas(64) std::array<float, special_m * 2> special_lhs{};
  alignas(64) std::array<float, 2 * n> special_rhs{};
  alignas(64) std::array<float, special_m * n> special_out{};
  initialize_special_problem_v1<special_m, n>(special_lhs, special_rhs,
                                               special_out);
  const planner::CpuGemmProblemV1 special_problem{
      special_m, n, 2, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  const auto special_status =
      variant == ValidationVariantV1::parallel_avx2
          ? cpu_execute_parallel_packed_avx2_v1(
                context, special_problem, special_lhs.data(),
                special_rhs.data(), special_out.data(), workspace.data(),
                workspace.size(), 2, CpuProviderNestingPolicyV1::native_only,
                &execution_report)
          : cpu_execute_parallel_packed_avx512_v1(
                context, special_problem, special_lhs.data(),
                special_rhs.data(), special_out.data(), workspace.data(),
                workspace.size(), 2, CpuProviderNestingPolicyV1::native_only,
                &execution_report);
  bool special_finite = false;
  classify_special_output_v1<special_m, n>(
      special_lhs, special_rhs, special_out, &special_finite,
      &report.gradual_subnormal_correct,
      &report.nan_input_propagated,
      &report.infinity_classes_preserved,
      &report.inf_times_zero_is_nan,
      &report.opposite_infinity_sum_is_nan);
  report.nonfinite_classes_correct =
      report.nan_input_propagated &&
      report.infinity_classes_preserved && report.inf_times_zero_is_nan &&
      report.opposite_infinity_sum_is_nan;
  report.finite_correct = report.finite_correct &&
                          special_status == CpuParallelGemmStatusV1::success &&
                          execution_report.actual_threads == 2 &&
                          special_finite;

  report.control_state_preserved =
      context.run_tasks(2, 2, CpuProviderNestingPolicyV1::native_only,
                        verify_worker_control_v1, &control) ==
          CpuExecutionStatusV1::success &&
      control.preserved[0] && control.preserved[1];
  report.authenticated_threads = 2;
  report.conformant =
      report.available && report.environment_compatible &&
      report.finite_correct && report.gradual_subnormal_correct &&
      report.nonfinite_classes_correct && report.control_state_preserved;
  return report;
}

}  // namespace

CpuRuntimeValidationEvidenceV1 validate_cpu_runtime_variants_v1(
    CpuExecutionContextV1 &context) noexcept {
  CpuRuntimeValidationEvidenceV1 evidence;
  if (!context.info().accepting_work ||
      context.info().actual_worker_count == 0)
    return evidence;

  constexpr std::size_t serial_workspace_bytes = 512U * 1024U;
  AlignedValidationWorkspaceV1 serial_workspace(serial_workspace_bytes);
  if (!serial_workspace) return evidence;
  evidence.reference_f32 = validate_serial_variant_v1(
      context, ValidationVariantV1::reference, serial_workspace);
  evidence.tiled_f32 = validate_serial_variant_v1(
      context, ValidationVariantV1::tiled, serial_workspace);
  evidence.compiler_vectorized_f32 = validate_serial_variant_v1(
      context, ValidationVariantV1::compiler_vectorized, serial_workspace);
  evidence.reference_f32_runtime_validated =
      evidence.reference_f32.conformant;
  evidence.tiled_f32_runtime_validated = evidence.tiled_f32.conformant;
  evidence.compiler_vectorized_f32_runtime_validated =
      evidence.compiler_vectorized_f32.conformant;

  const auto openblas = validate_serial_variant_v1(
      context, ValidationVariantV1::openblas, serial_workspace);
  evidence.external_openblas_f32_runtime_validated =
      openblas_conformance_report_v1().conformant && openblas.conformant;
  evidence.packed_avx2_f32 = validate_serial_variant_v1(
      context, ValidationVariantV1::packed_avx2, serial_workspace);
  evidence.packed_avx512_f32 = validate_serial_variant_v1(
      context, ValidationVariantV1::packed_avx512, serial_workspace);
  evidence.packed_avx2_f32_runtime_validated =
      evidence.packed_avx2_f32.conformant;
  evidence.packed_avx512_f32_runtime_validated =
      evidence.packed_avx512_f32.conformant;

  if (context.info().actual_worker_count < 2) return evidence;
  constexpr std::size_t parallel_workspace_bytes = 1024U * 1024U;
  AlignedValidationWorkspaceV1 parallel_workspace(parallel_workspace_bytes);
  if (!parallel_workspace) return evidence;
  evidence.parallel_avx2_f32 = validate_parallel_variant_v1(
      context, ValidationVariantV1::parallel_avx2, parallel_workspace);
  evidence.parallel_avx512_f32 = validate_parallel_variant_v1(
      context, ValidationVariantV1::parallel_avx512, parallel_workspace);
  evidence.parallel_avx2_f32_runtime_validated =
      evidence.parallel_avx2_f32.conformant;
  evidence.parallel_avx512_f32_runtime_validated =
      evidence.parallel_avx512_f32.conformant;
  return evidence;
}

}  // namespace matcore::mdslc::runtime
