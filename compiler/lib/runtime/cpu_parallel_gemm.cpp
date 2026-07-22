#include "cpu_parallel_gemm.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace matcore::mdslc::runtime {
namespace {

struct ByteSpan {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

bool checked_multiply(std::size_t lhs, std::size_t rhs,
                      std::size_t *result) noexcept {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checked_round_up(std::size_t value, std::size_t alignment,
                      std::size_t *result) noexcept {
  if (result == nullptr || alignment == 0 ||
      (alignment & (alignment - 1U)) != 0) {
    return false;
  }
  const std::size_t remainder = value & (alignment - 1U);
  if (remainder == 0) {
    *result = value;
    return true;
  }
  const std::size_t increment = alignment - remainder;
  if (increment > std::numeric_limits<std::size_t>::max() - value) return false;
  *result = value + increment;
  return true;
}

bool make_span(const void *pointer, std::size_t bytes, ByteSpan *span) noexcept {
  if (pointer == nullptr || span == nullptr) return false;
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
  span->begin = address;
  span->end = address + bytes;
  return true;
}

bool make_matrix_span(const void *pointer, std::size_t rows,
                      std::size_t columns, ByteSpan *span) noexcept {
  std::size_t elements = 0;
  std::size_t bytes = 0;
  return checked_multiply(rows, columns, &elements) &&
         checked_multiply(elements, sizeof(float), &bytes) &&
         make_span(pointer, bytes, span);
}

bool overlaps(const ByteSpan &lhs, const ByteSpan &rhs) noexcept {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool pointer_has_alignment(const void *pointer,
                           std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0 &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

CpuParallelGemmStatusV1 map_packed_status(
    CpuPackedGemmStatusV1 status) noexcept {
  switch (status) {
    case CpuPackedGemmStatusV1::success:
      return CpuParallelGemmStatusV1::success;
    case CpuPackedGemmStatusV1::invalid_problem:
      return CpuParallelGemmStatusV1::invalid_problem;
    case CpuPackedGemmStatusV1::null_pointer:
      return CpuParallelGemmStatusV1::null_pointer;
    case CpuPackedGemmStatusV1::invalid_pointer_alignment:
      return CpuParallelGemmStatusV1::invalid_pointer_alignment;
    case CpuPackedGemmStatusV1::alias_violation:
      return CpuParallelGemmStatusV1::alias_violation;
    case CpuPackedGemmStatusV1::arithmetic_overflow:
      return CpuParallelGemmStatusV1::arithmetic_overflow;
    case CpuPackedGemmStatusV1::isa_unavailable:
      return CpuParallelGemmStatusV1::isa_unavailable;
    case CpuPackedGemmStatusV1::workspace_misaligned:
      return CpuParallelGemmStatusV1::workspace_misaligned;
    case CpuPackedGemmStatusV1::workspace_insufficient:
      return CpuParallelGemmStatusV1::workspace_insufficient;
    case CpuPackedGemmStatusV1::invalid_prepacked_b:
      return CpuParallelGemmStatusV1::invalid_problem;
  }
  return CpuParallelGemmStatusV1::worker_task_failed;
}

struct ParallelGemmJobV1 {
  planner::CpuGemmProblemV1 problem;
  const float *lhs = nullptr;
  float *out = nullptr;
  CpuPackedBViewV1 packed_b;
  std::byte *workspace = nullptr;
  std::size_t worker_region_offset = 0;
  std::size_t per_worker_stride = 0;
};

CpuExecutionStatusV1 execute_row_band(std::size_t task_index,
                                      std::size_t worker_index,
                                      void *user_data) noexcept {
  auto *job = static_cast<ParallelGemmJobV1 *>(user_data);
  const std::size_t m = static_cast<std::size_t>(job->problem.m);
  const std::size_t n = static_cast<std::size_t>(job->problem.n);
  const std::size_t k = static_cast<std::size_t>(job->problem.k);
  const std::size_t row_begin = task_index * kCpuPackedGemmMcV1;
  const std::size_t rows = std::min(kCpuPackedGemmMcV1, m - row_begin);
  planner::CpuGemmProblemV1 band = job->problem;
  band.m = static_cast<std::int64_t>(rows);

  void *worker_workspace =
      job->workspace + job->worker_region_offset +
      worker_index * job->per_worker_stride;
  const auto status = cpu_execute_packed_avx2_prepacked_b_v1(
      band, job->lhs + row_begin * k, job->out + row_begin * n,
      job->packed_b, worker_workspace, job->per_worker_stride);
  return status == CpuPackedGemmStatusV1::success
             ? CpuExecutionStatusV1::success
             : CpuExecutionStatusV1::callback_failed;
}

CpuParallelGemmStatusV1 validate_tensor_and_workspace(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    const CpuParallelGemmWorkspaceRequirementsV1 &requirements,
    std::size_t supplied_workspace_bytes) noexcept {
  if (lhs == nullptr || rhs == nullptr || out == nullptr || workspace == nullptr)
    return CpuParallelGemmStatusV1::null_pointer;
  const auto alignment = static_cast<std::size_t>(problem.minimum_alignment_bytes);
  if (!pointer_has_alignment(lhs, alignof(float)) ||
      !pointer_has_alignment(rhs, alignof(float)) ||
      !pointer_has_alignment(out, alignof(float)) ||
      !pointer_has_alignment(lhs, alignment) ||
      !pointer_has_alignment(rhs, alignment) ||
      !pointer_has_alignment(out, alignment)) {
    return CpuParallelGemmStatusV1::invalid_pointer_alignment;
  }
  if (!pointer_has_alignment(workspace, requirements.alignment_bytes))
    return CpuParallelGemmStatusV1::workspace_misaligned;
  if (supplied_workspace_bytes < requirements.total_bytes)
    return CpuParallelGemmStatusV1::workspace_insufficient;

  const auto m = static_cast<std::size_t>(problem.m);
  const auto n = static_cast<std::size_t>(problem.n);
  const auto k = static_cast<std::size_t>(problem.k);
  ByteSpan lhs_span;
  ByteSpan rhs_span;
  ByteSpan out_span;
  ByteSpan workspace_span;
  if (!make_matrix_span(lhs, m, k, &lhs_span) ||
      !make_matrix_span(rhs, k, n, &rhs_span) ||
      !make_matrix_span(out, m, n, &out_span) ||
      !make_span(workspace, requirements.total_bytes, &workspace_span)) {
    return CpuParallelGemmStatusV1::arithmetic_overflow;
  }
  if (overlaps(out_span, lhs_span) || overlaps(out_span, rhs_span) ||
      overlaps(workspace_span, lhs_span) ||
      overlaps(workspace_span, rhs_span) ||
      overlaps(workspace_span, out_span)) {
    return CpuParallelGemmStatusV1::alias_violation;
  }
  return CpuParallelGemmStatusV1::success;
}

}  // namespace

CpuParallelGemmStatusV1 cpu_parallel_packed_avx2_workspace_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    std::uint32_t execution_threads,
    CpuParallelGemmWorkspaceRequirementsV1 *requirements) noexcept {
  if (requirements == nullptr) return CpuParallelGemmStatusV1::null_pointer;
  if (execution_threads == 0)
    return CpuParallelGemmStatusV1::invalid_thread_count;

  planner::CpuGemmProblemV1 band = problem;
  if (problem.m > static_cast<std::int64_t>(kCpuPackedGemmMcV1)) {
    band.m = static_cast<std::int64_t>(kCpuPackedGemmMcV1);
  }
  CpuPackedGemmWorkspaceRequirementsV1 packed_b_requirements;
  const auto packed_b_status =
      cpu_packed_avx2_prepacked_b_requirements_v1(problem,
                                                  &packed_b_requirements);
  if (packed_b_status != CpuPackedGemmStatusV1::success)
    return map_packed_status(packed_b_status);

  CpuPackedGemmWorkspaceRequirementsV1 packed_a_requirements;
  const auto status = cpu_packed_avx2_workspace_requirements_v1(
      band, CpuPackedGemmWorkspaceModeV1::transient_a_with_prepacked_b,
      &packed_a_requirements);
  if (status != CpuPackedGemmStatusV1::success) return map_packed_status(status);

  std::size_t worker_region_offset = 0;
  std::size_t worker_stride = 0;
  std::size_t worker_bytes = 0;
  std::size_t total = 0;
  if (!checked_round_up(packed_b_requirements.total_bytes,
                        kCpuPackedGemmWorkspaceAlignmentV1,
                        &worker_region_offset) ||
      !checked_round_up(packed_a_requirements.total_bytes,
                        kCpuPackedGemmWorkspaceAlignmentV1, &worker_stride) ||
      !checked_multiply(worker_stride, execution_threads, &worker_bytes) ||
      worker_bytes > std::numeric_limits<std::size_t>::max() -
                         worker_region_offset) {
    return CpuParallelGemmStatusV1::arithmetic_overflow;
  }
  total = worker_region_offset + worker_bytes;
  CpuParallelGemmWorkspaceRequirementsV1 result;
  result.execution_threads = execution_threads;
  result.shared_packed_b_bytes = packed_b_requirements.total_bytes;
  result.worker_region_offset = worker_region_offset;
  result.per_worker_bytes = packed_a_requirements.total_bytes;
  result.per_worker_stride_bytes = worker_stride;
  result.total_bytes = total;
  *requirements = result;
  return CpuParallelGemmStatusV1::success;
}

CpuParallelGemmStatusV1 cpu_execute_parallel_packed_avx2_v1(
    CpuExecutionContextV1 &context,
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes, std::uint32_t requested_threads,
    CpuProviderNestingPolicyV1 nesting_policy,
    CpuParallelGemmReportV1 *report) noexcept {
  if (report == nullptr) return CpuParallelGemmStatusV1::null_pointer;
  *report = CpuParallelGemmReportV1{};
  report->requested_threads = requested_threads;
  if (requested_threads == 0)
    return CpuParallelGemmStatusV1::invalid_thread_count;
  if (nesting_policy != CpuProviderNestingPolicyV1::native_only &&
      nesting_policy != CpuProviderNestingPolicyV1::external_provider_active) {
    return CpuParallelGemmStatusV1::invalid_problem;
  }

  CpuPackedGemmWorkspaceRequirementsV1 problem_validation;
  const auto problem_status = cpu_packed_avx2_workspace_requirements_v1(
      problem, CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
      &problem_validation);
  if (problem_status != CpuPackedGemmStatusV1::success)
    return map_packed_status(problem_status);
  if (!cpu_packed_avx2_runtime_usable_v1())
    return CpuParallelGemmStatusV1::isa_unavailable;

  const CpuExecutionContextInfoV1 context_info = context.info();
  if (!context_info.accepting_work || context_info.actual_worker_count == 0)
    return CpuParallelGemmStatusV1::context_unavailable;
  if (requested_threads > context_info.actual_worker_count)
    return CpuParallelGemmStatusV1::invalid_thread_count;
  const std::size_t m = static_cast<std::size_t>(problem.m);
  const std::size_t macro_tile_count =
      (m + kCpuPackedGemmMcV1 - 1U) / kCpuPackedGemmMcV1;
  const auto actual_threads = static_cast<std::uint32_t>(
      std::min<std::size_t>(requested_threads, macro_tile_count));
  if (actual_threads == 0)
    return CpuParallelGemmStatusV1::invalid_thread_count;
  if (nesting_policy == CpuProviderNestingPolicyV1::external_provider_active &&
      actual_threads > 1) {
    return CpuParallelGemmStatusV1::nested_parallelism_rejected;
  }

  CpuParallelGemmWorkspaceRequirementsV1 requirements;
  const auto query = cpu_parallel_packed_avx2_workspace_requirements_v1(
      problem, actual_threads, &requirements);
  if (query != CpuParallelGemmStatusV1::success) return query;
  const auto preflight = validate_tensor_and_workspace(
      problem, lhs, rhs, out, workspace, requirements, workspace_bytes);
  if (preflight != CpuParallelGemmStatusV1::success) return preflight;

  auto *workspace_start = static_cast<std::byte *>(workspace);
  CpuPackedBViewV1 packed_b;
  const auto pack_status = cpu_prepare_packed_b_avx2_v1(
      problem, rhs, workspace_start + requirements.shared_packed_b_offset,
      requirements.shared_packed_b_bytes, &packed_b);
  if (pack_status != CpuPackedGemmStatusV1::success)
    return map_packed_status(pack_status);

  ParallelGemmJobV1 job{problem,
                        lhs,
                        out,
                        packed_b,
                        workspace_start,
                        requirements.worker_region_offset,
                        requirements.per_worker_stride_bytes};
  const CpuExecutionStatusV1 execution = context.run_tasks(
      macro_tile_count, actual_threads, nesting_policy, execute_row_band, &job);
  if (execution == CpuExecutionStatusV1::nested_parallelism_rejected)
    return CpuParallelGemmStatusV1::nested_parallelism_rejected;
  if (execution != CpuExecutionStatusV1::success)
    return CpuParallelGemmStatusV1::worker_task_failed;

  const CpuExecutionContextInfoV1 finished = context.info();
  report->actual_threads = actual_threads;
  report->macro_tile_count = macro_tile_count;
  report->workspace_bytes = requirements.total_bytes;
  report->shared_packed_b_bytes = requirements.shared_packed_b_bytes;
  report->per_worker_workspace_bytes = requirements.per_worker_bytes;
  report->context_submission = finished.completed_submissions;
  return CpuParallelGemmStatusV1::success;
}

const char *cpu_parallel_gemm_status_message_v1(
    CpuParallelGemmStatusV1 status) noexcept {
  switch (status) {
    case CpuParallelGemmStatusV1::success:
      return "success";
    case CpuParallelGemmStatusV1::invalid_problem:
      return "invalid parallel GEMM problem";
    case CpuParallelGemmStatusV1::null_pointer:
      return "required parallel GEMM pointer is null";
    case CpuParallelGemmStatusV1::invalid_pointer_alignment:
      return "parallel GEMM tensor violates declared alignment";
    case CpuParallelGemmStatusV1::alias_violation:
      return "parallel GEMM buffers overlap illegally";
    case CpuParallelGemmStatusV1::arithmetic_overflow:
      return "parallel GEMM size arithmetic overflowed";
    case CpuParallelGemmStatusV1::isa_unavailable:
      return "parallel AVX2/FMA execution is unavailable";
    case CpuParallelGemmStatusV1::workspace_misaligned:
      return "parallel GEMM workspace is not 64-byte aligned";
    case CpuParallelGemmStatusV1::workspace_insufficient:
      return "parallel GEMM workspace is too small";
    case CpuParallelGemmStatusV1::invalid_thread_count:
      return "parallel GEMM thread request is invalid";
    case CpuParallelGemmStatusV1::context_unavailable:
      return "CPU execution context is unavailable";
    case CpuParallelGemmStatusV1::nested_parallelism_rejected:
      return "nested native/provider parallel execution is prohibited";
    case CpuParallelGemmStatusV1::worker_task_failed:
      return "parallel GEMM worker task failed";
  }
  return "unknown parallel GEMM status";
}

}  // namespace matcore::mdslc::runtime
