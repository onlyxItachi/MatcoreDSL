#include "cpu_parallel_gemm.h"
#include "cpu_packed_b_format.h"
#include "cpu_planner_v3.h"
#include "fp_environment_v1.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>

namespace matcore::mdslc::runtime {
namespace {

using RuntimeUsableFnV1 = bool (*)() noexcept;
using WorkspaceRequirementsFnV1 = CpuPackedGemmStatusV1 (*)(
    const planner::CpuGemmProblemV1 &, CpuPackedGemmWorkspaceModeV1,
    CpuPackedGemmWorkspaceRequirementsV1 *) noexcept;
using PrepackedRequirementsFnV1 = CpuPackedGemmStatusV1 (*)(
    const planner::CpuGemmProblemV1 &,
    CpuPackedGemmWorkspaceRequirementsV1 *) noexcept;
using PreparePackedBFnV1 = CpuPackedGemmStatusV1 (*)(
    const planner::CpuGemmProblemV1 &, const float *, void *, std::size_t,
    CpuPackedBViewV1 *) noexcept;
using ExecutePrepackedFnV1 = CpuPackedGemmStatusV1 (*)(
    const planner::CpuGemmProblemV1 &, const float *, float *,
    const CpuPackedBViewV1 &, void *, std::size_t) noexcept;
using MicrokernelFnV1 = void (*)(const float *, const float *, std::size_t,
                                 float *, std::size_t, std::uint32_t,
                                 std::uint32_t, bool) noexcept;
using FullMicrokernel16FnV1 = void (*)(const float *, const float *,
                                      std::size_t, float *, std::size_t,
                                      bool) noexcept;
using FullMicrokernel32FnV1 = void (*)(
    const float *, const float *, const float *, std::size_t, float *,
    std::size_t, bool) noexcept;

struct PackedBackendOpsV1 {
  RuntimeUsableFnV1 runtime_usable = nullptr;
  WorkspaceRequirementsFnV1 workspace_requirements = nullptr;
  PrepackedRequirementsFnV1 prepacked_requirements = nullptr;
  PreparePackedBFnV1 prepare_b = nullptr;
  ExecutePrepackedFnV1 execute_prepacked = nullptr;
  MicrokernelFnV1 microkernel = nullptr;
  FullMicrokernel16FnV1 full_microkernel_16 = nullptr;
  FullMicrokernel32FnV1 full_microkernel_32 = nullptr;
};

inline constexpr PackedBackendOpsV1 kPackedAvx2Ops{
    cpu_packed_avx2_runtime_usable_v1,
    cpu_packed_avx2_workspace_requirements_v1,
    cpu_packed_avx2_prepacked_b_requirements_v1,
    cpu_prepare_packed_b_avx2_v1,
    cpu_execute_packed_avx2_prepacked_b_v1,
    detail::matcore_cpu_packed_avx2_4x16_microkernel_f32_v1,
    detail::matcore_cpu_packed_avx2_4x16_full_microkernel_f32_v2,
    nullptr};

inline constexpr PackedBackendOpsV1 kPackedAvx512Ops{
    cpu_packed_avx512_runtime_usable_v1,
    cpu_packed_avx512_workspace_requirements_v1,
    cpu_packed_avx512_prepacked_b_requirements_v1,
    cpu_prepare_packed_b_avx512_v1,
    cpu_execute_packed_avx512_prepacked_b_v1,
    detail::matcore_cpu_packed_avx512_4x16_microkernel_f32_v1,
    nullptr,
    detail::matcore_internal_cpu_packed_avx512_4x32_full_microkernel_f32_m7};

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
  PreparePackedBFnV1 prepare_b = nullptr;
  ExecutePrepackedFnV1 execute_prepacked = nullptr;
  MicrokernelFnV1 microkernel = nullptr;
  FullMicrokernel16FnV1 full_microkernel_16 = nullptr;
  FullMicrokernel32FnV1 full_microkernel_32 = nullptr;
  std::byte *workspace = nullptr;
  std::size_t worker_region_offset = 0;
  std::size_t per_worker_stride = 0;
  std::size_t row_quantum = 0;
  std::size_t row_group_count = 0;
  std::size_t row_task_count = 0;
  std::size_t column_panel_count = 0;
  std::size_t column_task_count = 0;
  std::size_t task_count = 0;
  std::uint32_t active_threads = 0;
  bool parallel_pack_b = false;
  const float *rhs = nullptr;
  float *packed_b_destination = nullptr;
  std::atomic<std::uint32_t> *packed_b_workers_ready = nullptr;
  std::atomic<bool> *packed_b_pack_failed = nullptr;
  std::atomic<bool> *packed_b_published = nullptr;
  std::atomic<bool> *fp_environment_rejected = nullptr;
};

inline constexpr std::size_t kCpuOutputCacheLineBytesV1 =
    planner::kCpuParallelCacheLineFloatCountV1 * sizeof(float);
bool should_parallel_pack_b(
    const planner::CpuGemmProblemV1 &problem,
    const planner::CpuParallelTaskPlanV1 &task_plan,
    const CpuParallelGemmWorkspaceRequirementsV1 &requirements,
    std::uint32_t actual_threads) noexcept {
  // Milestone 7 retained only a bounded candidate/baseline diagnostic for
  // cooperative B preparation. It did not establish a final-checkpoint
  // no-regression boundary matrix for the former broad shape rule. Keep the
  // implementation dormant until a future evidence contract authenticates
  // its complete activation region.
  (void)problem;
  (void)task_plan;
  (void)requirements;
  (void)actual_threads;
  return false;
}

std::size_t divide_round_up(std::size_t numerator,
                            std::size_t denominator) noexcept {
  return numerator / denominator + (numerator % denominator != 0 ? 1U : 0U);
}

bool validate_task_plan_and_packed_b(
    const planner::CpuGemmProblemV1 &problem, const float *rhs,
    const float *out, const planner::CpuParallelTaskPlanV1 &plan,
    const CpuPackedBViewV1 &packed_b) noexcept {
  const auto n = static_cast<std::size_t>(problem.n);
  const auto k = static_cast<std::size_t>(problem.k);
  std::size_t padded_columns = 0;
  std::size_t expected_packed_elements = 0;
  std::size_t expected_packed_bytes = 0;
  const auto maximum_size =
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  const planner::CpuParallelTaskPlanV1 expected =
      planner::plan_cpu_parallel_tasks_v1(problem, plan.actual_threads);
  if (plan.row_quantum == 0 || plan.row_group_count == 0 ||
      plan.row_task_count == 0 || plan.column_panel_count == 0 ||
      plan.column_task_count == 0 || plan.actual_threads == 0 ||
      plan.macro_tile_count > maximum_size || plan.row_quantum > maximum_size ||
      plan.row_group_count > maximum_size ||
      plan.row_task_count > maximum_size ||
      plan.column_panel_count > maximum_size ||
      plan.column_task_count > maximum_size || plan.task_count > maximum_size ||
      plan.macro_tile_count != expected.macro_tile_count ||
      plan.row_quantum != expected.row_quantum ||
      plan.row_group_count != expected.row_group_count ||
      plan.row_task_count != expected.row_task_count ||
      plan.column_panel_count != expected.column_panel_count ||
      plan.column_task_count != expected.column_task_count ||
      plan.task_count != expected.task_count ||
      plan.actual_threads != expected.actual_threads) {
    return false;
  }
  if (plan.column_task_count > 1 &&
      (n % planner::kCpuParallelCacheLineFloatCountV1 != 0 ||
       problem.minimum_alignment_bytes < kCpuOutputCacheLineBytesV1 ||
       !pointer_has_alignment(out, kCpuOutputCacheLineBytesV1))) {
    return false;
  }
  if (!checked_round_up(n, kCpuPackedGemmNrV1, &padded_columns) ||
      !checked_multiply(padded_columns, k, &expected_packed_elements) ||
      !checked_multiply(expected_packed_elements, sizeof(float),
                        &expected_packed_bytes)) {
    return false;
  }
  return packed_b.source_data == rhs &&
         detail::cpu_validate_packed_b_view_metadata_v1(
             problem, packed_b, expected_packed_bytes,
             expected_packed_elements);
}

void pack_a_block(const float *lhs, std::size_t leading_dimension,
                  std::size_t row_begin, std::size_t k_begin,
                  std::size_t rows, std::size_t depth,
                  float *packed_a) noexcept {
  const std::size_t padded_rows =
      divide_round_up(rows, kCpuPackedGemmMrV1) * kCpuPackedGemmMrV1;
  std::size_t destination = 0;
  for (std::size_t row = 0; row < padded_rows;
       row += kCpuPackedGemmMrV1) {
    for (std::size_t p = 0; p < depth; ++p) {
      for (std::size_t lane = 0; lane < kCpuPackedGemmMrV1; ++lane) {
        packed_a[destination++] =
            row + lane < rows
                ? lhs[(row_begin + row + lane) * leading_dimension + k_begin +
                      p]
                : 0.0F;
      }
    }
  }
}

void compute_output_block(const ParallelGemmJobV1 &job, const float *lhs,
                          float *out, std::size_t row_begin,
                          std::size_t column_begin, std::size_t k_begin,
                          std::size_t rows, std::size_t columns,
                          std::size_t depth, float *packed_a,
                          const float *packed_b) noexcept {
  const std::size_t k = static_cast<std::size_t>(job.problem.k);
  const std::size_t n = static_cast<std::size_t>(job.problem.n);
  pack_a_block(lhs, k, row_begin, k_begin, rows, depth, packed_a);
  for (std::size_t row = 0; row < rows; row += kCpuPackedGemmMrV1) {
    const auto tile_rows = static_cast<std::uint32_t>(
        std::min(kCpuPackedGemmMrV1, rows - row));
    const float *a_panel =
        packed_a + (row / kCpuPackedGemmMrV1) * depth * kCpuPackedGemmMrV1;
    for (std::size_t column = 0; column < columns;) {
      const auto tile_columns = static_cast<std::uint32_t>(
          std::min(kCpuPackedGemmNrV1, columns - column));
      const float *b_panel =
          packed_b + (column / kCpuPackedGemmNrV1) * depth *
                         kCpuPackedGemmNrV1;
      float *output =
          out + (row_begin + row) * n + column_begin + column;
      if (tile_rows == kCpuPackedGemmMrV1 &&
          job.full_microkernel_32 != nullptr &&
          columns - column >= 2 * kCpuPackedGemmNrV1) {
        job.full_microkernel_32(
            a_panel, b_panel,
            b_panel + depth * kCpuPackedGemmNrV1, depth, output, n,
            k_begin != 0);
        column += 2 * kCpuPackedGemmNrV1;
      } else if (tile_rows == kCpuPackedGemmMrV1 &&
                 tile_columns == kCpuPackedGemmNrV1 &&
                 job.full_microkernel_16 != nullptr) {
        job.full_microkernel_16(a_panel, b_panel, depth, output, n,
                                k_begin != 0);
        column += kCpuPackedGemmNrV1;
      } else {
        job.microkernel(a_panel, b_panel, depth, output, n, tile_rows,
                        tile_columns, k_begin != 0);
        column += tile_columns;
      }
    }
  }
}

CpuExecutionStatusV1 execute_output_partition(std::size_t task_index,
                                              std::size_t worker_index,
                                              void *user_data) noexcept {
  auto *job = static_cast<ParallelGemmJobV1 *>(user_data);
  if (job == nullptr || job->execute_prepacked == nullptr ||
      job->microkernel == nullptr ||
      task_index >= job->task_count || job->row_task_count == 0 ||
      job->column_task_count == 0) {
    return CpuExecutionStatusV1::callback_failed;
  }
  const std::size_t m = static_cast<std::size_t>(job->problem.m);
  const std::size_t n = static_cast<std::size_t>(job->problem.n);
  const std::size_t k = static_cast<std::size_t>(job->problem.k);
  const std::size_t row_task_index = task_index / job->column_task_count;
  const std::size_t column_task_index = task_index % job->column_task_count;

  const std::size_t base_groups =
      job->row_group_count / job->row_task_count;
  const std::size_t extra_groups =
      job->row_group_count % job->row_task_count;
  const std::size_t group_begin =
      row_task_index * base_groups +
      std::min(row_task_index, extra_groups);
  const std::size_t groups =
      base_groups + (row_task_index < extra_groups ? 1U : 0U);
  const std::size_t row_begin = group_begin * job->row_quantum;
  const std::size_t rows =
      std::min(m - row_begin, groups * job->row_quantum);

  const std::size_t base_column_panels =
      job->column_panel_count / job->column_task_count;
  const std::size_t extra_column_panels =
      job->column_panel_count % job->column_task_count;
  const std::size_t column_panel_begin =
      column_task_index * base_column_panels +
      std::min(column_task_index, extra_column_panels);
  const std::size_t column_panels =
      base_column_panels +
      (column_task_index < extra_column_panels ? 1U : 0U);
  const std::size_t column_begin =
      column_panel_begin * kCpuPackedGemmNcV1;
  const std::size_t column_end =
      std::min(n, (column_panel_begin + column_panels) *
                      kCpuPackedGemmNcV1);

  void *worker_workspace =
      job->workspace + job->worker_region_offset +
      worker_index * job->per_worker_stride;
  if (job->column_task_count == 1) {
    planner::CpuGemmProblemV1 band = job->problem;
    band.m = static_cast<std::int64_t>(rows);
    const auto status = job->execute_prepacked(
        band, job->lhs + row_begin * k, job->out + row_begin * n,
        job->packed_b, worker_workspace, job->per_worker_stride);
    return status == CpuPackedGemmStatusV1::success
               ? CpuExecutionStatusV1::success
               : CpuExecutionStatusV1::callback_failed;
  }

  auto *packed_a = static_cast<float *>(worker_workspace);
  for (std::size_t column = column_begin; column < column_end;
       column += kCpuPackedGemmNcV1) {
    const std::size_t columns =
        std::min(kCpuPackedGemmNcV1, column_end - column);
    const std::size_t padded_columns =
        divide_round_up(columns, kCpuPackedGemmNrV1) *
        kCpuPackedGemmNrV1;
    std::size_t packed_column_offset = 0;
    std::size_t packed_column_elements = 0;
    if (!checked_multiply(column, k, &packed_column_offset) ||
        !checked_multiply(padded_columns, k, &packed_column_elements) ||
        packed_column_offset > job->packed_b.packed_elements ||
        packed_column_elements >
            job->packed_b.packed_elements - packed_column_offset) {
      return CpuExecutionStatusV1::callback_failed;
    }
    const float *packed_column_panel =
        job->packed_b.packed_data + packed_column_offset;
    std::size_t packed_k_offset = 0;
    for (std::size_t p = 0; p < k; p += kCpuPackedGemmKcV1) {
      const std::size_t depth = std::min(kCpuPackedGemmKcV1, k - p);
      std::size_t packed_k_elements = 0;
      if (!checked_multiply(padded_columns, depth, &packed_k_elements) ||
          packed_k_offset > packed_column_elements ||
          packed_k_elements > packed_column_elements - packed_k_offset) {
        return CpuExecutionStatusV1::callback_failed;
      }
      for (std::size_t row = 0; row < rows;
           row += kCpuPackedGemmMcV1) {
        const std::size_t block_rows =
            std::min(kCpuPackedGemmMcV1, rows - row);
        compute_output_block(
            *job, job->lhs + row_begin * k, job->out + row_begin * n, row,
            column, p, block_rows, columns, depth, packed_a,
            packed_column_panel + packed_k_offset);
      }
      packed_k_offset += packed_k_elements;
    }
    if (packed_k_offset != packed_column_elements)
      return CpuExecutionStatusV1::callback_failed;
  }
  return CpuExecutionStatusV1::success;
}

CpuExecutionStatusV1 authenticate_worker_fp_environment(
    std::size_t, void *user_data) noexcept {
  auto *job = static_cast<ParallelGemmJobV1 *>(user_data);
  if (job == nullptr || job->fp_environment_rejected == nullptr)
    return CpuExecutionStatusV1::invalid_configuration;
  if (platform::inspect_current_fp_environment_v1()
          .explicit_gemm_f32_v1_compatible) {
    return CpuExecutionStatusV1::success;
  }
  job->fp_environment_rejected->store(true, std::memory_order_relaxed);
  return CpuExecutionStatusV1::callback_failed;
}

CpuExecutionStatusV1 prepare_b_and_execute_output_partitions(
    std::size_t, std::size_t worker_index, void *user_data) noexcept {
  auto *job = static_cast<ParallelGemmJobV1 *>(user_data);
  if (job == nullptr || job->active_threads == 0 ||
      worker_index >= job->active_threads || job->rhs == nullptr ||
      job->packed_b_destination == nullptr ||
      job->packed_b_workers_ready == nullptr ||
      job->packed_b_pack_failed == nullptr ||
      job->packed_b_published == nullptr || job->prepare_b == nullptr) {
    return CpuExecutionStatusV1::callback_failed;
  }

  bool local_pack_failed = false;
  if (job->parallel_pack_b) {
    const std::size_t n = static_cast<std::size_t>(job->problem.n);
    const std::size_t k = static_cast<std::size_t>(job->problem.k);
    for (std::size_t panel = worker_index; panel < job->column_panel_count;
         panel += job->active_threads) {
      const std::size_t column_begin = panel * kCpuPackedGemmNcV1;
      const std::size_t columns =
          std::min(kCpuPackedGemmNcV1, n - column_begin);
      std::size_t expected_panel_elements = 0;
      if (!checked_multiply(
              divide_round_up(columns, kCpuPackedGemmNrV1) *
                  kCpuPackedGemmNrV1,
              k, &expected_panel_elements) ||
          detail::cpu_pack_b_column_panel_v1(
              job->rhs, n, k, column_begin, columns,
              job->packed_b_destination + column_begin * k) !=
              expected_panel_elements) {
        local_pack_failed = true;
        break;
      }
    }
  } else if (worker_index == 0) {
    CpuPackedBViewV1 prepared;
    if (job->prepare_b(job->problem, job->rhs, job->packed_b_destination,
                       job->packed_b.storage_bytes, &prepared) !=
        CpuPackedGemmStatusV1::success) {
      local_pack_failed = true;
    } else {
      job->packed_b = prepared;
      job->packed_b_published->store(true, std::memory_order_release);
    }
  }

  if (local_pack_failed)
    job->packed_b_pack_failed->store(true, std::memory_order_release);
  job->packed_b_workers_ready->fetch_add(1, std::memory_order_release);
  while (!job->packed_b_pack_failed->load(std::memory_order_acquire) &&
         (job->parallel_pack_b
              ? job->packed_b_workers_ready->load(std::memory_order_acquire) !=
                    job->active_threads
              : !job->packed_b_published->load(std::memory_order_acquire))) {
    std::this_thread::yield();
  }
  if (job->packed_b_pack_failed->load(std::memory_order_acquire))
    return CpuExecutionStatusV1::callback_failed;
  if (!validate_task_plan_and_packed_b(job->problem, job->rhs, job->out,
                                       planner::plan_cpu_parallel_tasks_v1(
                                           job->problem,
                                           job->active_threads),
                                       job->packed_b)) {
    return CpuExecutionStatusV1::callback_failed;
  }

  for (std::size_t task = worker_index; task < job->task_count;
       task += job->active_threads) {
    const CpuExecutionStatusV1 status =
        execute_output_partition(task, worker_index, job);
    if (status != CpuExecutionStatusV1::success) return status;
  }
  return CpuExecutionStatusV1::success;
}

CpuParallelGemmStatusV1 workspace_requirements(
    const PackedBackendOpsV1 &ops,
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
      ops.prepacked_requirements(problem, &packed_b_requirements);
  if (packed_b_status != CpuPackedGemmStatusV1::success)
    return map_packed_status(packed_b_status);

  CpuPackedGemmWorkspaceRequirementsV1 packed_a_requirements;
  const auto status = ops.workspace_requirements(
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

CpuParallelGemmStatusV1 validate_tensor_and_workspace(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    const CpuParallelGemmWorkspaceRequirementsV1 &requirements,
    std::size_t supplied_workspace_bytes) noexcept;

CpuParallelGemmStatusV1 execute_parallel(
    const PackedBackendOpsV1 &ops, CpuExecutionContextV1 &context,
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
  const auto problem_status = ops.workspace_requirements(
      problem, CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
      &problem_validation);
  if (problem_status != CpuPackedGemmStatusV1::success)
    return map_packed_status(problem_status);
  if (!ops.runtime_usable()) return CpuParallelGemmStatusV1::isa_unavailable;

  const CpuExecutionContextInfoV1 context_info = context.info();
  if (!context_info.accepting_work || context_info.actual_worker_count == 0)
    return CpuParallelGemmStatusV1::context_unavailable;
  if (requested_threads > context_info.actual_worker_count)
    return CpuParallelGemmStatusV1::invalid_thread_count;
  const planner::CpuParallelTaskPlanV1 task_plan =
      planner::plan_cpu_parallel_tasks_v1(problem, requested_threads);
  const std::uint32_t actual_threads = task_plan.actual_threads;
  if (actual_threads == 0)
    return CpuParallelGemmStatusV1::invalid_thread_count;
  if (nesting_policy == CpuProviderNestingPolicyV1::external_provider_active &&
      actual_threads > 1) {
    return CpuParallelGemmStatusV1::nested_parallelism_rejected;
  }

  CpuParallelGemmWorkspaceRequirementsV1 requirements;
  const auto query = workspace_requirements(ops, problem, actual_threads,
                                            &requirements);
  if (query != CpuParallelGemmStatusV1::success) return query;
  const auto preflight = validate_tensor_and_workspace(
      problem, lhs, rhs, out, workspace, requirements, workspace_bytes);
  if (preflight != CpuParallelGemmStatusV1::success) return preflight;

  auto *workspace_start = static_cast<std::byte *>(workspace);
  auto *packed_b_destination = reinterpret_cast<float *>(
      workspace_start + requirements.shared_packed_b_offset);
  const bool parallel_pack_b = should_parallel_pack_b(
      problem, task_plan, requirements, actual_threads);
  CpuPackedBViewV1 packed_b = detail::cpu_make_packed_b_view_v1(
      problem, rhs, packed_b_destination,
      requirements.shared_packed_b_bytes,
      requirements.shared_packed_b_bytes / sizeof(float));
  if (parallel_pack_b) {
    // Cooperative packing uses the deterministic view prepared above. The
    // current production policy keeps this branch dormant pending evidence.
  } else {
    // Worker zero publishes the fully authenticated view after every active
    // worker passes the floating-point preflight barrier.
    packed_b.packed_data = packed_b_destination;
  }

  std::atomic<std::uint32_t> packed_b_workers_ready{0};
  std::atomic<bool> packed_b_pack_failed{false};
  std::atomic<bool> packed_b_published{parallel_pack_b};
  std::atomic<bool> fp_environment_rejected{false};
  ParallelGemmJobV1 job;
  job.problem = problem;
  job.lhs = lhs;
  job.out = out;
  job.packed_b = packed_b;
  job.prepare_b = ops.prepare_b;
  job.execute_prepacked = ops.execute_prepacked;
  job.microkernel = ops.microkernel;
  job.full_microkernel_16 = ops.full_microkernel_16;
  job.full_microkernel_32 = ops.full_microkernel_32;
  job.workspace = workspace_start;
  job.worker_region_offset = requirements.worker_region_offset;
  job.per_worker_stride = requirements.per_worker_stride_bytes;
  job.row_quantum = static_cast<std::size_t>(task_plan.row_quantum);
  job.row_group_count = static_cast<std::size_t>(task_plan.row_group_count);
  job.row_task_count = static_cast<std::size_t>(task_plan.row_task_count);
  job.column_panel_count =
      static_cast<std::size_t>(task_plan.column_panel_count);
  job.column_task_count =
      static_cast<std::size_t>(task_plan.column_task_count);
  job.task_count = static_cast<std::size_t>(task_plan.task_count);
  job.active_threads = actual_threads;
  job.parallel_pack_b = parallel_pack_b;
  job.rhs = rhs;
  job.packed_b_destination = packed_b_destination;
  job.packed_b_workers_ready = &packed_b_workers_ready;
  job.packed_b_pack_failed = &packed_b_pack_failed;
  job.packed_b_published = &packed_b_published;
  job.fp_environment_rejected = &fp_environment_rejected;
  const CpuExecutionStatusV1 execution = context.run_tasks_with_preflight(
      static_cast<std::size_t>(actual_threads), actual_threads,
      nesting_policy, authenticate_worker_fp_environment,
      prepare_b_and_execute_output_partitions, &job);
  if (fp_environment_rejected.load(std::memory_order_relaxed))
    return CpuParallelGemmStatusV1::unsupported_fp_environment;
  if (execution == CpuExecutionStatusV1::nested_parallelism_rejected)
    return CpuParallelGemmStatusV1::nested_parallelism_rejected;
  if (execution != CpuExecutionStatusV1::success)
    return CpuParallelGemmStatusV1::worker_task_failed;

  const CpuExecutionContextInfoV1 finished = context.info();
  report->actual_threads = actual_threads;
  report->packed_b_threads = parallel_pack_b ? actual_threads : 0;
  report->macro_tile_count =
      static_cast<std::size_t>(task_plan.macro_tile_count);
  report->row_task_count =
      static_cast<std::size_t>(task_plan.row_task_count);
  report->column_task_count =
      static_cast<std::size_t>(task_plan.column_task_count);
  report->task_count = static_cast<std::size_t>(task_plan.task_count);
  report->workspace_bytes = requirements.total_bytes;
  report->shared_packed_b_bytes = requirements.shared_packed_b_bytes;
  report->per_worker_workspace_bytes = requirements.per_worker_bytes;
  report->context_submission = finished.completed_submissions;
  return CpuParallelGemmStatusV1::success;
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
  return workspace_requirements(kPackedAvx2Ops, problem, execution_threads,
                                requirements);
}

CpuParallelGemmStatusV1 cpu_parallel_packed_avx512_workspace_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    std::uint32_t execution_threads,
    CpuParallelGemmWorkspaceRequirementsV1 *requirements) noexcept {
  return workspace_requirements(kPackedAvx512Ops, problem, execution_threads,
                                requirements);
}

CpuParallelGemmStatusV1 cpu_execute_parallel_packed_avx2_v1(
    CpuExecutionContextV1 &context,
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes, std::uint32_t requested_threads,
    CpuProviderNestingPolicyV1 nesting_policy,
    CpuParallelGemmReportV1 *report) noexcept {
  return execute_parallel(kPackedAvx2Ops, context, problem, lhs, rhs, out,
                          workspace, workspace_bytes, requested_threads,
                          nesting_policy, report);
}

CpuParallelGemmStatusV1 cpu_execute_parallel_packed_avx512_v1(
    CpuExecutionContextV1 &context,
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes, std::uint32_t requested_threads,
    CpuProviderNestingPolicyV1 nesting_policy,
    CpuParallelGemmReportV1 *report) noexcept {
  return execute_parallel(kPackedAvx512Ops, context, problem, lhs, rhs, out,
                          workspace, workspace_bytes, requested_threads,
                          nesting_policy, report);
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
      return "requested parallel packed ISA execution is unavailable";
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
    case CpuParallelGemmStatusV1::unsupported_fp_environment:
      return "parallel GEMM worker floating-point environment is unsupported";
  }
  return "unknown parallel GEMM status";
}

}  // namespace matcore::mdslc::runtime
