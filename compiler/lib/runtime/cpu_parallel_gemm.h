#ifndef MATCORE_MDSLC_RUNTIME_CPU_PARALLEL_GEMM_H
#define MATCORE_MDSLC_RUNTIME_CPU_PARALLEL_GEMM_H

#include "cpu_execution_context.h"
#include "cpu_gemm_backend.h"
#include "cpu_packed_avx512.h"

#include <cstddef>
#include <cstdint>

namespace matcore::mdslc::runtime {

inline constexpr std::uint32_t kCpuParallelGemmVersionV1 = 1;
inline constexpr const char kCpuParallelPackedAvx2StableIdV1[] =
    "cpu.native-parallel.avx2-fma.f32.v1";
inline constexpr const char kCpuParallelPackedAvx512StableIdV1[] =
    "cpu.native-parallel.avx512-fma.f32.v1";

enum class CpuParallelGemmStatusV1 : std::uint32_t {
  success = 0,
  invalid_problem = 1,
  null_pointer = 2,
  invalid_pointer_alignment = 3,
  alias_violation = 4,
  arithmetic_overflow = 5,
  isa_unavailable = 6,
  workspace_misaligned = 7,
  workspace_insufficient = 8,
  invalid_thread_count = 9,
  context_unavailable = 10,
  nested_parallelism_rejected = 11,
  worker_task_failed = 12,
};

struct CpuParallelGemmWorkspaceRequirementsV1 {
  std::uint32_t version = kCpuParallelGemmVersionV1;
  std::uint32_t execution_threads = 0;
  std::size_t alignment_bytes = kCpuPackedGemmWorkspaceAlignmentV1;
  std::size_t shared_packed_b_offset = 0;
  std::size_t shared_packed_b_bytes = 0;
  std::size_t worker_region_offset = 0;
  std::size_t per_worker_bytes = 0;
  std::size_t per_worker_stride_bytes = 0;
  std::size_t total_bytes = 0;
};

struct CpuParallelGemmReportV1 {
  std::uint32_t version = kCpuParallelGemmVersionV1;
  std::uint32_t requested_threads = 0;
  std::uint32_t actual_threads = 0;
  // Internal runtime evidence: zero means B was prepared by the submitting
  // thread before dispatch; values greater than one mean the caller-owned B
  // image was prepared cooperatively by that many persistent workers.
  std::uint32_t packed_b_threads = 0;
  std::size_t macro_tile_count = 0;
  std::size_t row_task_count = 0;
  std::size_t column_task_count = 0;
  std::size_t task_count = 0;
  std::size_t workspace_bytes = 0;
  std::size_t shared_packed_b_bytes = 0;
  std::size_t per_worker_workspace_bytes = 0;
  std::uint64_t context_submission = 0;
};

CpuParallelGemmStatusV1 cpu_parallel_packed_avx2_workspace_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    std::uint32_t execution_threads,
    CpuParallelGemmWorkspaceRequirementsV1 *requirements) noexcept;

CpuParallelGemmStatusV1 cpu_parallel_packed_avx512_workspace_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    std::uint32_t execution_threads,
    CpuParallelGemmWorkspaceRequirementsV1 *requirements) noexcept;

// Executes independent rectangles from the deterministic shared M/N task
// planner. Task t is owned by worker (t % actual_threads). The complete
// workspace is a caller-owned arena containing one shared immutable packed-B
// image followed by cache-line-aligned, non-overlapping transient-A worker
// slices. B packing remains part of end-to-end execution: it is ordinarily
// completed by the submitting thread before dispatch, while a narrow
// authenticated envelope cooperatively prepares disjoint final B panels and
// publishes them before compute.
CpuParallelGemmStatusV1 cpu_execute_parallel_packed_avx2_v1(
    CpuExecutionContextV1 &context,
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes, std::uint32_t requested_threads,
    CpuProviderNestingPolicyV1 nesting_policy,
    CpuParallelGemmReportV1 *report) noexcept;

CpuParallelGemmStatusV1 cpu_execute_parallel_packed_avx512_v1(
    CpuExecutionContextV1 &context,
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes, std::uint32_t requested_threads,
    CpuProviderNestingPolicyV1 nesting_policy,
    CpuParallelGemmReportV1 *report) noexcept;

const char *cpu_parallel_gemm_status_message_v1(
    CpuParallelGemmStatusV1 status) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
