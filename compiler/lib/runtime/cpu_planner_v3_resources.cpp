#include "cpu_planner_v3_resources.h"

#include "cpu_packed_avx512.h"

#include <limits>

namespace matcore::mdslc::runtime {
namespace {

std::uint64_t size_to_u64(std::size_t value, bool *valid) noexcept {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
      if (valid != nullptr) *valid = false;
      return 0;
    }
  }
  return static_cast<std::uint64_t>(value);
}

void populate_parallel_workspace(
    const CpuParallelGemmWorkspaceRequirementsV1 &requirements,
    bool query_succeeded, bool *valid, std::uint64_t *shared_bytes,
    std::uint64_t *per_worker_bytes, std::uint32_t *alignment) noexcept {
  if (valid == nullptr || shared_bytes == nullptr ||
      per_worker_bytes == nullptr || alignment == nullptr) {
    return;
  }
  *valid = query_succeeded;
  if (!query_succeeded) return;
  *shared_bytes = size_to_u64(requirements.worker_region_offset, valid);
  *per_worker_bytes = size_to_u64(requirements.per_worker_stride_bytes, valid);
  if (requirements.alignment_bytes >
      std::numeric_limits<std::uint32_t>::max()) {
    *valid = false;
    return;
  }
  *alignment = static_cast<std::uint32_t>(requirements.alignment_bytes);
}

}  // namespace

planner::CpuGemmImplementationResourcesV2
augment_cpu_gemm_implementation_resources_v2(
    const planner::CpuGemmProblemV1 &problem,
    const planner::CpuGemmImplementationResourcesV1 &baseline,
    const CpuExecutionContextV1 *execution_context,
    const CpuRuntimeValidationEvidenceV1 &validation_evidence) noexcept {
  planner::CpuGemmImplementationResourcesV2 result;
  result.baseline = baseline;
  if (execution_context != nullptr) {
    const CpuExecutionContextInfoV1 info = execution_context->info();
    result.execution_context_available = info.accepting_work;
    result.execution_context_worker_capacity = info.actual_worker_count;
  }

  result.native_parallel_avx2_fma_compiled =
      cpu_packed_avx2_build_available_v1();
  CpuParallelGemmWorkspaceRequirementsV1 avx2_workspace;
  const bool avx2_workspace_valid =
      cpu_parallel_packed_avx2_workspace_requirements_v1(
          problem, 1, &avx2_workspace) == CpuParallelGemmStatusV1::success;
  populate_parallel_workspace(
      avx2_workspace, avx2_workspace_valid,
      &result.native_parallel_avx2_workspace_size_valid,
      &result.native_parallel_avx2_shared_workspace_bytes,
      &result.native_parallel_avx2_per_worker_workspace_bytes,
      &result.native_parallel_avx2_workspace_alignment);

  result.native_packed_avx512_fma_compiled =
      cpu_packed_avx512_build_available_v1();
  result.native_packed_avx512_fma_runtime_validated =
      validation_evidence.version == kCpuRuntimeValidationEvidenceVersionV1 &&
      validation_evidence.packed_avx512_f32_runtime_validated &&
      cpu_packed_avx512_runtime_usable_v1();
  CpuPackedGemmWorkspaceRequirementsV1 avx512_single;
  const auto avx512_single_status =
      cpu_packed_avx512_workspace_requirements_v1(
          problem, CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
          &avx512_single);
  result.native_packed_avx512_workspace_size_valid =
      avx512_single_status == CpuPackedGemmStatusV1::success;
  if (result.native_packed_avx512_workspace_size_valid) {
    result.native_packed_avx512_workspace_bytes =
        size_to_u64(avx512_single.total_bytes,
                    &result.native_packed_avx512_workspace_size_valid);
    if (avx512_single.alignment_bytes >
        std::numeric_limits<std::uint32_t>::max()) {
      result.native_packed_avx512_workspace_size_valid = false;
    } else {
      result.native_packed_avx512_workspace_alignment =
          static_cast<std::uint32_t>(avx512_single.alignment_bytes);
    }
  }

  result.native_parallel_avx512_fma_compiled =
      cpu_packed_avx512_build_available_v1();
  CpuParallelGemmWorkspaceRequirementsV1 avx512_workspace;
  const bool avx512_workspace_valid =
      cpu_parallel_packed_avx512_workspace_requirements_v1(
          problem, 1, &avx512_workspace) == CpuParallelGemmStatusV1::success;
  populate_parallel_workspace(
      avx512_workspace, avx512_workspace_valid,
      &result.native_parallel_avx512_workspace_size_valid,
      &result.native_parallel_avx512_shared_workspace_bytes,
      &result.native_parallel_avx512_per_worker_workspace_bytes,
      &result.native_parallel_avx512_workspace_alignment);
  return result;
}

}  // namespace matcore::mdslc::runtime
