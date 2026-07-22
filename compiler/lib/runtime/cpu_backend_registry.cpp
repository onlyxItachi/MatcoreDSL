#include "cpu_backend_registry.h"

#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"

namespace matcore::mdslc::runtime {

planner::CpuGemmImplementationResourcesV1
discover_cpu_gemm_implementation_resources_v1(
    const planner::CpuGemmProblemV1 &problem,
    std::uint32_t requested_threads) noexcept {
  const OpenBlasProviderInfoV1 provider = openblas_provider_info_v1();
  planner::CpuGemmImplementationResourcesV1 resources;
  resources.openblas_linked = provider.linked;
  resources.openblas_local_thread_control = provider.linked;
  if (provider.maximum_reported_threads > 0) {
    resources.openblas_maximum_threads =
        static_cast<std::uint32_t>(provider.maximum_reported_threads);
  }
  resources.native_packed_avx2_fma_compiled =
      cpu_packed_avx2_build_available_v1();
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  const CpuPackedGemmStatusV1 workspace_status =
      cpu_packed_avx2_workspace_requirements_v1(
          problem, CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
          &requirements);
  resources.native_packed_workspace_size_valid =
      workspace_status == CpuPackedGemmStatusV1::success;
  if (resources.native_packed_workspace_size_valid) {
    resources.native_packed_workspace_bytes = requirements.total_bytes;
    resources.native_packed_workspace_alignment =
        static_cast<std::uint32_t>(requirements.alignment_bytes);
  }
  resources.requested_threads = requested_threads;
  return resources;
}

}  // namespace matcore::mdslc::runtime
