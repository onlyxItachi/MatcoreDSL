#ifndef MATCORE_MDSLC_RUNTIME_CPU_BACKEND_REGISTRY_H
#define MATCORE_MDSLC_RUNTIME_CPU_BACKEND_REGISTRY_H

#include "cpu_planner_v2.h"

#include <cstdint>

namespace matcore::mdslc::runtime {

enum class CpuExternalProviderProbeV1 : std::uint8_t {
  exclude = 0,
  include = 1,
};

planner::CpuGemmImplementationResourcesV1
discover_cpu_gemm_implementation_resources_v1(
    const planner::CpuGemmProblemV1 &problem,
    std::uint32_t requested_threads,
    CpuExternalProviderProbeV1 external_provider_probe) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
