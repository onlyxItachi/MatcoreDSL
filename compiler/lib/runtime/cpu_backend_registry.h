#ifndef MATCORE_MDSLC_RUNTIME_CPU_BACKEND_REGISTRY_H
#define MATCORE_MDSLC_RUNTIME_CPU_BACKEND_REGISTRY_H

#include "cpu_planner_v2.h"

#include <cstdint>

namespace matcore::mdslc::runtime {

planner::CpuGemmImplementationResourcesV1
discover_cpu_gemm_implementation_resources_v1(
    const planner::CpuGemmProblemV1 &problem,
    std::uint32_t requested_threads) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
