#ifndef MATCORE_MDSLC_RUNTIME_CPU_PLANNER_V3_RESOURCES_H
#define MATCORE_MDSLC_RUNTIME_CPU_PLANNER_V3_RESOURCES_H

#include "cpu_execution_context.h"
#include "cpu_parallel_gemm.h"
#include "cpu_planner_v3.h"

namespace matcore::mdslc::runtime {

inline constexpr std::uint32_t kCpuRuntimeValidationEvidenceVersionV1 = 1;

struct CpuRuntimeValidationEvidenceV1 {
  std::uint32_t version = kCpuRuntimeValidationEvidenceVersionV1;
  // This is injected build/package evidence, not inferred from CPUID. Runtime
  // usability is still checked independently before the fact is accepted.
  bool packed_avx512_f32_runtime_validated = false;
};

planner::CpuGemmImplementationResourcesV2
augment_cpu_gemm_implementation_resources_v2(
    const planner::CpuGemmProblemV1 &problem,
    const planner::CpuGemmImplementationResourcesV1 &baseline,
    const CpuExecutionContextV1 *execution_context,
    const CpuRuntimeValidationEvidenceV1 &validation_evidence = {}) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
