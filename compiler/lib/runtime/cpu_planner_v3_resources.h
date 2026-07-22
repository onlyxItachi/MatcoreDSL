#ifndef MATCORE_MDSLC_RUNTIME_CPU_PLANNER_V3_RESOURCES_H
#define MATCORE_MDSLC_RUNTIME_CPU_PLANNER_V3_RESOURCES_H

#include "cpu_execution_context.h"
#include "cpu_parallel_gemm.h"
#include "cpu_planner_v3.h"

namespace matcore::mdslc::runtime {

inline constexpr std::uint32_t kCpuRuntimeValidationEvidenceVersionV1 = 1;

struct CpuRuntimeValidationEvidenceV1 {
  std::uint32_t version = kCpuRuntimeValidationEvidenceVersionV1;
  // Each bit is exact process-local execution evidence for the named stable
  // variant. Evidence is never inherited merely because another variant uses
  // the same ISA, microkernel, provider library, or worker context.
  bool reference_f32_runtime_validated = false;
  bool tiled_f32_runtime_validated = false;
  bool compiler_vectorized_f32_runtime_validated = false;
  bool external_openblas_f32_runtime_validated = false;
  bool packed_avx2_f32_runtime_validated = false;
  bool packed_avx512_f32_runtime_validated = false;
  bool parallel_avx2_f32_runtime_validated = false;
  bool parallel_avx512_f32_runtime_validated = false;
};

planner::CpuGemmImplementationResourcesV2
augment_cpu_gemm_implementation_resources_v2(
    const planner::CpuGemmProblemV1 &problem,
    const planner::CpuGemmImplementationResourcesV1 &baseline,
    const CpuExecutionContextV1 *execution_context,
    const CpuRuntimeValidationEvidenceV1 &validation_evidence = {}) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
