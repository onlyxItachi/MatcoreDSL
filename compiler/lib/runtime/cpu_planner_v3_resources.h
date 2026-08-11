#ifndef MATCORE_MDSLC_RUNTIME_CPU_PLANNER_V3_RESOURCES_H
#define MATCORE_MDSLC_RUNTIME_CPU_PLANNER_V3_RESOURCES_H

#include "cpu_execution_context.h"
#include "cpu_parallel_gemm.h"
#include "cpu_planner_v3.h"

namespace matcore::mdslc::runtime {

inline constexpr std::uint32_t kCpuRuntimeValidationEvidenceVersionV1 = 1;

struct CpuVariantConformanceReportV1 {
  std::uint32_t version = kCpuRuntimeValidationEvidenceVersionV1;
  bool available = false;
  bool environment_compatible = false;
  bool finite_correct = false;
  bool gradual_subnormal_correct = false;
  bool nan_input_propagated = false;
  bool infinity_classes_preserved = false;
  bool inf_times_zero_is_nan = false;
  bool opposite_infinity_sum_is_nan = false;
  bool nonfinite_classes_correct = false;
  bool control_state_preserved = false;
  std::uint32_t authenticated_threads = 0;
  bool conformant = false;
};

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
  CpuVariantConformanceReportV1 reference_f32;
  CpuVariantConformanceReportV1 tiled_f32;
  CpuVariantConformanceReportV1 compiler_vectorized_f32;
  CpuVariantConformanceReportV1 packed_avx2_f32;
  CpuVariantConformanceReportV1 packed_avx512_f32;
  CpuVariantConformanceReportV1 parallel_avx2_f32;
  CpuVariantConformanceReportV1 parallel_avx512_f32;
};

planner::CpuGemmImplementationResourcesV2
augment_cpu_gemm_implementation_resources_v2(
    const planner::CpuGemmProblemV1 &problem,
    const planner::CpuGemmImplementationResourcesV1 &baseline,
    const CpuExecutionContextV1 *execution_context,
    const CpuRuntimeValidationEvidenceV1 &validation_evidence = {}) noexcept;

}  // namespace matcore::mdslc::runtime

#endif
