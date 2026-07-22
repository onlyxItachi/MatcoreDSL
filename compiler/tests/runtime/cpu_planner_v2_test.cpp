#include "cpu_planner_v2.h"

#include <cstdint>
#include <iostream>
#include <string>

int main() {
  namespace planner = matcore::mdslc::planner;
  const planner::CpuGemmProblemV1 problem{
      128, 128, 128, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  const planner::CpuCapabilitiesV1 capabilities{
      planner::kCpuCapabilitiesVersionV1, planner::CpuArchitectureV1::x86_64,
      true,
      planner::feature_bit(planner::CpuFeatureV1::portable_scalar_f32) |
          planner::feature_bit(planner::CpuFeatureV1::avx2) |
          planner::feature_bit(planner::CpuFeatureV1::fma),
      256};
  const planner::CpuGemmImplementationResourcesV1 resources{
      true, true, true, true, 131072, 64, 1};

  const planner::CpuGemmPlanV2 plan = planner::plan_cpu_gemm_v2(
      problem, capabilities, resources,
      planner::CpuGemmRequestV2::force_native_packed_avx2_fma);
  if (plan.status != planner::CpuPlanStatusV1::selected ||
      plan.selected_id != "cpu.native-packed.avx2-fma.f32.v1" ||
      plan.candidates.size() != planner::kCpuGemmCandidateCountV2 ||
      plan.candidates[4].required_workspace_bytes != 131072 ||
      plan.candidates[4].required_workspace_alignment != 64) {
    std::cerr << "forced native packed plan failed\n";
    return 1;
  }

  auto unavailable = resources;
  unavailable.openblas_linked = false;
  const planner::CpuGemmPlanV2 rejected = planner::plan_cpu_gemm_v2(
      problem, capabilities, unavailable,
      planner::CpuGemmRequestV2::force_external_openblas);
  if (rejected.status != planner::CpuPlanStatusV1::forced_variant_illegal ||
      rejected.candidates[3].legal ||
      rejected.candidates[3].reason != "OpenBLAS CBLAS adapter is not linked") {
    std::cerr << "unavailable OpenBLAS was not rejected deterministically\n";
    return 1;
  }

  auto two_threads = resources;
  two_threads.requested_threads = 2;
  const planner::CpuGemmPlanV2 single_thread_rejected =
      planner::plan_cpu_gemm_v2(
          problem, capabilities, two_threads,
          planner::CpuGemmRequestV2::force_native_packed_avx2_fma);
  if (single_thread_rejected.status !=
          planner::CpuPlanStatusV1::forced_variant_illegal ||
      single_thread_rejected.candidates[4].reason !=
          "native packed v1 is single-threaded") {
    std::cerr << "packed v1 accepted a multi-thread request\n";
    return 1;
  }

  auto missing_portable = capabilities;
  missing_portable.features =
      planner::feature_bit(planner::CpuFeatureV1::avx2) |
      planner::feature_bit(planner::CpuFeatureV1::fma);
  const planner::CpuGemmPlanV2 untrusted_external = planner::plan_cpu_gemm_v2(
      problem, missing_portable, resources,
      planner::CpuGemmRequestV2::force_external_openblas);
  const planner::CpuGemmPlanV2 untrusted_native = planner::plan_cpu_gemm_v2(
      problem, missing_portable, resources,
      planner::CpuGemmRequestV2::force_native_packed_avx2_fma);
  if (untrusted_external.status !=
          planner::CpuPlanStatusV1::forced_variant_illegal ||
      untrusted_external.candidates[3].legal ||
      untrusted_native.status !=
          planner::CpuPlanStatusV1::forced_variant_illegal ||
      untrusted_native.candidates[4].legal ||
      untrusted_external.candidates[3].reason !=
          "required CPU feature set is unavailable" ||
      untrusted_native.candidates[4].reason !=
          "required CPU feature set is unavailable") {
    std::cerr << "v2 planner ignored registry feature requirements\n";
    return 1;
  }

  const std::size_t required =
      planner::format_cpu_gemm_plan_v2(plan, nullptr, 0);
  std::string diagnostic(required + 1, '\0');
  planner::format_cpu_gemm_plan_v2(plan, diagnostic.data(), diagnostic.size());
  diagnostic.resize(required);
  if (diagnostic.find("workspace=131072") == std::string::npos ||
      diagnostic.find("cpu.external.openblas.f32.v1") == std::string::npos) {
    std::cerr << "planner v2 diagnostic omitted resource evidence\n";
    return 1;
  }
  return 0;
}
