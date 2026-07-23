#include "cpu_planner_v2.h"

#include <cstdint>
#include <iostream>
#include <string>

int main() {
  namespace planner = matcore::mdslc::planner;
  const planner::CpuCapabilitiesV1 discovered =
      planner::discover_cpu_capabilities_v1();
#if defined(__clang__) && defined(_MSC_VER) && defined(_M_X64)
  if (discovered.architecture != planner::CpuArchitectureV1::x86_64 ||
      !discovered.detection_complete ||
      (planner::has_feature(discovered, planner::CpuFeatureV1::avx2) &&
       discovered.usable_vector_bits < 256)) {
    std::cerr << "clang-cl v1 CPU capability discovery failed closed incorrectly\n";
    return 1;
  }
#endif
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
      true, true, true, true, 131072, 64, 24, 1};

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

  auto excessive_openblas_threads = resources;
  excessive_openblas_threads.requested_threads =
      excessive_openblas_threads.openblas_maximum_threads + 1;
  const planner::CpuGemmPlanV2 excessive_openblas_rejected =
      planner::plan_cpu_gemm_v2(
          problem, capabilities, excessive_openblas_threads,
          planner::CpuGemmRequestV2::force_external_openblas);
  if (excessive_openblas_rejected.status !=
          planner::CpuPlanStatusV1::forced_variant_illegal ||
      excessive_openblas_rejected.candidates[3].legal ||
      excessive_openblas_rejected.candidates[3].actual_threads != 0 ||
      excessive_openblas_rejected.candidates[3].reason !=
          "OpenBLAS requested thread count exceeds provider maximum") {
    std::cerr << "OpenBLAS provider thread ceiling was not enforced\n";
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

  const planner::CpuGemmProblemV1 tiny_problem{
      2, 3, 2, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  const planner::CpuGemmPlanV2 tiny_auto = planner::plan_cpu_gemm_v2(
      tiny_problem, capabilities, resources,
      planner::CpuGemmRequestV2::automatic);
  const planner::CpuGemmProblemV1 crossover_problem{
      16, 16, 16, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  const planner::CpuGemmPlanV2 crossover_auto = planner::plan_cpu_gemm_v2(
      crossover_problem, capabilities, resources,
      planner::CpuGemmRequestV2::automatic);
  const planner::CpuGemmProblemV1 skinny_problem{
      64, 7, 19, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  const planner::CpuGemmPlanV2 skinny_auto = planner::plan_cpu_gemm_v2(
      skinny_problem, capabilities, resources,
      planner::CpuGemmRequestV2::automatic);
  const planner::CpuGemmProblemV1 row_vector_problem{
      1, 4096, 4096, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, 64};
  const planner::CpuGemmPlanV2 row_vector_auto = planner::plan_cpu_gemm_v2(
      row_vector_problem, capabilities, resources,
      planner::CpuGemmRequestV2::automatic);
  if (tiny_auto.selected_id != "cpu.reference.f32.v1" ||
      crossover_auto.selected_id != "cpu.external.openblas.f32.v1" ||
      skinny_auto.selected_id != "cpu.external.openblas.f32.v1" ||
      row_vector_auto.status != planner::CpuPlanStatusV1::selected ||
      row_vector_auto.selected_id == "cpu.external.openblas.f32.v1" ||
      (planner::cpu_compiler_vectorization_build_available_v1() &&
       row_vector_auto.selected_id !=
           "cpu.compiler-vectorized.avx2-fma.f32.v1")) {
    std::cerr << "calibrated OpenBLAS crossover rule regressed\n";
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
