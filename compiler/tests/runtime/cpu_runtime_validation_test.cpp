#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"
#include "cpu_packed_avx512.h"
#include "cpu_runtime_validation.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string_view>

namespace {

namespace runtime = matcore::mdslc::runtime;
namespace planner = matcore::mdslc::planner;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void expect_conformant(
    const runtime::CpuVariantConformanceReportV1 &report,
    std::uint32_t threads, std::string_view name) {
  const bool valid =
      report.version == runtime::kCpuRuntimeValidationEvidenceVersionV1 &&
      report.available && report.environment_compatible &&
      report.finite_correct && report.gradual_subnormal_correct &&
      report.nan_input_propagated && report.infinity_classes_preserved &&
      report.inf_times_zero_is_nan && report.opposite_infinity_sum_is_nan &&
      report.nonfinite_classes_correct && report.control_state_preserved &&
      report.authenticated_threads == threads && report.conformant;
  if (!valid) {
    std::cerr << "conformance detail " << name << ": available="
              << report.available << " environment="
              << report.environment_compatible << " finite="
              << report.finite_correct << " gradual="
              << report.gradual_subnormal_correct << " nan="
              << report.nan_input_propagated << " infinity="
              << report.infinity_classes_preserved << " inf*0="
              << report.inf_times_zero_is_nan << " opposite-inf="
              << report.opposite_infinity_sum_is_nan << " nonfinite="
              << report.nonfinite_classes_correct << " control="
              << report.control_state_preserved << " threads="
              << report.authenticated_threads << " conformant="
              << report.conformant << '\n';
  }
  expect(valid, name);
}

}  // namespace

int main() {
  runtime::CpuExecutionContextConfigV1 config;
  config.requested_threads = 2;
  config.maximum_threads = 2;
  runtime::CpuExecutionStatusV1 create_status{};
  runtime::CpuWorkerAffinityReportV1 affinity;
  auto context = runtime::CpuExecutionContextV1::create(
      config, &create_status, &affinity);
  expect(context != nullptr &&
             create_status == runtime::CpuExecutionStatusV1::success,
         "two-worker validation context creates");
  if (context == nullptr) return 1;

  const auto evidence = runtime::validate_cpu_runtime_variants_v1(*context);
  expect(evidence.version ==
             runtime::kCpuRuntimeValidationEvidenceVersionV1,
         "runtime evidence version is exact");
  expect_conformant(evidence.reference_f32, 1,
                    "reference finite/subnormal/nonfinite/control conformance");
  expect_conformant(evidence.tiled_f32, 1,
                    "tiled finite/subnormal/nonfinite/control conformance");
  if (planner::cpu_compiler_vectorization_build_available_v1()) {
    expect_conformant(
        evidence.compiler_vectorized_f32, 1,
        "compiler-vectorized finite/subnormal/nonfinite/control conformance");
  } else {
    expect(!evidence.compiler_vectorized_f32.available &&
               !evidence.compiler_vectorized_f32_runtime_validated,
           "instrumented compiler-vectorized path is not advertised");
  }

  const bool avx2 = runtime::cpu_packed_avx2_runtime_usable_v1();
  if (avx2) {
    expect_conformant(
        evidence.packed_avx2_f32, 1,
        "packed AVX2 finite/subnormal/nonfinite/control conformance");
    expect_conformant(
        evidence.parallel_avx2_f32, 2,
        "parallel AVX2 authenticates both workers and numerical classes");
  } else {
    expect(!evidence.packed_avx2_f32.available &&
               !evidence.packed_avx2_f32_runtime_validated &&
               !evidence.parallel_avx2_f32.available &&
               !evidence.parallel_avx2_f32_runtime_validated,
           "unavailable AVX2 variants are not inferred as conformant");
  }

  const bool avx512 = runtime::cpu_packed_avx512_runtime_usable_v1();
  if (avx512) {
    expect_conformant(
        evidence.packed_avx512_f32, 1,
        "packed AVX-512 finite/subnormal/nonfinite/control conformance");
    expect_conformant(
        evidence.parallel_avx512_f32, 2,
        "parallel AVX-512 authenticates both workers and numerical classes");
  } else {
    expect(!evidence.packed_avx512_f32.available &&
               !evidence.packed_avx512_f32_runtime_validated &&
               !evidence.parallel_avx512_f32.available &&
               !evidence.parallel_avx512_f32_runtime_validated,
           "unavailable AVX-512 variants are not inferred as conformant");
  }

  const auto provider = runtime::openblas_provider_info_v1();
  const auto provider_conformance = runtime::openblas_conformance_report_v1();
  expect(provider.linked == provider_conformance.provider_linked,
         "OpenBLAS report is keyed to the linked provider identity");
  if (provider.linked) {
    expect(provider_conformance.conformant &&
               evidence.external_openblas_f32_runtime_validated,
           "linked OpenBLAS is advertised only after immutable conformance");
  } else {
    expect(!provider_conformance.conformant &&
               !evidence.external_openblas_f32_runtime_validated,
           "unlinked OpenBLAS is never advertised");
  }

  return failures == 0 ? 0 : 1;
}
