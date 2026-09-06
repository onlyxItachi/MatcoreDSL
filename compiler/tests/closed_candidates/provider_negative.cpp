#include "closed_host_v1.h"
#include "matcore/runtime_c.h"
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
namespace ch = matcore::mdslc::runtime::closed_host_v1;
const char *mode = nullptr;
int calls = 0;

// Deliberately hostile substitute for the existing runtime boundary, linked
// ONLY into this negative-control executable. No production
// callback/configuration.
extern "C" matcore_status_v0 matcore_runtime_gemm_f32_execute_v1(
    const matcore_tensor_desc_v0 *out, const matcore_tensor_desc_v0 *lhs,
    const matcore_tensor_desc_v0 *rhs, const matcore_policy_v0 *,
    const matcore_cpu_gemm_execution_options_v1 *, void *, std::size_t,
    matcore_cpu_gemm_plan_report_v2 *report) noexcept {
  ++calls;
  report->plan_status = MATCORE_CPU_PLAN_STATUS_SELECTED_V1;
  report->selected_stable_id = "cpu.external.openblas.f32.v1";
  report->selected_actual_threads = 1;
  auto *a = static_cast<float *>(lhs->data),
       *b = static_cast<float *>(rhs->data);
  auto *c = static_cast<float *>(out->data);
  for (std::int64_t i = 0; i < out->dims[0]; ++i)
    for (std::int64_t j = 0; j < out->dims[1]; ++j) {
      float sum = 0;
      for (std::int64_t p = 0; p < lhs->dims[1]; ++p) {
        const float product = a[i * lhs->dims[1] + p] * b[p * rhs->dims[1] + j];
        sum = sum + product;
      }
      c[i * out->dims[1] + j] = sum;
    }
  matcore_status_v0 status{};
  status.struct_size = sizeof(status);
  if (std::strcmp(mode, "wrong_numeric") == 0)
    c[0] = 13;
  if (std::strcmp(mode, "wrong_variant") == 0)
    report->selected_stable_id = "cpu.reference.f32.v1";
  if (std::strcmp(mode, "wrong_threads") == 0)
    report->selected_actual_threads = 2;
  if (std::strcmp(mode, "control_corruption") == 0)
    _mm_setcsr(_mm_getcsr() | 0x8000U);
  if (std::strcmp(mode, "partial_failure") == 0 && calls == 6) {
    c[0] = 12345;
    status.code = MATCORE_STATUS_EXTERNAL_PROVIDER_FAILURE_V0;
  }
  return status;
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  mode = argv[1];
  float a = 2, b = 3, external = 91;
  ch::Session session(ch::Options{ch::Candidate::authenticated_openblas});
  ch::Value lhs, rhs, result;
  if (!session.read(1, {&a, 1, 1, 1}, lhs) ||
      !session.publish(2, lhs, {&external, 1, 1, 1, ch::Access::read_write}) ||
      !session.read(3, {&b, 1, 1, 1}, rhs))
    return 1;
  std::fenv_t saved;
  std::fegetenv(&saved);
  std::fesetround(FE_DOWNWARD);
  std::feraiseexcept(FE_INEXACT);
  const auto mxcsr = _mm_getcsr();
  const auto status =
      session.gemm(4, lhs, rhs, ch::Numeric::reassociate_f32, result);
  const auto report = session.candidateReport();
  const bool partial = std::strcmp(mode, "partial_failure") == 0;
  const auto wanted = std::strcmp(mode, "wrong_numeric") == 0
                          ? ch::Code::candidate_incompatible
                          : ch::Code::candidate_failure;
  const bool correct =
      status.code == wanted && !result.valid() && external == 2 &&
      status.completed_frontier == 3 && status.completed_effect_frontier == 2 &&
      status.failed_frontier == 4 && report.provider_contract_checked &&
      report.provider_probe_invoked && !report.value_issued &&
      report.invocation_attempted == partial &&
      (partial ? calls == 6 : calls >= 1) && _mm_getcsr() == mxcsr &&
      std::fegetround() == FE_DOWNWARD;
  std::fesetenv(&saved);
  if (!correct) {
    std::cerr << mode << " status=" << static_cast<int>(status.code)
              << " calls=" << calls << "\n";
    return 1;
  }
  std::cout << mode
            << ": rejected, isolated output discarded, earlier effect/FP state "
               "preserved\n";
}
