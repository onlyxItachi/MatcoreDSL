#include <closed_host_v1.h>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
namespace rt = matcore::mdslc::runtime::closed_host_v1;
namespace {
int checks = 0, failures = 0;
void check(bool value, const char *what) {
  ++checks;
  if (!value) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}
bool same(float a, float b) {
  return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}
void owner(const char *symbol, const char *expected) {
  auto *address = dlsym(RTLD_DEFAULT, symbol);
  Dl_info info{};
  check(address && dladdr(address, &info) && info.dli_fname &&
        std::filesystem::canonical(info.dli_fname) == std::filesystem::canonical(expected),
        "dynamic symbol resolves to exact installed owner");
}
void run(rt::Candidate candidate, rt::Implementation expected, bool fused) {
  std::array<float, 64> input{};
  input[0] = -1; input[1] = 0x1.000002p0F; input[2] = 1;
  input[10] = 0x1.fffffep-1F;
  rt::Session session({candidate});
  rt::Value a, c;
  check(bool(session.read(1, {input.data(), 8, 8, 64, rt::Access::read_write}, a)),
        "private installed read creates original Value");
  std::fenv_t saved{}, before{}, after{};
  const bool savedOk = std::fegetenv(&saved) == 0;
  const bool roundOk = std::fesetround(FE_DOWNWARD) == 0;
  const bool flagsOk = std::feraiseexcept(FE_INEXACT | FE_UNDERFLOW) == 0;
  _mm_setcsr(_mm_getcsr() | 0x8040U);
  const auto mxcsr = _mm_getcsr();
  const bool capturedOk = std::fegetenv(&before) == 0;
  check(savedOk && roundOk && flagsOk && capturedOk &&
        std::fegetround() == FE_DOWNWARD && (mxcsr & 0x8040U) == 0x8040U &&
        std::fetestexcept(FE_INEXACT | FE_UNDERFLOW) == (FE_INEXACT | FE_UNDERFLOW),
        "private nondefault caller FP environment was actually established");
  errno = EDOM;
  const auto status = session.gemm(2, a, a, rt::Numeric::reassociate_f32, c);
  const int observedErrno = errno;
  const auto observedMxcsr = _mm_getcsr();
  std::fegetenv(&after);
  std::fesetenv(&saved);
  check(observedErrno == EDOM && observedMxcsr == mxcsr &&
        before.__control_word == after.__control_word &&
        before.__status_word == after.__status_word,
        "private installed invocation preserves errno and FP state");
  check(bool(status) && c.valid() && c.rows() == 8 && c.columns() == 8 &&
        c.data() != a.data(), "installed candidate creates private output for A==B");
  const auto report = session.candidateReport();
  check(report.frontier == 2 && report.requested == candidate &&
        report.actual == expected && report.numeric == rt::Numeric::reassociate_f32 &&
        report.code == rt::Code::ok && report.invocation_attempted &&
        report.value_issued && report.actual_threads == 1 &&
        !report.provider_contract_checked && !report.provider_probe_invoked,
        "private report identifies actual invocation without provider evidence");
  check(same(c.data()[2], fused ? std::fma(input[1], input[10], -1.0F) : 0.0F),
        "private report agrees with real full-tile rounding discriminator");
  for (std::size_t i = 0; i < input.size(); ++i)
    check(same(a.data()[i], input[i]), "same-input GEMM preserves original Value");
  std::cout << "Private installed implementation: " << rt::implementationName(report.actual) << '\n';
}
void empty(bool zeroK) {
  std::array<float, 8> backing{};
  rt::Session session({rt::Candidate::generated_reassociate});
  rt::Value a, b, c;
  check(bool(session.read(1, {backing.data(), zeroK ? 4U : 0U, zeroK ? 0U : 1U,
                            0, rt::Access::read_write}, a)), "private empty A");
  check(bool(session.read(2, {backing.data(), zeroK ? 0U : 1U, 8,
                            zeroK ? 0U : 8U, rt::Access::read_write}, b)), "private empty B");
  check(bool(session.gemm(3, a, b, rt::Numeric::reassociate_f32, c)),
        "guarded empty or zero-K is adapter-local success");
  const auto report = session.candidateReport();
  check(report.actual == (zeroK ? rt::Implementation::zero_reduction : rt::Implementation::empty_output) &&
        !report.invocation_attempted && report.value_issued && report.actual_threads == 0,
        "empty report never pretends generated leaf execution");
  if (zeroK) for (unsigned i = 0; i < 32; ++i)
    check(same(c.data()[i], 0.0F), "adapter zero-K creates positive zeros");
  rt::Value rejected;
  check(session.gemm(4, a, b, rt::Numeric::strict_f32, rejected).code ==
        rt::Code::candidate_incompatible, "strict still refuses after empty success");
  const auto refusal = session.candidateReport();
  check(refusal.actual == rt::Implementation::none && !refusal.invocation_attempted &&
        !refusal.value_issued && !rejected.valid(),
        "profile refusal does not claim implementation execution");
}
}
int main(int argc, char **argv) {
  if (argc != 3) return 2;
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
    std::cout << "SKIP independent installed private adapter: AVX2/FMA unavailable\n";
    return 77;
  }
  owner("matcore_closed_host_private_value_abi_v2", argv[1]);
  owner("matcore_runtime_gemm_f32_execute_v1", argv[2]);
  check(!dlsym(RTLD_DEFAULT, "__matcore_reassociate_gemm_f32_avx2_v1") &&
        !dlsym(RTLD_DEFAULT, "_mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1"),
        "issued leaf and wrapper are not dynamically exported");
  run(rt::Candidate::generated_reassociate, rt::Implementation::generated_reassociate, true);
  run(rt::Candidate::generated_strict, rt::Implementation::generated_strict, false);
  run(rt::Candidate::automatic, rt::Implementation::generated_strict, false);
  run(rt::Candidate::native_strict, rt::Implementation::native_strict, false);
  empty(false);
  empty(true);
  std::cout << "Independent installed private adapter: " << checks << " checks; "
            << failures << " failures\n";
  return failures ? 1 : 0;
}
