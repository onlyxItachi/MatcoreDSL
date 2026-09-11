#include "closed_host_v1.h"
#include "closed_generated_capability_v1.h"

#include <array>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include "../support/closed_fp_fixture.h"

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
namespace ch = matcore::mdslc::runtime::closed_host_v1;
namespace pl = matcore::mdslc::platform;
namespace {
int checks = 0, failures = 0, discoveries = 0, invocations = 0;
pl::CpuCapabilitiesV2 discovered;
void check(bool value, const char *what) {
  ++checks;
  if (!value) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}
pl::CpuCapabilitiesV2 supported() {
  pl::CpuCapabilitiesV2 value;
  value.architecture = pl::ArchitectureKindV1::x86_64;
  constexpr auto bits = pl::feature_bit(pl::CpuFeatureV2::avx2) |
                        pl::feature_bit(pl::CpuFeatureV2::fma);
  value.hardware = {bits, bits};
  value.os_enabled = {bits, bits};
  value.os_xstate_mask_known = true;
  value.os_xstate_mask = 6;
  return value;
}
struct Memref {
  float *allocated, *aligned;
  std::int64_t offset, sizes[2], strides[2];
};
static_assert(sizeof(Memref) == 56);
void predicates() {
  const auto good = supported();
  check(ch::detail::generatedReassociateCpuSupported(good),
        "direct hardware/OS proof does not require unrelated packed selftest");
  for (int variant = 0; variant < 16; ++variant) {
    auto bad = good;
    switch (variant) {
    case 0: bad.version = 1; break;
    case 1: bad.architecture = pl::ArchitectureKindV1::unknown; break;
    case 2: bad.architecture = pl::ArchitectureKindV1::aarch64; break;
    case 3: bad.hardware.known = 0; break;
    case 4: bad.os_enabled.known = 0; break;
    case 5: bad.hardware.available &= ~pl::feature_bit(pl::CpuFeatureV2::avx2); break;
    case 6: bad.hardware.available &= ~pl::feature_bit(pl::CpuFeatureV2::fma); break;
    case 7: bad.os_enabled.available &= ~pl::feature_bit(pl::CpuFeatureV2::avx2); break;
    case 8: bad.os_enabled.available &= ~pl::feature_bit(pl::CpuFeatureV2::fma); break;
    case 9: bad.os_xstate_mask_known = false; break;
    case 10: bad.os_xstate_mask = 2; break;
    case 11: bad.os_xstate_mask = 4; break;
    case 12: bad.hardware.known |= UINT64_C(1) << 63; break;
    case 13: bad.os_enabled.available |= UINT64_C(1) << 63; break;
    case 14: bad.hardware = {}; bad.runtime_validation = good.hardware; break;
    case 15: bad.os_enabled = {}; bad.implementation = good.hardware; break;
    }
    check(!ch::detail::generatedReassociateCpuSupported(bad),
          "missing, unknown, malformed or unrelated proof rejected");
  }
}
void guardedShapes() {
  for (bool available : {false, true})
    for (auto shape : {std::array<std::uint64_t, 3>{4,8,2}, {0,8,2},
                       {4,0,2}, {4,8,0}}) {
      const auto [m,n,k] = shape;
      std::array<float, 32> input{}, output{};
      input.fill(1);
      output.fill(-9);
      ch::Session session(ch::Options{ch::Candidate::generated_reassociate});
      ch::Value a,b,c;
      session.read(1, {input.data(),m,k,32}, a);
      session.read(2, {input.data(),k,n,32}, b);
      const auto allocations = session.allocationAttemptsForTesting();
      discovered = available ? supported() : pl::CpuCapabilitiesV2{};
      discoveries = invocations = 0;
      const auto saved_mxcsr = closed_fp_fixture::snapshot();
      errno = EDOM;
      const auto status = session.gemm(3,a,b,ch::Numeric::reassociate_f32,c);
      check(errno == EDOM && closed_fp_fixture::snapshot() == saved_mxcsr,
            "discovery preserves caller errno and FP state");
      check(discoveries == 1, "forced candidate checks hardware even for empty math");
      check(status.code == (available ? ch::Code::ok : ch::Code::candidate_unavailable),
            "availability gate precedes empty/zero shortcut");
      if (!available) {
        check(session.allocationAttemptsForTesting() == allocations,
              "unavailable gate precedes output allocation");
        check(!c.valid() && invocations == 0, "unavailable cannot invoke or issue");
        continue;
      }
      check(invocations == (m && n && k ? 1 : 0), "only positive math enters leaf");
      check(bool(session.publish(4,c,{output.data(),m,n,32,ch::Access::read_write})),
            "guarded result publishes");
      for (std::uint64_t i = 0; i < m*n; ++i)
        check(output[i] == static_cast<float>(k), "exact bounded math or positive zero");
      if (!k && m && n)
        check(!std::signbit(output[0]), "zero reduction is positive zero");
    }
}
void noUnchosenProbe() {
  for (auto candidate : {ch::Candidate::automatic, ch::Candidate::native_strict,
                        ch::Candidate::generated_strict, ch::Candidate::existing_native,
                        ch::Candidate::authenticated_openblas,
                        ch::Candidate::generated_reassociate}) {
    float x=2;
    ch::Session session(ch::Options{candidate});
    ch::Value a,c;
    session.read(1,{&x,1,1,1},a);
    discoveries = invocations = 0;
    const auto mxcsr = closed_fp_fixture::snapshot();
    errno = ERANGE;
    const auto status = session.gemm(2,a,a,ch::Numeric::strict_f32,c);
    check(discoveries == 0 && invocations == 0, "unchosen or strict-refused never probes");
    check(errno == ERANGE && mxcsr == closed_fp_fixture::snapshot(), "unchosen preserves errno/FP");
    if (candidate == ch::Candidate::generated_reassociate)
      check(status.code == ch::Code::candidate_incompatible,
            "strict profile fails before absent hardware test");
  }
}
void prefix() {
  float x=3, out=-1;
  ch::Session session(ch::Options{ch::Candidate::generated_reassociate});
  ch::Value a,c;
  session.read(1,{&x,1,1,1},a);
  discovered = supported();
  discoveries = invocations = 0;
  check(bool(session.gemm(2,a,a,ch::Numeric::reassociate_f32,c)), "first math succeeds");
  check(bool(session.publish(3,c,{&out,1,1,1,ch::Access::read_write})), "prefix publication");
  check(bool(session.observe(4,{&out,1,1,1})), "prefix observation");
  const auto old = c;
  const auto status = session.gemm(5,a,a,ch::Numeric::strict_f32,c);
  check(status.code == ch::Code::candidate_incompatible && status.failed_frontier == 5 &&
        status.completed_frontier == 4 && status.completed_effect_frontier == 4 &&
        status.publications == 1 && status.observations == 1,
        "late strict failure preserves exact effect prefix");
  check(discoveries == 1 && invocations == 1, "refused second math does not probe/invoke");
  check(c.data() == old.data() && c.data()[0] == 9 && out == 9,
        "failed math leaves old owning value and publication intact");
  auto observation = session.observation(0);
  auto result = std::move(session).takeResult();
  check(result.observation_count() == 1 && observation.data()[0] == 9,
        "retirement preserves owning observation");
}
} // namespace

// This executable substitutes only test-linked definitions, never a production
// setter, installed hook or source authority. Production calls the real platform.
namespace matcore::mdslc::platform {
CpuCapabilitiesV2 discover_cpu_capabilities_v2(const CpuImplementationAvailabilityV2 &) noexcept {
  ++discoveries;
  errno = EINVAL;
  return discovered;
}
}
extern "C" void _mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1(
    Memref *a, Memref *b, Memref *c) {
  ++invocations;
  check(c->aligned != a->aligned && c->aligned != b->aligned,
        "adapter supplies separate output even when input/input aliases");
  for (std::int64_t i=0; i<c->sizes[0]; ++i)
    for (std::int64_t j=0; j<c->sizes[1]; ++j) {
      float sum=0;
      for (std::int64_t k=0; k<a->sizes[1]; ++k)
        sum += a->aligned[i*a->strides[0]+k]*b->aligned[k*b->strides[0]+j];
      c->aligned[i*c->strides[0]+j] = sum;
    }
}
int main() {
  std::fenv_t original;
  std::fegetenv(&original);
  std::fesetround(FE_DOWNWARD);
  std::feraiseexcept(FE_INEXACT | FE_UNDERFLOW);
  closed_fp_fixture::enableFlush();
  predicates(); guardedShapes(); noUnchosenProbe(); prefix();
  std::fesetenv(&original);
  std::cout << "Generated reassociate guard: " << checks << " checks; " << failures << " failures\n";
  return failures ? 1 : 0;
}
