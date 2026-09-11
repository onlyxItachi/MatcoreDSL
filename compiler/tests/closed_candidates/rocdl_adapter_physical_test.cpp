// Real hardware test of the production adapter, separate from fake HIP faults.
#include "closed_rocdl_candidate_v1.h"
#include <hip/hip_runtime_api.h>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
namespace ch = matcore::mdslc::runtime::closed_host_v1;
namespace {
unsigned cases = 0, comparisons = 0, failures = 0;
bool same(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) ||
      std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}
void check(bool pass, const char *what) {
  if (!pass) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
void run(const char *name, unsigned m, unsigned n, unsigned k,
         const std::vector<float> &a, const std::vector<float> &b, bool alias=false) {
  std::fesetenv(FE_DFL_ENV);
  std::vector<float> expected(m*n, 0.0f), output(m*n + 2, -171.f);
  for (unsigned i=0; i<m; ++i)
    for (unsigned j=0; j<n; ++j) {
      volatile float sum = 0.f;
      for (unsigned p=0; p<k; ++p) {
        volatile float product = a[i*k+p] * b[p*n+j];
        sum = sum + product;
      }
      expected[i*n+j] = sum;
    }
  // Seed caller-visible HIP TLS with an actual error and a valid current device.
  int originalDevice = -1;
  check(hipGetDevice(&originalDevice) == hipSuccess, "read caller HIP device");
  check(hipSetDevice(-1) == hipErrorInvalidDevice, "seed actual caller HIP error");
  const auto originalError = hipPeekAtLastError();
  check(originalError != hipSuccess, "HIP error seed is observable");
  std::fesetround(FE_DOWNWARD);
  std::feraiseexcept(FE_INEXACT | FE_UNDERFLOW);
  _mm_setcsr(_mm_getcsr() | 0x8040u);
  const auto mxcsr = _mm_getcsr();
  const auto fpFlags = std::fetestexcept(FE_ALL_EXCEPT);
  errno = EDOM;
  auto code = ch::detail::rocdlGemmCandidate({a.data(),m,k},
      {alias ? a.data() : b.data(),k,n}, {output.data()+1,m,n});
  check(errno == EDOM && _mm_getcsr() == mxcsr &&
        std::fegetround() == FE_DOWNWARD &&
        std::fetestexcept(FE_ALL_EXCEPT) == fpFlags,
        "real adapter preserves errno and caller FP controls/status");
  check(hipPeekAtLastError() == originalError, "real caller HIP error preserved");
  int afterDevice = -1;
  check(hipGetDevice(&afterDevice) == hipSuccess && afterDevice == originalDevice,
        "real caller HIP device preserved");
  std::fesetenv(FE_DFL_ENV);
  check(code == ch::Code::ok, name);
  check(output.front() == -171.f && output.back() == -171.f, "host output guards");
  for (unsigned i=0; i<m*n; ++i) {
    ++comparisons;
    if (!same(output[i+1], expected[i])) {
      ++failures;
      std::fprintf(stderr, "%s[%u]: actual=%a expected=%a\n",
                   name, i, output[i+1], expected[i]);
    }
  }
  ++cases;
}
} // namespace
int main() {
  if (ch::detail::rocdlCandidateAvailable() != ch::Code::ok) {
    std::fprintf(stderr, "FAIL: real gfx1150 unavailable; no fallback\n");
    return 1;
  }
  std::uint32_t state=0x8f23a519;
  auto next=[&]{state ^= state<<13; state ^= state>>17; state ^= state<<5; return state;};
  for (unsigned i=0; i<64; ++i) {
    const auto m=next()%34, n=next()%38, k=next()%36;
    std::vector<float> a(m*k), b(k*n);
    for (auto &v:a) v=(static_cast<int>(next()%129)-64)/16.f;
    for (auto &v:b) v=(static_cast<int>(next()%129)-64)/16.f;
    run("rectangular",m,n,k,a,b);
  }
  run("alias",3,3,3,{1,2,3,4,5,6,7,8,9},{1,2,3,4,5,6,7,8,9},true);
  run("zero-K",3,7,0,{},{});
  run("zero-M",0,7,3,{},std::vector<float>(21));
  run("zero-N",3,0,7,std::vector<float>(21),{});
  run("FMA discriminator",1,1,2,{-1.f,1.f+0x1p-23f},{1.f,1.f-0x1p-23f});
  run("K-order",1,1,4,{0x1p24f,1.f,-0x1p24f,1.f},{1,1,1,1});
  run("subnormal result",1,1,1,{0x1p-126f},{.5f});
  run("subnormal input",1,1,1,{0x1p-149f},{1.f});
  run("signed zero",1,1,1,{-0.f},{1.f});
  run("NaN",1,1,1,{std::numeric_limits<float>::quiet_NaN()},{1.f});
  run("infinity",1,1,1,{std::numeric_limits<float>::infinity()},{1.f});
  run("invalid product",1,1,1,{std::numeric_limits<float>::infinity()},{0.f});
  std::printf("PHYSICAL ROCDL ADAPTER cases=%u output_comparisons=%u failures=%u\n",
               cases,comparisons,failures);
  return failures ? 1 : 0;
}
