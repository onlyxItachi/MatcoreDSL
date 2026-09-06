#include "closed_host_v1.h"
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif
#pragma STDC FENV_ACCESS ON
namespace h = matcore::mdslc::runtime::closed_host_v1;
static unsigned checks = 0, failures = 0;
static void expect(bool ok, const char *label) {
  ++checks;
  if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", label); }
}
static h::ResourceView view(float *p, unsigned long long m,
                            unsigned long long n, unsigned long long cap,
                            h::Access access = h::Access::read_write) {
  return {p,m,n,cap,access};
}
template <std::size_t N>
static std::array<float,N> extract(const h::Value &v) {
  std::array<float,N> out{};
  h::Session s;
  expect(static_cast<bool>(s.publish(1,v,view(out.data(),v.rows(),v.columns(),N))),
         "immutable value can be copied out through checked publication");
  return out;
}
static h::Value read(h::Session &s, h::Frontier f, float *p,
                     unsigned long long m, unsigned long long n,
                     unsigned long long cap) {
  h::Value result;
  expect(static_cast<bool>(s.read(f,view(p,m,n,cap),result)), "valid snapshot read");
  return result;
}

static void aliases_and_rectangular_math() {
  std::array<float,8> bytes{1,2,3,4,5,6,7,8};
  h::Session s;
  auto old = read(s,1,bytes.data(),2,2,8);
  auto replacement = read(s,2,bytes.data()+4,2,2,4);
  expect(static_cast<bool>(s.publish(3,replacement,view(bytes.data()+1,2,2,7))),
         "shifted overlapping output is legal with independent value storage");
  expect((extract<4>(old) == std::array<float,4>{1,2,3,4}),
         "old value survives partially overlapping publication");
  auto late = read(s,4,bytes.data(),2,2,8);
  expect((extract<4>(late) == std::array<float,4>{1,5,6,7}),
         "late alias read observes current bytes, not entry snapshot");
  expect(static_cast<bool>(s.observe(5,view(bytes.data(),2,2,8))),
         "observation records current immutable contents");
  expect(static_cast<bool>(s.publish(6,old,view(bytes.data(),2,2,8))),
         "same physical destination can be reused");
  expect(s.observation(0).valid() &&
         extract<4>(s.observation(0)) == std::array<float,4>{1,5,6,7},
         "observation contents survive a later overlapping publication");
  expect(s.observationFrontier(0) == 5 && !s.observation(1).valid() &&
         s.observationFrontier(1) == 0, "observation identity and bounds");
  expect(bytes[4] == 8 && bytes[5] == 6 && bytes[7] == 8,
         "only requested publication footprint is overwritten");

  std::array<float,6> a{1,2,3,4,5,6};
  std::array<float,6> b{1,2,3,4,5,6};
  std::array<float,6> d{2,0,1,3,4,1};
  h::Session math;
  auto av = read(math,1,a.data(),2,3,6);
  auto bv = read(math,2,b.data(),3,2,6);
  auto dv = read(math,3,d.data(),3,2,6);
  h::Value c,e;
  expect(static_cast<bool>(math.gemm(4,av,bv,h::Numeric::strict_f32,c)),
         "rectangular first GEMM admitted");
  expect((extract<4>(c) == std::array<float,4>{22,28,49,64}),
         "rectangular first GEMM has ordered math");
  expect(static_cast<bool>(math.gemm(5,dv,c,h::Numeric::strict_f32,e)),
         "rhs-carried rectangular second GEMM admitted");
  expect((extract<6>(e) == std::array<float,6>{44,56,169,220,137,176}),
         "rhs-carried GEMM never commutes operands");
}

static void allocation_and_order() {
  for (unsigned fail = 1; fail <= 32; ++fail) {
    std::array<float,4> a{1,2,3,4}, b{2,0,1,3};
    std::array<float,4> first{-1,-1,-1,-1}, second{-2,-2,-2,-2};
    h::Session s;
    s.configureForTesting({fail,nullptr,nullptr});
    h::Value av,bv,c,late,d;
    s.read(1,view(a.data(),2,2,4),av);
    s.read(2,view(b.data(),2,2,4),bv);
    s.gemm(3,av,bv,h::Numeric::strict_f32,c);
    s.publish(4,c,view(first.data(),2,2,4));
    s.read(5,view(first.data(),2,2,4),late);
    s.gemm(6,late,bv,h::Numeric::strict_f32,d);
    s.observe(7,view(first.data(),2,2,4));
    s.publish(8,d,view(second.data(),2,2,4));
    s.complete(9);
    const auto st = s.status();
    expect(st.code == h::Code::ok || st.code == h::Code::allocation_failure,
           "injected allocation failure retains priority and remains sticky");
    expect(first == (st.publications >= 1 ? std::array<float,4>{4,6,10,12}
                                         : std::array<float,4>{-1,-1,-1,-1}),
           "every allocation failure preserves exact first-publication prefix");
    expect(second == (st.publications >= 2 ? std::array<float,4>{14,18,32,36}
                                          : std::array<float,4>{-2,-2,-2,-2}),
           "allocation failure never leaks partial second publication");
    expect(st.observations <= 1 && st.publications <= 2,
           "effect counters cannot advance past failed frontier");
    if (!st) {
      const auto attempts = s.allocationAttemptsForTesting();
      s.observe(100,view(a.data(),2,2,4));
      expect(s.status().failed_frontier == st.failed_frontier &&
             s.allocationAttemptsForTesting() == attempts,
             "sticky failure prevents later observer allocation/effect");
    }
  }
  std::array<float,4> a{1,2,3,4}, b{2,0,1,3}, out{-1,-1,-1,-1};
  h::Session s;
  auto av=read(s,1,a.data(),2,2,4), bv=read(s,2,b.data(),2,2,4);
  h::Value c,d;
  s.gemm(3,av,bv,h::Numeric::strict_f32,c);
  s.publish(4,c,view(out.data(),2,2,4));
  auto bad=read(s,5,a.data(),1,4,4);
  auto rejected=s.gemm(6,bad,c,h::Numeric::strict_f32,d);
  expect(rejected.code == h::Code::shape_mismatch && rejected.publications == 1 &&
         out == std::array<float,4>{4,6,10,12},
         "second shape failure cannot roll back successful first publication");
}

static h::Code partial(h::detail::CandidateInput, h::detail::CandidateInput,
                       h::detail::CandidateOutput out, void *) {
  if (out.rows && out.columns) out.data[0]=911.0f;
  return h::Code::candidate_failure;
}
static h::Code throws(h::detail::CandidateInput a, h::detail::CandidateInput b,
                      h::detail::CandidateOutput out, void *p) {
  partial(a,b,out,p);
  throw std::runtime_error("injected candidate exception");
}
struct Reenter { h::Session *session; float *external; h::Code nested=h::Code::ok; };
static h::Code reenter(h::detail::CandidateInput, h::detail::CandidateInput,
                       h::detail::CandidateOutput out, void *p) {
  auto &c=*static_cast<Reenter*>(p);
  c.nested=c.session->observe(100,view(c.external,2,2,4)).code;
  for (std::uint64_t i=0;i<out.rows*out.columns;++i) out.data[i]=42;
  return h::Code::ok;
}
static h::Code change_fp(h::detail::CandidateInput, h::detail::CandidateInput,
                         h::detail::CandidateOutput, void *) {
  fesetround(FE_UPWARD);
  return h::Code::ok;
}
static void candidates_and_reentrancy() {
  for (auto callback : {partial,throws,reenter,change_fp}) {
    std::array<float,4> a{1,2,3,4}, out{-3,-3,-3,-3};
    h::Session s;
    Reenter context{&s,out.data()};
    s.configureForTesting({0,callback,&context});
    auto av=read(s,1,a.data(),2,2,4);
    h::Value result;
    auto st=s.gemm(2,av,av,h::Numeric::strict_f32,result);
    expect(!st && !result.valid(), "failed/reentrant candidate cannot issue valid value");
    s.publish(3,result,view(out.data(),2,2,4));
    expect(out == std::array<float,4>{-3,-3,-3,-3},
           "candidate prefix mutation stays private on every failure");
    if(callback == reenter)
      expect(context.nested == h::Code::reentrant_use &&
             s.status().code == h::Code::reentrant_use,
             "outer success cannot overwrite nested sticky reentry rejection");
  }
  {
    float a[1]{1}; h::Session s;
    auto v=read(s,1,a,1,1,1);
    s.configureForTesting({1,nullptr,nullptr});
    expect(!s.status(), "test hook reconfiguration rejects after first action");
    (void)v;
  }
}

static void invalid_inputs() {
  float data[4]{1,2,3,4};
  const h::ResourceView invalid[] = {
    view(nullptr,1,1,1),
    view(data,2,2,3),
    view(data,1,1,4,static_cast<h::Access>(255)),
    view(data,UINT64_MAX,1,4),
    view(data,INT64_MAX,INT64_MAX,UINT64_MAX),
    view(reinterpret_cast<float*>(reinterpret_cast<char*>(data)+1),1,1,1),
    view(reinterpret_cast<float*>(UINTPTR_MAX-3),1,2,2)
  };
  for(auto v:invalid) {
    h::Session s; h::Value out;
    expect(!s.read(1,v,out) && !out.valid(), "invalid resource rejected before dereference");
  }
  {
    h::Session s; h::Value out;
    expect(s.read(0,view(data,2,2,4),out).code == h::Code::invalid_frontier,
           "zero frontier never authorizes read");
  }
  {
    h::Session s; h::Value out;
    auto a=read(s,1,data,2,2,4);
    expect(s.gemm(2,a,a,static_cast<h::Numeric>(255),out).code != h::Code::ok &&
           !out.valid(), "unknown numerical profile never silently falls back");
  }
  {
    h::Session s; auto a=read(s,1,data,2,2,4);
    expect(s.publish(2,a,view(data,2,2,4,h::Access::read_only)).code ==
           h::Code::access_denied, "read-only destination publication rejected");
  }
  {
    h::Session s; h::Value out;
    expect(static_cast<bool>(s.complete(1)), "empty session can complete");
    expect(s.read(2,view(data,2,2,4),out).code == h::Code::already_complete,
           "completed session cannot resume execution");
  }
  {
    h::Session s; h::Value z,b,result;
    expect(static_cast<bool>(s.read(1,view(nullptr,2,0,0),z)), "empty K input valid");
    expect(static_cast<bool>(s.read(2,view(nullptr,0,3,0),b)), "empty K rhs valid");
    expect(static_cast<bool>(s.gemm(3,z,b,h::Numeric::strict_f32,result)),
           "zero-K GEMM retains meaningful nonempty zero result");
    expect(extract<6>(result)==std::array<float,6>{0,0,0,0,0,0},
           "zero-K output is initialized additive identity");
  }
}

static void fp_scope() {
  fenv_t original{};
  expect(fegetenv(&original)==0, "save environment for independent fixture");
  expect(fesetround(FE_DOWNWARD)==0 && feclearexcept(FE_ALL_EXCEPT)==0 &&
         feraiseexcept(FE_INVALID)==0, "construct nondefault caller environment");
#if defined(__x86_64__)
  _mm_setcsr(_mm_getcsr() | (1u<<15u) | (1u<<6u));
  const auto mxcsr=_mm_getcsr();
#endif
  const auto flags=fetestexcept(FE_ALL_EXCEPT);
  float x[1]{std::numeric_limits<float>::max()};
  h::Session s; auto a=read(s,1,x,1,1,1);
  h::Value result;
  expect(static_cast<bool>(s.gemm(2,a,a,h::Numeric::strict_f32,result)),
         "closed math runs in its own default environment");
  expect(fegetround()==FE_DOWNWARD && fetestexcept(FE_ALL_EXCEPT)==flags,
         "caller rounding and preexisting flags restored after overflow math");
#if defined(__x86_64__)
  expect(_mm_getcsr()==mxcsr, "exact caller MXCSR restored including FTZ DAZ flags");
#endif
  float normal[1]{std::bit_cast<float>(std::uint32_t{0x00800000})};
  float half[1]{0.5f};
  h::Session gradual;
  auto n=read(gradual,1,normal,1,1,1), hvalue=read(gradual,2,half,1,1,1);
  h::Value subnormal;
  expect(static_cast<bool>(gradual.gemm(3,n,hvalue,h::Numeric::strict_f32,subnormal)),
         "isolated environment permits gradual underflow despite caller FTZ");
  expect(std::bit_cast<std::uint32_t>(extract<1>(subnormal)[0]) == 0x00400000,
         "new scalar candidate preserves subnormal output bits");
  expect(fegetround()==FE_DOWNWARD && fetestexcept(FE_ALL_EXCEPT)==flags,
         "caller FP state remains exact after subnormal result and publication");
#if defined(__x86_64__)
  expect(_mm_getcsr()==mxcsr, "caller FTZ DAZ flags restored after gradual math");
#endif
  expect(fesetenv(&original)==0, "restore fixture environment");
}
int main() {
  aliases_and_rectangular_math(); allocation_and_order();
  candidates_and_reentrancy(); invalid_inputs(); fp_scope();
  std::printf("Independent host adapter: %u checks, %u failures\n", checks,failures);
  return failures ? 1 : 0;
}
