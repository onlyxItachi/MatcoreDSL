#include "closed_host_v1.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace ch = matcore::mdslc::runtime::closed_host_v1;
namespace {
int checks = 0, failures = 0;
void expect(bool condition, const char *name) {
  ++checks;
  if (!condition) { ++failures; std::cerr << "FAIL: " << name << '\n'; }
}
void ok(ch::Status status, const char *name) {
  expect(status.code == ch::Code::ok, name);
  if (status.code != ch::Code::ok)
    std::cerr << "  " << ch::message(status.code) << " at "
              << status.failed_frontier << '\n';
}
ch::ResourceView view(std::vector<float> &data, std::uint64_t rows,
                      std::uint64_t columns,
                      ch::Access access = ch::Access::read_write) {
  return {data.data(), rows, columns, data.size(), access};
}
std::vector<float> contents(const ch::Value &value) {
  std::vector<float> result(value.rows() * value.columns(), -901.0F);
  ch::Session extraction;
  ok(extraction.publish(1, value, view(result, value.rows(), value.columns())),
     "inspect immutable value through independent publication");
  return result;
}
std::vector<float> oracle(const std::vector<float> &a,
                          const std::vector<float> &b,
                          std::size_t m, std::size_t n, std::size_t k) {
  std::vector<float> result(m * n);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0;
      for (std::size_t t = 0; t < k; ++t)
        sum += static_cast<double>(a[i*k+t]) * b[t*n+j];
      result[i*n+j] = static_cast<float>(sum);
    }
  return result;
}

void mathematical_values_and_aliases() {
  std::vector<float> a{1,2,3,4,5,6}, b{2,3,1,4,2,1}, d{2,1,0,3};
  std::vector<float> c(4, -13), e(4, -17), f(4, -19);
  ch::Session s;
  ch::Value va, vb, vd, vc, lhs_carry, rhs_carry;
  ok(s.read(1, view(a,2,3),2,3,va), "rectangular lhs snapshot");
  ok(s.read(2, view(b,3,2),vb), "rectangular rhs snapshot");
  ok(s.read(3, view(d,2,2),vd), "second GEMM operand snapshot");
  ok(s.gemm(4,va,vb,ch::Numeric::strict_f32,vc), "rectangular strict GEMM");
  ok(s.publish(5,vc,view(c,2,2)), "first publication");
  ok(s.gemm(6,vc,vd,ch::Numeric::strict_f32,lhs_carry), "lhs carried GEMM");
  ok(s.gemm(7,vd,vc,ch::Numeric::reassociate_f32,rhs_carry), "rhs carried GEMM");
  ok(s.publish(8,lhs_carry,view(e,2,2)), "lhs carried publication");
  ok(s.publish(9,rhs_carry,view(f,2,2)), "rhs carried publication");
  const auto expected = oracle(a,b,2,2,3);
  expect(c == expected && e == oracle(expected,d,2,2,2) &&
         f == oracle(d,expected,2,2,2) && e != f,
         "independent rectangular noncommuting mathematical oracle");
  ok(s.complete(10), "ordered completion");
  expect(s.status().completed && s.status().publications == 3 &&
         s.status().completed_effect_frontier == 9,
         "completion separate from publication frontier");

  std::vector<float> storage{1,2,3,4,5,6};
  ch::Session alias;
  ch::Value old, replacement, late;
  ok(alias.read(1,{storage.data()+1,2,2,4,ch::Access::read_only},old),
     "old value through subview");
  ok(alias.read(2,view(d,2,2),replacement), "replacement snapshot");
  ok(alias.publish(3,replacement,{storage.data()+2,2,2,4,ch::Access::read_write}),
     "partially overlapping destination publication");
  ok(alias.read(4,{storage.data()+1,2,2,4,ch::Access::read_only},late),
     "late read of partially overlapping view");
  expect(contents(old) == std::vector<float>({2,3,4,5}) &&
         contents(late) == std::vector<float>({2,2,1,0}),
         "old logical values survive aliases and differ from late resource reads");
  ok(alias.observe(5,{storage.data()+1,2,2,4,ch::Access::read_only}),
     "observation captures current contents");
  const auto observed = alias.observation(0);
  ok(alias.publish(6,old,{storage.data()+1,2,2,4,ch::Access::read_write}),
     "output reuse through overlapping descriptor");
  ok(alias.observe(7,{storage.data()+1,2,2,4,ch::Access::read_only}),
     "second observation captures changed contents");
  expect(contents(observed) == std::vector<float>({2,2,1,0}) &&
         contents(alias.observation(1)) == std::vector<float>({2,3,4,5}) &&
         alias.observationFrontier(0) == 5 && alias.observationFrontier(1) == 7 &&
         !alias.observation(2).valid() && alias.observationFrontier(2) == 0,
         "observation records are immutable and stable across record growth");
}

struct Injection { ch::Session *session; ch::ResourceView destination; };
ch::Code partialFailure(ch::detail::CandidateInput, ch::detail::CandidateInput,
                         ch::detail::CandidateOutput out, void *) {
  if (out.rows && out.columns) out.data[0] = 1234.0F;
  return ch::Code::candidate_failure;
}
ch::Code throwingFailure(ch::detail::CandidateInput, ch::detail::CandidateInput,
                          ch::detail::CandidateOutput out, void *) {
  if (out.rows && out.columns) out.data[0] = 1234.0F;
  throw std::runtime_error("test-only fallible candidate");
}
ch::Code reenter(ch::detail::CandidateInput, ch::detail::CandidateInput,
                 ch::detail::CandidateOutput out, void *context) {
  auto &injection = *static_cast<Injection *>(context);
  injection.session->observe(100,injection.destination);
  if (out.rows && out.columns) out.data[0] = 888.0F;
  return ch::Code::ok;
}
ch::Code changeFp(ch::detail::CandidateInput, ch::detail::CandidateInput,
                  ch::detail::CandidateOutput, void *) {
  std::fesetround(FE_UPWARD);
  std::feraiseexcept(FE_OVERFLOW);
  return ch::Code::ok;
}

void failures_and_prefixes() {
  std::vector<float> a{1,2,3,4}, b{2,0,1,2}, c(4,-7), e(4,-11);
  for (auto hook : {partialFailure, throwingFailure, reenter, changeFp}) {
    std::fill(c.begin(),c.end(),-7);
    std::fill(e.begin(),e.end(),-11);
    ch::Session s;
    ch::Value va,vb,vc,second;
    ch::Session firstCandidate;
    ok(firstCandidate.read(1,view(a,2,2),va), "precompute first candidate lhs");
    ok(firstCandidate.read(2,view(b,2,2),vb), "precompute first candidate rhs");
    ok(firstCandidate.gemm(3,va,vb,ch::Numeric::strict_f32,vc), "first candidate success");
    Injection injection{&s,view(e,2,2)};
    s.configureForTesting({0,hook,&injection});
    ok(s.read(1,view(a,2,2),va), "failure fixture lhs");
    ok(s.read(2,view(b,2,2),vb), "failure fixture rhs");
    ok(s.publish(4,vc,view(c,2,2)), "prefix first publication");
    ok(s.observe(5,view(c,2,2)), "prefix observation");
    second = va;
    const auto failed = s.gemm(6,vc,vb,ch::Numeric::strict_f32,second);
    expect(failed.code == (hook == reenter ? ch::Code::reentrant_use :
                                           ch::Code::candidate_failure) &&
           failed.failed_frontier == 6 && failed.completed_frontier == 5 &&
           failed.completed_effect_frontier == 5 && failed.publications == 1 &&
           failed.observations == 1 && !failed.completed,
           "candidate failure preserves exact completed prefix");
    expect(contents(second) == a, "failed candidate leaves prior output Value unchanged");
    const auto later = s.publish(7,second,view(e,2,2));
    s.observe(8,view(e,2,2));
    s.complete(9);
    expect(later.code == failed.code && s.status().failed_frontier == 6 &&
           c == oracle(a,b,2,2,2) && e == std::vector<float>(4,-11) &&
           s.status().observations == 1,
           "no later effects; successful first publication is not rolled back");
  }
  ch::Session s;
  ch::Value va,vb,vc;
  ok(s.read(1,view(a,2,2),va), "shape failure lhs");
  ok(s.read(2,view(b,2,2),vb), "shape failure rhs");
  ok(s.publish(3,va,view(c,2,2)), "publication before shape failure");
  ch::Value bad;
  ok(s.read(4,{nullptr,0,2,0,ch::Access::read_only},bad), "zero-row operand");
  const auto failure = s.gemm(5,va,bad,ch::Numeric::strict_f32,vc);
  expect(failure.code == ch::Code::shape_mismatch && failure.publications == 1 &&
         failure.completed_frontier == 4 && c == a,
         "second shape validation failure preserves first publication");
}

void allocation_failures() {
  std::vector<float> a{1,2,3,4}, destination(4,-3);
  ch::Session values;
  ch::Value original;
  ok(values.read(1,view(a,2,2),original), "allocation fixture immutable input");
  for (std::uint64_t n : {1,2}) {
    ch::Session s;
    s.configureForTesting({n,nullptr,nullptr});
    ch::Value output = original;
    const auto failure = s.read(1,view(a,2,2),output);
    expect(failure.code == ch::Code::allocation_failure &&
           s.allocationAttemptsForTesting() == n && contents(output) == a,
           "read allocation failure preserves previous Value");
    ch::Session math;
    math.configureForTesting({n,nullptr,nullptr});
    ok(math.publish(1,original,view(destination,2,2)), "publication before allocation failure");
    const auto failed_math = math.gemm(2,original,original,ch::Numeric::strict_f32,output);
    expect(failed_math.code == ch::Code::allocation_failure &&
           failed_math.publications == 1 && destination == a && contents(output) == a,
           "private result allocation failure preserves publication prefix");
  }
  for (std::uint64_t n : {1,2,3}) {
    ch::Session s;
    s.configureForTesting({n,nullptr,nullptr});
    const auto failure = s.observe(1,view(a,2,2));
    expect(failure.code == ch::Code::allocation_failure &&
           failure.observations == 0 && !s.observation(0).valid() &&
           failure.completed_effect_frontier == 0,
           "snapshot or record allocation failure cannot retire observation");
  }
  ch::Session noAllocation;
  noAllocation.configureForTesting({1,nullptr,nullptr});
  ok(noAllocation.publish(1,original,view(destination,2,2)),
     "publication needs no fallible allocation");
  expect(noAllocation.allocationAttemptsForTesting() == 0 && destination == a,
         "no fallible bookkeeping after publication starts");
}

void invalid_shapes_and_access() {
  std::vector<float> data{1,2,3,4};
  const auto max = std::numeric_limits<std::uint64_t>::max();
  std::array<ch::ResourceView,7> invalid{{
      {nullptr,2,2,4,ch::Access::read_only},
      {data.data(),2,2,3,ch::Access::read_only},
      {data.data(),max,0,4,ch::Access::read_only},
      {data.data(),std::uint64_t{1} << 62,4,max,ch::Access::read_only},
      {reinterpret_cast<float *>(reinterpret_cast<std::uintptr_t>(data.data())+1),
       1,1,1,ch::Access::read_only},
      {reinterpret_cast<float *>(std::numeric_limits<std::uintptr_t>::max()-3),
       1,2,2,ch::Access::read_only},
      {data.data(),2,2,4,static_cast<ch::Access>(55)}}};
  for (const auto v : invalid) {
    ch::Session s;
    ch::Value value;
    expect(!s.read(1,v,value) && !value.valid(), "invalid storage rejected before read");
  }
  ch::Session s;
  ch::Value value;
  expect(s.read(1,view(data,2,2),1,4,value).code == ch::Code::shape_mismatch,
         "requested shape cannot silently reinterpret external descriptor");
  ch::Session source;
  ok(source.read(1,view(data,2,2),value), "valid snapshot for access tests");
  ch::Session output;
  expect(output.publish(1,value,view(data,2,2,ch::Access::read_only)).code ==
             ch::Code::access_denied,
         "read-only destination rejected before write");
  ch::Session invalidValue;
  ch::Value absent;
  expect(invalidValue.gemm(1,absent,value,ch::Numeric::strict_f32,absent).code ==
             ch::Code::invalid_value, "default Value grants no value authority");
  ch::Session invalidProfile;
  expect(invalidProfile.gemm(1,value,value,static_cast<ch::Numeric>(99),absent).code ==
             ch::Code::invalid_value, "unknown numerical permissions fail closed");
  ch::Session frontiers;
  expect(frontiers.read(0,view(data,2,2),absent).code == ch::Code::invalid_frontier,
         "zero frontier rejected");
  ch::Session order;
  ok(order.read(2,view(data,2,2),absent), "valid first frontier");
  expect(order.publish(2,absent,view(data,2,2)).code == ch::Code::invalid_frontier,
         "duplicate frontier rejected before effect");
  ch::Session completed;
  ok(completed.complete(1), "empty session completion");
  expect(completed.read(2,view(data,2,2),absent).code == ch::Code::already_complete,
         "completion cannot be reopened");
  ch::Session configuredLate;
  ok(configuredLate.read(1,view(data,2,2),absent), "test configuration misuse input");
  configuredLate.configureForTesting({});
  expect(configuredLate.status().code == ch::Code::invalid_frontier &&
         configuredLate.status().failed_frontier == 0,
         "test injection cannot change a running contract");
}

void zero_and_numerical_contract() {
  ch::Session s;
  ch::Value a,b,c;
  ok(s.read(1,{nullptr,2,0,0,ch::Access::read_only},a), "zero-K lhs no dereference");
  ok(s.read(2,{nullptr,0,3,0,ch::Access::read_only},b), "zero-K rhs no dereference");
  ok(s.gemm(3,a,b,ch::Numeric::strict_f32,c), "zero-K GEMM");
  const auto result = contents(c);
  expect(result.size() == 6 && std::all_of(result.begin(),result.end(),[](float f) {
           return std::bit_cast<std::uint32_t>(f) == 0;
         }), "empty reduction produces positive f32 zero");
  ch::Session hugeZero;
  const auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  ok(hugeZero.read(1,{nullptr,max,0,0,ch::Access::read_only},a), "huge empty lhs");
  ok(hugeZero.read(2,{nullptr,0,0,0,ch::Access::read_only},b), "empty rhs");
  ok(hugeZero.gemm(3,a,b,ch::Numeric::strict_f32,c), "empty output skips huge outer loop");
  ok(hugeZero.publish(4,c,{nullptr,max,0,0,ch::Access::read_write}), "empty publication no pointer arithmetic");
  std::vector<float> lhs{-16785408.0F,4097.0F}, rhs{1.0F,4097.0F};
  ch::Session noFma;
  ok(noFma.read(1,view(lhs,1,2),a), "FMA counterexample lhs");
  ok(noFma.read(2,view(rhs,2,1),b), "FMA counterexample rhs");
  ok(noFma.gemm(3,a,b,ch::Numeric::strict_f32,c), "strict no-FMA execution");
  expect(contents(c) == std::vector<float>{0.0F}, "multiply and addition round separately");
  lhs = {16777216.0F,1.0F,-16777216.0F}; rhs = {1,1,1};
  ch::Session order;
  ok(order.read(1,view(lhs,1,3),a), "reassociation counterexample lhs");
  ok(order.read(2,view(rhs,3,1),b), "reassociation counterexample rhs");
  ok(order.gemm(3,a,b,ch::Numeric::strict_f32,c), "strict increasing-K execution");
  expect(contents(c) == std::vector<float>{0.0F}, "strict increasing-K rounding retained");
}

void fp_isolation() {
#if defined(__linux__) && defined(__x86_64__)
  std::fenv_t original;
  expect(std::fegetenv(&original) == 0, "save original FP environment");
  std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_INVALID);
  std::fesetround(FE_DOWNWARD);
  _mm_setcsr(_mm_getcsr() | (1U << 15U) | (1U << 6U));
  const auto csr = _mm_getcsr();
  std::vector<float> a{std::numeric_limits<float>::max()}, b{2.0F};
  ch::Session s;
  ch::Value va,vb,vc;
  ok(s.read(1,view(a,1,1),va), "FP fixture read lhs preserves ambient state");
  ok(s.read(2,view(b,1,1),vb), "FP fixture read rhs preserves ambient state");
  ok(s.gemm(3,va,vb,ch::Numeric::strict_f32,vc), "overflow is IEEE value, not checked failure");
  expect(_mm_getcsr() == csr && std::fegetround() == FE_DOWNWARD &&
         std::fetestexcept(FE_ALL_EXCEPT) == FE_INVALID,
         "successful candidate restores complete caller FP state");
  expect(std::isinf(contents(vc)[0]), "candidate used nearest-even rather than caller downward rounding");
  ch::Session failing;
  failing.configureForTesting({0,changeFp,nullptr});
  expect(failing.gemm(4,va,vb,ch::Numeric::strict_f32,vc).code == ch::Code::candidate_failure,
         "post-call FP control mutation rejects candidate");
  expect(_mm_getcsr() == csr && std::fegetround() == FE_DOWNWARD &&
         std::fetestexcept(FE_ALL_EXCEPT) == FE_INVALID,
         "failing candidate restores complete caller FP state");
  std::fesetenv(&original);
#endif
}
} // namespace

int main() {
#if !defined(__linux__) || !defined(__x86_64__)
  ch::Session s;
  ch::Value value;
  expect(s.read(1,{},value).code == ch::Code::unsupported_fp_environment,
         "unknown platform fails closed");
#else
  mathematical_values_and_aliases();
  failures_and_prefixes();
  allocation_failures();
  invalid_shapes_and_access();
  zero_and_numerical_contract();
  fp_isolation();
#endif
  std::cout << checks << " checks; " << failures << " failures\n";
  return failures != 0;
}
