#include "closed_host_v1.h"
#include "closed_gpu_capability_v1.h"
#include <cerrno>
#include <cfenv>
#include <cstring>
#include <iostream>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
namespace ch = matcore::mdslc::runtime::closed_host_v1;
int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const bool nvvm = std::strcmp(argv[1], "nvvm") == 0;
  if (!nvvm && std::strcmp(argv[1], "rocdl") != 0) return 2;
  const auto candidate = nvvm ? ch::Candidate::generated_nvvm : ch::Candidate::generated_rocdl;
  unsigned failures = 0, checks = 0;
  auto check = [&](bool value, const char *label) {
    ++checks;
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
  };
  std::fesetround(FE_DOWNWARD);
  std::feraiseexcept(FE_INEXACT | FE_DIVBYZERO);
  const int flags = std::fetestexcept(FE_ALL_EXCEPT);
  errno = EDOM;
  float a[] = {1,2,3,4,5,6}, b[] = {1,2,3,4,5,6};
  const float expected[] = {22,28,49,64};
  ch::Session session({candidate});
  ch::Value av, bv, cv;
  check(bool(session.read(1,{a,2,3,6},av)), "lhs snapshot");
  check(bool(session.read(2,{b,3,2,6},bv)), "rhs snapshot");
  check(bool(session.gemm(3,av,bv,ch::Numeric::strict_f32,cv)), "actual GPU completion");
  const auto report = session.candidateReport();
  check(report.requested == candidate && report.code == ch::Code::ok,
        "forced request/result identity");
  check(report.actual == (nvvm ? ch::Implementation::generated_nvvm
                              : ch::Implementation::generated_rocdl),
        "no CPU/provider substitution");
  check(report.invocation_attempted && report.value_issued && report.actual_threads == 0,
        "GPU candidate completed; not mislabeled CPU worker count");
  check(!report.provider_contract_checked && !report.provider_probe_invoked,
        "GPU is not borrowed OpenBLAS evidence");
  check(cv.valid() && cv.rows()==2 && cv.columns()==2, "private result shape");
  if (cv.valid()) for (unsigned i=0; i<4; ++i) check(cv.data()[i]==expected[i], "rectangular value");
  check(std::fegetround()==FE_DOWNWARD && std::fetestexcept(FE_ALL_EXCEPT)==flags,
        "caller observable FP state preserved");
  check(errno==EDOM, "caller errno preserved");

  ch::Session oversized({candidate});
  ch::Value emptyA, emptyB, old = av;
  check(bool(oversized.read(1,{nullptr,1025,0,0},emptyA)), "empty lhs logical shape");
  check(bool(oversized.read(2,{nullptr,0,1024,0},emptyB)), "empty rhs logical shape");
  check(oversized.gemm(3,emptyA,emptyB,ch::Numeric::strict_f32,old).code ==
        ch::Code::candidate_incompatible, "zero-K cannot bypass recipe qualification");
  check(old.data()==av.data() && old.rows()==2 && old.columns()==3,
        "failed qualification preserves prior logical result");
  check(!oversized.candidateReport().invocation_attempted &&
        !oversized.candidateReport().value_issued, "no oversized kernel or result");
  std::fesetenv(FE_DFL_ENV);
  std::cout << checks << " GPU registry checks; " << failures << " failures\n";
  return failures ? 1 : 0;
}
