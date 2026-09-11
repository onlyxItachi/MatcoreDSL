// Synthetic candidate availability proves registry ordering, NOT GPU execution.
#include "closed_host_v1.h"
#include "closed_gpu_capability_v1.h"
#include <iostream>
#include <limits>
#include <vector>
namespace ch = matcore::mdslc::runtime::closed_host_v1;
unsigned invocations = 0;
namespace matcore::mdslc::runtime::closed_host_v1::detail {
Code rocdlCandidateAvailable() noexcept { return Code::ok; }
Code rocdlGemmCandidate(CandidateInput, CandidateInput, CandidateOutput) noexcept {
  ++invocations;
  return Code::candidate_failure;
}
}
int main() {
  unsigned checks=0, failures=0;
  auto check=[&](bool value, const char *what) {
    ++checks;
    if (!value) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
  };
  using ch::detail::closedGpuShapeCompatibleV1;
  check(closedGpuShapeCompatibleV1(1024,1024,64), "inclusive work/output boundaries");
  check(!closedGpuShapeCompatibleV1(1024,1024,65), "work limit");
  check(!closedGpuShapeCompatibleV1(1025,1024,0), "zero-K output limit");
  check(!closedGpuShapeCompatibleV1(0,65536,0), "empty math still qualified");
  check(!closedGpuShapeCompatibleV1(~std::uint64_t{0},~std::uint64_t{0},0), "no overflow authority");
  for (const bool zero : {true,false}) {
    const std::uint64_t m=zero?1025:256, n=zero?1024:256, k=zero?0:1025;
    std::vector<float> a(m*k, 1), b(k*n, 2);
    ch::Session session({ch::Candidate::generated_rocdl});
    ch::Value av,bv,cv;
    float old=19;
    check(bool(session.read(1,{&old,1,1,1},cv)), "old result exists");
    const auto old_data=cv.data();
    check(bool(session.read(2,{a.data(),m,k,a.size()},av)), "lhs read");
    check(bool(session.read(3,{b.data(),k,n,b.size()},bv)), "rhs read");
    const auto allocations=session.allocationAttemptsForTesting();
    check(session.gemm(4,av,bv,ch::Numeric::strict_f32,cv).code ==
          ch::Code::candidate_incompatible, "qualification refuses before candidate");
    check(session.allocationAttemptsForTesting()==allocations, "no private output allocation");
    check(cv.data()==old_data && cv.rows()==1 && cv.data()[0]==19, "old result identity unchanged");
    check(!session.candidateReport().invocation_attempted && !session.candidateReport().value_issued,
          "no false invocation/result evidence");
  }
  check(invocations==0, "leaf never entered outside qualification");
  std::cout << checks << " GPU qualification checks; " << failures << " failures\n";
  return failures?1:0;
}
