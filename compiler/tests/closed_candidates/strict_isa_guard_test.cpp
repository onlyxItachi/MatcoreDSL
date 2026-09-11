#include "closed_host_v1.h"
#include "../../lib/platform/closed_cpu_isa_v1.h"
#include "../support/closed_fp_fixture.h"
#include <array>
#include <cerrno>
#include <cfenv>
#include <iostream>
namespace ch=matcore::mdslc::runtime::closed_host_v1;
namespace p=matcore::mdslc::platform;
namespace {
unsigned checks=0, failures=0, discoveries=0, invocations=0;
unsigned actual_leaf=0;
p::ClosedCpuIsaFactsV1 discovered;
void check(bool okay,const char *what) { ++checks; if(!okay) { ++failures; std::cerr<<"FAIL "<<what<<'\n'; } }
p::ClosedCpuIsaFactsV1 supported() {
  p::ClosedCpuIsaFactsV1 f;
  f.native_linux_x86=f.leaf1_known=f.leaf7_known=f.xcr0_known=true;
  f.leaf1_ecx=p::kClosedAvxEcxV1|p::kClosedAvx512EcxV1;
  f.leaf1_edx=p::kClosedX86EdxV1; f.leaf7_ebx=(1U<<5)|(1U<<16); f.xcr0=0xe6;
  return f;
}
struct Memref { float *allocated,*aligned; std::int64_t offset,sizes[2],strides[2]; };
void leaf(Memref *a,Memref *b,Memref *c,unsigned identity) {
  ++invocations; actual_leaf=identity;
  check(c->aligned!=a->aligned && c->aligned!=b->aligned,"private output disjoint from aliasing inputs");
  for(std::int64_t i=0;i<c->sizes[0];++i) for(std::int64_t j=0;j<c->sizes[1];++j) {
    float sum=0;
    for(std::int64_t k=0;k<a->sizes[1];++k) {
      volatile float product=a->aligned[i*a->strides[0]+k]*b->aligned[k*b->strides[0]+j];
      sum=sum+product;
    }
    c->aligned[i*c->strides[0]+j]=sum;
  }
}
}
// Test-only replacement of direct discovery and leaf definitions. No production
// setter, global override or source-provided capability record exists.
namespace matcore::mdslc::platform {
ClosedCpuIsaFactsV1 discoverClosedCpuIsaFactsV1() noexcept { ++discoveries; return discovered; }
}
extern "C" void _mlir_ciface___matcore_strict_gemm_f32_avx_v1(Memref*a,Memref*b,Memref*c) { leaf(a,b,c,1); }
extern "C" void _mlir_ciface___matcore_strict_gemm_f32_avx2_v1(Memref*a,Memref*b,Memref*c) { leaf(a,b,c,2); }
extern "C" void _mlir_ciface___matcore_strict_gemm_f32_avx512f_v1(Memref*a,Memref*b,Memref*c) { leaf(a,b,c,3); }
int main() {
  std::fenv_t saved; std::fegetenv(&saved);
  std::fesetround(FE_DOWNWARD); std::feraiseexcept(FE_INEXACT|FE_UNDERFLOW);
  closed_fp_fixture::enableFlush();
  unsigned identity=0;
  for(auto candidate:{ch::Candidate::generated_strict_avx,ch::Candidate::generated_strict_avx2,ch::Candidate::generated_strict_avx512f}) {
    ++identity;
    for(bool available:{false,true}) for(auto numeric:{ch::Numeric::strict_f32,ch::Numeric::reassociate_f32})
      for(auto shape:{std::array<std::uint64_t,3>{2,33,2},{0,33,2},{2,0,2},{2,33,0}}) {
        const auto [m,n,k]=shape;
        std::array<float,128> input,out; input.fill(1); out.fill(-9);
        ch::Session s(ch::Options{candidate}); ch::Value a,b,c;
        s.read(1,{input.data(),m,k,128},a); s.read(2,{input.data(),k,n,128},b);
        const auto allocations=s.allocationAttemptsForTesting();
        discovered=available?supported():p::ClosedCpuIsaFactsV1{};
        discoveries=invocations=actual_leaf=0; errno=EDOM;
        const auto fp=closed_fp_fixture::snapshot();
        const auto status=s.gemm(3,a,b,numeric,c);
        check(errno==EDOM && fp==closed_fp_fixture::snapshot(),"preserves exact caller FP state/errno");
        check(discoveries==1,"forced ISA gate checked before empty shortcut");
        check(status.code==(available?ch::Code::ok:ch::Code::candidate_unavailable),"exact forced availability");
        if(!available) {
          check(s.allocationAttemptsForTesting()==allocations && !c.valid() && !invocations &&
                !s.candidateReport().invocation_attempted,"unavailable before allocation/invocation, no fallback");
          check(!s.publish(4,c,{out.data(),m,n,128,ch::Access::read_write}),"failure sticky");
          check(out.front()==-9 && out.back()==-9,"failed publication leaves output unchanged");
        } else {
          check(invocations==(m&&n&&k?1U:0U),"only nonempty positive reduction invokes");
          if(invocations) check(actual_leaf==identity,"exact distinct requested leaf invoked");
          check(bool(s.publish(4,c,{out.data(),m,n,128,ch::Access::read_write})),"supported route publishes");
          for(std::uint64_t i=0;i<m*n;++i) check(out[i]==static_cast<float>(k),"bounded exact arithmetic");
        }
      }
    // A late capability failure cannot erase completed effects or old Value.
    float x=3,out=-1; ch::Session s(ch::Options{candidate}); ch::Value a,c;
    s.read(1,{&x,1,1,1},a); discovered=supported();
    check(bool(s.gemm(2,a,a,ch::Numeric::strict_f32,c)),"prefix computation");
    check(bool(s.publish(3,c,{&out,1,1,1,ch::Access::read_write})),"prefix publication");
    check(bool(s.observe(4,{&out,1,1,1})),"prefix observation");
    const auto old=c; discovered={}; invocations=0;
    const auto failed=s.gemm(5,a,a,ch::Numeric::strict_f32,c);
    check(failed.code==ch::Code::candidate_unavailable && failed.failed_frontier==5 &&
          failed.completed_frontier==4 && failed.completed_effect_frontier==4 &&
          failed.publications==1 && failed.observations==1 && !invocations &&
          c.data()==old.data() && c.data()[0]==9 && out==9,"late failure retains exact effect/value prefix");
  }
  for(auto candidate:{ch::Candidate::native_strict,ch::Candidate::automatic,ch::Candidate::generated_strict}) {
    float x=2; ch::Session s(ch::Options{candidate}); ch::Value a,c;
    s.read(1,{&x,1,1,1},a); discoveries=invocations=0;
    s.gemm(2,a,a,ch::Numeric::strict_f32,c);
    check(!discoveries&&!invocations,"unchosen ISA never probes or invokes");
  }
  std::fesetenv(&saved);
  std::cout<<"Strict ISA guard: "<<checks<<" checks; "<<failures<<" failures\n";
  return failures!=0;
}
