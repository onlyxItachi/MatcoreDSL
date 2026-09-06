#include "closed_host_v1.h"
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <thread>
#include <vector>
#include <xmmintrin.h>
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
namespace ch=matcore::mdslc::runtime::closed_host_v1;
std::uint64_t checks=0,failures=0;
void check(bool ok,const char*what) {++checks;if(!ok){++failures;if(failures<20)std::fprintf(stderr,"FAIL %s\n",what);}}
std::uint32_t bits(float x){return std::isnan(x)?0x7fc00000U:std::bit_cast<std::uint32_t>(x);}
float mul(float a,float b){volatile float p=a*b;return p;}
float add(float a,float b){volatile float q=a+b;return q;}
// Enumerate binary add trees over each original product and exactly one +0,
// optionally contracting a product leaf and its sibling into an FMA. This
// preserves every original multiplication/index and permits all reduction
// permutations. It does not assume one arbitrary numerical oracle is required.
std::set<std::uint32_t> allowed(const float*a,const float*b,unsigned k,unsigned stride){
  const unsigned size=1U<<(k+1U);
  std::vector<std::set<std::uint32_t>> values(size);
  values[1].insert(bits(0));
  for(unsigned p=0;p<k;++p)values[1U<<(p+1U)].insert(bits(mul(a[p],b[p*stride])));
  for(unsigned mask=1;mask<size;++mask){
    if(std::has_single_bit(mask))continue;
    for(unsigned left=(mask-1U)&mask;left;left=(left-1U)&mask){
      const auto right=mask^left;if(!right)continue;
      for(auto l:values[left])for(auto r:values[right])values[mask].insert(bits(add(std::bit_cast<float>(l),std::bit_cast<float>(r))));
      if(std::has_single_bit(left)&&left!=1U){
        const auto p=std::countr_zero(left)-1U;
        for(auto r:values[right])values[mask].insert(bits(std::fma(a[p],b[p*stride],std::bit_cast<float>(r))));
      }
    }
  }
  return values.back();
}
float strict(const float*a,const float*b,unsigned k,unsigned stride){float sum=0;for(unsigned p=0;p<k;++p)sum=add(sum,mul(a[p],b[p*stride]));return sum;}
void simultaneousFirstUse(){
  constexpr int count=8;
  std::barrier start(count);
  std::atomic<int> bad=0,probes=0;
  std::array<std::thread,count> threads;
  for(int t=0;t<count;++t)threads[t]=std::thread([&,t]{
    std::fesetenv(FE_DFL_ENV);
    const std::array<int,4> modes{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO,FE_TONEAREST};
    std::fesetround(modes[t%4]);std::feraiseexcept(FE_INEXACT|FE_DIVBYZERO);
    const auto saved=_mm_getcsr();const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    float a[6]={1,2,3,4,5,6},b[6]={1,0,0,1,1,1},out[4]={};
    ch::Session s(ch::Options{ch::Candidate::authenticated_openblas});ch::Value av,bv,cv;
    if(!s.read(1,{a,2,3,6},av)||!s.read(2,{b,3,2,6},bv)){++bad;std::fprintf(stderr,"worker %d read failed\n",t);}
    start.arrive_and_wait();
    const auto status=s.gemm(3,av,bv,ch::Numeric::reassociate_f32,cv);
    const auto report=s.candidateReport();
    if(!status||!report.provider_contract_checked||!report.value_issued||report.actual!=ch::Implementation::authenticated_openblas||report.actual_threads!=1){++bad;std::fprintf(stderr,"worker %d status=%d checked=%d issued=%d actual=%d threads=%u probe=%d\n",t,int(status.code),report.provider_contract_checked,report.value_issued,int(report.actual),report.actual_threads,report.provider_probe_invoked);}
    if(report.provider_probe_invoked)++probes;
    if(_mm_getcsr()!=saved||std::fegetround()!=modes[t%4]||std::fetestexcept(FE_ALL_EXCEPT)!=flags){++bad;std::fprintf(stderr,"worker %d FP actual=%x saved=%x round=%d wanted=%d flags=%d wanted=%d\n",t,_mm_getcsr(),saved,std::fegetround(),modes[t%4],std::fetestexcept(FE_ALL_EXCEPT),flags);}
    if(!s.publish(4,cv,{out,2,2,4,ch::Access::read_write})||out[0]!=4||out[1]!=5||out[2]!=10||out[3]!=11){++bad;std::fprintf(stderr,"worker %d publish status=%d outputs=%a,%a,%a,%a\n",t,int(s.status().code),double(out[0]),double(out[1]),double(out[2]),double(out[3]));}
  });
  for(auto&t:threads)t.join();
  check(bad==0,"distinct-session first-use numerical/thread/fenv coherence");
  check(probes==1,"exactly one closed provider probe among simultaneous requests");
}
void arithmetic(){
  const float inf=std::numeric_limits<float>::infinity(),tiny=std::numeric_limits<float>::denorm_min();
  const std::array<float,24> pool{0,-0.0F,1,-1,inf,-inf,std::numeric_limits<float>::quiet_NaN(),tiny,-tiny,0x1p-126F,-0x1p-126F,0x1.000002p0F,0x1.fffffep-1F,0x1p100F,-0x1p100F,0x1p-100F,-0x1p-100F,3,-7,0x1.fffffep127F,-0x1.fffffep127F,0x1p-24F,-0x1p-24F,0.5F};
  std::uint64_t state=0x928426D1;
  auto random=[&](){state^=state<<13;state^=state>>7;state^=state<<17;return state;};
  for(unsigned test=0;test<180;++test){
    const unsigned k=1+test%4,m=1+random()%13,n=1+random()%17;
    std::vector<float>a(m*k),b(k*n),out(m*n);
    for(auto&v:a)v=pool[random()%pool.size()];for(auto&v:b)v=pool[random()%pool.size()];
    std::vector<std::set<std::uint32_t>> possibilities;
    for(unsigned i=0;i<m;++i)for(unsigned j=0;j<n;++j)possibilities.push_back(allowed(a.data()+i*k,b.data()+j,k,n));
    for(auto candidate:{ch::Candidate::native_strict,ch::Candidate::generated_strict,ch::Candidate::existing_native,ch::Candidate::authenticated_openblas}){
      const bool exact=candidate==ch::Candidate::native_strict||candidate==ch::Candidate::generated_strict;
      ch::Session s(ch::Options{candidate});ch::Value av,bv,cv;
      check(bool(s.read(1,{a.data(),m,k,a.size()},av)),"random lhs snapshot");check(bool(s.read(2,{b.data(),k,n,b.size()},bv)),"random rhs snapshot");
      check(bool(s.gemm(3,av,bv,exact?ch::Numeric::strict_f32:ch::Numeric::reassociate_f32,cv)),"random candidate ready");
      check(bool(s.publish(4,cv,{out.data(),m,n,out.size(),ch::Access::read_write})),"random publication");
      for(unsigned i=0;i<m;++i)for(unsigned j=0;j<n;++j){
        const bool legal=exact?bits(out[i*n+j])==bits(strict(a.data()+i*k,b.data()+j,k,n)):possibilities[i*n+j].contains(bits(out[i*n+j]));
        check(legal,"K1-K4 expression-family/strict membership");
        if(!legal&&failures<5)std::fprintf(stderr,"test=%u m=%u n=%u k=%u candidate=%u i=%u j=%u actual=%a\n",test,m,n,k,unsigned(candidate),i,j,double(out[i*n+j]));
      }
    }
  }
}
int main(){std::fesetenv(FE_DFL_ENV);simultaneousFirstUse();arithmetic();std::printf("%llu independent candidate arithmetic/concurrency checks, %llu failures\n",(unsigned long long)checks,(unsigned long long)failures);return failures?1:0;}
