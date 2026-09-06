#include "closed_host_v1.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <utility>

namespace h=matcore::mdslc::runtime::closed_host_v1;
namespace m=matcore::mdsl;
unsigned attempts=0, fail_at=0, checks=0, failures=0;
bool armed=false;
void *operator new(std::size_t n) {
  if(armed && ++attempts==fail_at) throw std::bad_alloc();
  if(auto *p=std::malloc(n?n:1))return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void *p) noexcept {std::free(p);}
void operator delete[](void *p) noexcept {std::free(p);}
void operator delete(void *p,std::size_t) noexcept {std::free(p);}
void operator delete[](void *p,std::size_t) noexcept {std::free(p);}
void check(bool v,const char *why){++checks;if(!v){++failures;std::fprintf(stderr,"FAIL %s\n",why);}}
bool old(const h::Value &v){return v.valid()&&v.rows()==2&&v.columns()==2&&v.data()[0]==1&&v.data()[3]==4;}
int main(){
  float input[4]{1,2,3,4};h::Value seed;
  {h::Session s;check(bool(s.read(1,{input,2,2,4},seed)),"seed read");}
  // Every genuine owned allocation across observation, aliased GEMM and a later
  // observation. Effect prefixes and old handles must survive each failure.
  for(unsigned fail=1;fail<=12;++fail){
    float output[4]{-1,-1,-1,-1};h::Value value(seed);m::Observation saved;
    {
      h::Session s;attempts=0;fail_at=fail;armed=true;
      s.publish(1,seed,{output,2,2,4,h::Access::read_write});
      s.observe(2,{output,2,2,4});
      s.gemm(3,value,value,h::Numeric::strict_f32,value);
      s.publish(4,value,{output,2,2,4,h::Access::read_write});
      s.observe(5,{output,2,2,4});s.complete(6);armed=false;
      auto result=std::move(s).takeResult();
      const auto status=s.status();
      check(old(seed),"independent original survives all result replacements");
      if(fail<=4){
        check(!result&&status.failed_frontier==2&&status.completed_frontier==1,"first observation allocation prefix");
        check(result.publication_count()==1&&result.observation_count()==0&&output[0]==1&&old(value),"first write retained before failed observation");
      }else if(fail<=6){
        check(!result&&status.failed_frontier==3&&status.completed_frontier==2,"GEMM allocation prefix");
        check(result.publication_count()==1&&result.observation_count()==1&&output[0]==1&&old(value),"output aliases inputs but failed replacement leaves old identity");
      }else if(fail<=9){
        check(!result&&status.failed_frontier==5&&status.completed_frontier==4,"late observation allocation prefix");
        check(result.publication_count()==2&&result.observation_count()==1&&output[0]==7&&value.data()[3]==22,"later write retained after late observation failure");
      }else{
        check(result&&status.completed_frontier==6&&result.publication_count()==2&&result.observation_count()==2,"fully completed ordered effects");
      }
      if(result.observation_count()){
        saved=result.observation(0);
        check(saved.data()[0]==1&&saved.data()[3]==4,"earlier observation retains original snapshot after output update");
      }
    }
    if(saved.valid())check(saved.data()[0]==1&&saved.data()[3]==4,"observation outlives result and session");
  }
  h::Value retained;
  {h::Session s;float observed[4]{1,2,3,4};
    for(unsigned i=1;i<=33;++i){check(bool(s.observe(i,{observed,2,2,4})),"observation vector growth");if(i==1)retained=s.observation(0);observed[0]+=1;}
  }
  check(old(retained),"private value outlives observation block after repeated growth");
  std::atomic<unsigned> wrong{0};std::array<std::thread,8> workers;
  for(auto &worker:workers)worker=std::thread([retained,&wrong]{
    for(unsigned i=0;i<4000;++i){h::Value a(retained),b(a),c(std::move(a));a=std::move(b);b=c;c=h::Value{};if(!old(a)||!old(b))++wrong;}
  });
  retained=h::Value{};
  for(auto &worker:workers)worker.join();
  check(wrong.load()==0,"last owners retained and released on independent host threads");
  std::printf("Independent private Value: %u checks, 32000 ownership cycles, %u failures\n",checks,failures);
  return failures?1:0;
}
