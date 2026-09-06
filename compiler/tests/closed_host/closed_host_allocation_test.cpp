#include "closed_host_v1.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>
namespace h = matcore::mdslc::runtime::closed_host_v1;
static bool armed=false;
static unsigned attempts=0, fail_at=0;
void *operator new(std::size_t n) {
  if (armed && ++attempts==fail_at) throw std::bad_alloc();
  if(void *p=std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p,std::size_t) noexcept { std::free(p); }
void operator delete[](void *p,std::size_t) noexcept { std::free(p); }
static unsigned checks=0, failures=0;
static void expect(bool condition,const char *message) {
  ++checks;
  if(!condition) {++failures; std::fprintf(stderr,"FAIL: %s\n",message);}
}
static h::ResourceView v(float *p) { return {p,2,2,4,h::Access::read_write}; }
int main() {
  for(unsigned n=1;n<=24;++n) {
    std::array<float,4>a{1,2,3,4},b{2,0,1,3},c{-1,-1,-1,-1},d{-2,-2,-2,-2};
    h::Session s; h::Value av,bv,cv,dv;
    attempts=0; fail_at=n; armed=true;
    s.read(1,v(a.data()),av);
    s.read(2,v(b.data()),bv);
    s.gemm(3,av,bv,h::Numeric::strict_f32,cv);
    const unsigned before_pub=attempts;
    s.publish(4,cv,v(c.data()));
    const bool publish_allocated=attempts!=before_pub;
    s.observe(5,v(c.data()));
    s.gemm(6,cv,bv,h::Numeric::strict_f32,dv);
    s.publish(7,dv,v(d.data()));
    s.complete(8);
    armed=false;
    const auto st=s.status();
    expect(!publish_allocated,"publication performs no actual C++ allocation");
    expect(st.code==h::Code::ok || st.code==h::Code::allocation_failure,
           "actual allocator failure becomes bounded sticky status");
    expect(c==(st.publications>=1?std::array<float,4>{4,6,10,12}
                                 :std::array<float,4>{-1,-1,-1,-1}),
           "actual OOM preserves first-publication prefix exactly");
    expect(d==(st.publications>=2?std::array<float,4>{14,18,32,36}
                                 :std::array<float,4>{-2,-2,-2,-2}),
           "actual OOM cannot leak a partial second output");
    expect(st.observations<=1 && st.publications<=2,
           "actual OOM cannot corrupt effect counters");
    if(!st) expect(attempts==n,"no later allocation follows actual sticky OOM");
  }
  std::printf("Production adapter actual-allocation sweep: %u checks, %u failures\n",checks,failures);
  return failures?1:0;
}
