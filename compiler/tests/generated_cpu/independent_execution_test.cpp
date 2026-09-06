// Independently authored adversarial machine-code oracle. No source admission or
// host adapter contract is inferred from these leaf-only execution checks.
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>
struct M { float *allocated; float *aligned; std::int64_t offset,sizes[2],strides[2]; };
static_assert(std::is_standard_layout_v<M> && sizeof(M)==56 && alignof(M)==8);
static_assert(offsetof(M,aligned)==8 && offsetof(M,offset)==16 && offsetof(M,sizes)==24 && offsetof(M,strides)==40);
extern "C" void _mlir_ciface___matcore_strict_gemm_f32_v1(M*,M*,M*);
M view(float *p,std::int64_t r,std::int64_t c) { return {p,p,0,{r,c},{c,1}}; }
int main() {
  std::fenv_t saved;
  if(fegetenv(&saved)||fesetenv(FE_DFL_ENV)) return 2;
  std::uint32_t state=0x239021;
  auto next=[&] { state^=state<<13; state^=state>>17; state^=state<<5; return state; };
  const float special[]={0,-0.0f,1,-1,0.25f,-0.25f,std::numeric_limits<float>::denorm_min(),
    std::numeric_limits<float>::max(),std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};
  int checks=0,failures=0;
  for(int trial=0;trial<1200;++trial) {
    const int m=next()%8,n=next()%8,k=next()%20;
    std::vector<float> a(m*k),b(k*n),out(m*n+2,-912.0f);
    for(auto &v:a) v=trial%3==0?special[next()%11]:static_cast<float>(static_cast<int>(next()%201)-100)/13.0f;
    for(auto &v:b) v=trial%3==0?special[next()%11]:static_cast<float>(static_cast<int>(next()%201)-100)/17.0f;
    auto av=view(a.empty()?nullptr:a.data(),m,k),bv=view(b.empty()?nullptr:b.data(),k,n),cv=view(out.data()+1,m,n);
    _mlir_ciface___matcore_strict_gemm_f32_v1(&av,&bv,&cv);
    for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
      volatile float sum=0.0f;
      for(int t=0;t<k;++t) { volatile float product=a[i*k+t]*b[t*n+j]; sum=sum+product; }
      const float expected=sum,actual=out[1+i*n+j];
      const bool equal=(std::isnan(expected)&&std::isnan(actual)) || std::bit_cast<std::uint32_t>(expected)==std::bit_cast<std::uint32_t>(actual);
      ++checks; failures+=!equal;
    }
    checks+=2; failures+=out.front()!=-912.0f; failures+=out.back()!=-912.0f;
  }
  fesetenv(&saved);
  std::printf("independent generated leaf: %d checks; %d failures\n",checks,failures);
  return failures!=0;
}

