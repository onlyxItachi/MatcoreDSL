#include "closed_host_v1.h"
#include <array>
#include <thread>
#include <utility>
namespace h=matcore::mdslc::runtime::closed_host_v1;
h::Value opaque_value_from_normal_configuration();
static_assert(sizeof(h::Value)==sizeof(void*));
int main() {
  auto original=opaque_value_from_normal_configuration();
  h::Value retained(original),output(std::move(original));
  if(original.valid())return 1;
  h::Session session;
  if(!session.gemm(1,output,output,h::Numeric::strict_f32,output))return 2;
  std::array<std::thread,4> threads;
  for(auto &thread:threads)thread=std::thread([retained]{
    for(unsigned i=0;i<1000;++i){h::Value a(retained),b(std::move(a));a=b;}
  });
  for(auto &thread:threads)thread.join();
  return output.data()[0]==7 && output.data()[3]==22 && retained.data()[0]==1 &&
    retained.data()[3]==4?0:3;
}
