#include "closed_host_v1.h"

#include <array>
#include <iostream>
#include <type_traits>

namespace ch = matcore::mdslc::runtime::closed_host_v1;
static_assert(!std::is_copy_constructible_v<ch::Session>);
static_assert(!std::is_move_constructible_v<ch::Session>);
static_assert(std::is_nothrow_copy_constructible_v<ch::Value>);

int main() {
  // This target links the production object, with no test macro or callback
  // injection export. It tests the actual built-in strict scalar candidate.
  std::array<float,6> a{1,2,3,4,5,6}, b{2,3,1,4,2,1};
  std::array<float,4> c{-1,-1,-1,-1};
  ch::Session session;
  ch::Value lhs,rhs,result;
#if !defined(__linux__) || !defined(__x86_64__)
  const auto status = session.read(1,{},lhs);
  return status.code == ch::Code::unsupported_fp_environment ? 0 : 1;
#else
  if (!session.read(1,{a.data(),2,3,6,ch::Access::read_only},lhs) ||
      !session.read(2,{b.data(),3,2,6,ch::Access::read_only},rhs) ||
      !session.gemm(3,lhs,rhs,ch::Numeric::strict_f32,result) ||
      !session.publish(4,result,{c.data(),2,2,4,ch::Access::read_write}) ||
      !session.observe(5,{c.data(),2,2,4,ch::Access::read_only}) ||
      !session.complete(6))
    return 1;
  const auto status = session.status();
  if (c != std::array<float,4>{10,14,25,38} ||
      status.publications != 1 || status.observations != 1 ||
      status.completed_frontier != 6 || status.completed_effect_frontier != 5 ||
      !status.completed || !session.observation(0).valid())
    return 1;
  std::cout << "production scalar adapter source-free API execution passed\n";
  return 0;
#endif
}
