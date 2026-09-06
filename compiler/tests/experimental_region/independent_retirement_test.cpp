#include <matcore/region.h>
#include "closed_host_v1.h"
#include <cstdio>
#include <optional>
#include <utility>

namespace h = matcore::mdslc::runtime::closed_host_v1;
namespace m = matcore::mdsl;
static unsigned checks=0, failures=0;
static void check(bool c,const char *m) {
  ++checks; if(!c) { ++failures; std::fprintf(stderr,"FAIL: %s\n",m); }
}
struct Reentry { h::Session *session; std::optional<m::Result> inner; };
static h::Code reenter(h::detail::CandidateInput, h::detail::CandidateInput,
                       h::detail::CandidateOutput out, void *state) {
  auto &r=*static_cast<Reentry *>(state);
  r.inner.emplace(std::move(*r.session).takeResult({"inner",1,2}));
  out.data[0]=99;
  return h::Code::ok;
}
int main() {
  m::Observation retained;
  {
    h::Session s; Reentry r{&s,{}};
    s.configureForTesting({0,reenter,&r});
    float a=2,b=3,published=0; h::Value av,bv,result;
    check(bool(s.read(1,{&a,1,1,1},av)),"read A");
    check(bool(s.read(2,{&b,1,1,1},bv)),"read B");
    check(bool(s.publish(3,av,{&published,1,1,1,h::Access::read_write})),"publish prefix");
    check(bool(s.observe(4,{&published,1,1,1})),"observe prefix");
    check(s.gemm(5,av,bv,h::Numeric::strict_f32,result).code==h::Code::reentrant_use,
          "inner retirement rejects active candidate");
    check(!result.valid() && published==2,"no value issued after reentrant failure");
    check(r.inner && r.inner->error()==h::Code::reentrant_use &&
          r.inner->observation_count()==0,"inner retirement never steals records");
    check(s.observation(0).valid(),"Session still owns prior observation after reentry");
    auto outer=std::move(s).takeResult({"outer",7,8});
    check(!outer && outer.error()==h::Code::reentrant_use &&
          outer.failed_frontier()==5 && outer.completed_frontier()==4 &&
          outer.completed_effect_frontier()==4 && outer.publication_count()==1 &&
          outer.observation_count()==1,"outer failure preserves complete effect prefix");
    retained=outer.observation(0);
    check(retained.valid() && retained.data()[0]==2,"outer retained snapshot intact");
    check(std::move(s).takeResult().error()==h::Code::already_complete,
          "outer retirement consumes exactly once");
    published=123;
  }
  check(retained.valid() && retained.data()[0]==2,"Observation survives Result and Session destruction");
  auto copy=retained;
  retained=m::Observation{};
  check(copy.valid() && copy.data()[0]==2,"copied owning observation independent of original handle");
  h::Session done;
  check(bool(done.complete(1)),"empty completion permitted");
  auto noOutput=std::move(done).takeResult({"ignored",1,1});
  check(noOutput && noOutput.observation_count()==0 && noOutput.failure_location().file==nullptr,
        "successful empty result drops failure site");
  noOutput=std::move(noOutput);
  check(noOutput.ok(),"self-move assignment preserves Result");
  std::printf("Independent Result retirement: %u checks, %u failures\n",checks,failures);
  return failures?1:0;
}
