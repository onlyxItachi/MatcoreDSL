#include <matcore/region.h>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <utility>
#include <vector>
namespace m=matcore::mdsl;
m::Result issue(bool twice);
static std::atomic<unsigned> calls{0},failAt{0},checks{0},failures{0};
void *operator new(std::size_t size) {
  const auto count=++calls;
  if(failAt && count==failAt) throw std::bad_alloc();
  if(auto *p=std::malloc(size?size:1)) return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p,std::size_t) noexcept { std::free(p); }
void operator delete[](void *p,std::size_t) noexcept { std::free(p); }
static void check(bool c,const char *m) {
  ++checks;if(!c) {++failures;std::fprintf(stderr,"FAIL: %s\n",m);}
}
int main() {
  // This host TU uses _GLIBCXX_DEBUG; the producer and runtime do not.
  std::vector<int> hostDebug{1,2};
  check(hostDebug.at(1)==2,"host debug STL works");
  for(bool twice:{false,true}) {
    calls=0; auto good=issue(twice); const auto normalCalls=calls.load();
    check(good.ok() && good.observation_count()==(twice?2:1),"ordinary helper result crosses debug-host boundary");
    for(unsigned allocation=1;allocation<=normalCalls;++allocation) {
      calls=0;failAt=allocation;
      auto failed=issue(twice);
      failAt=0;
      check(!failed && failed.error()==m::Error::allocation_failure,"every producer allocation fails closed");
      const auto frontier=failed.failed_frontier();
      check(frontier==1 || frontier==3 || (twice && frontier==5),"OOM only at fallible read or observe frontier");
      check(failed.publication_count()==(frontier==1?0:frontier==3?1:2),"OOM preserves exact prior publication prefix");
      check(failed.observation_count()==(frontier==5?1:0),"new block/vector OOM never retires failed observation");
      if(frontier==5) check(failed.observation(0).data()[0]==7,"earlier observation survives later record-growth failure");
    }
  }
  m::Observation survivor;
  {
    auto r=issue(true);survivor=r.observation(1);
    auto moved=std::move(r);
    check(!r && moved.ok(),"opaque Result move invalidates old owner");
    std::array<std::thread,8> workers;
    for(auto &t:workers) t=std::thread([copy=survivor] {
      for(unsigned i=0;i<2000;++i) {
        auto local=copy;
        m::Observation another;
        another=local;
        local=std::move(another);
        check(local.valid() && local.rows()==1 && local.columns()==1 && local.data()[0]==7,
              "independent owning handles concurrently retain/read/release immutable block");
      }
    });
    for(auto &t:workers)t.join();
  }
  check(survivor.valid() && survivor.data()[0]==7,"last observation owns block after Result destruction");
  std::printf("Opaque Result mixed-STL/OOM/concurrency: %u checks, %u failures\n",checks.load(),failures.load());
  return failures?1:0;
}
