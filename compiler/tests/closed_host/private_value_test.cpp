#include "closed_host_v1.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>

namespace h=matcore::mdslc::runtime::closed_host_v1;
static_assert(sizeof(h::Value)==sizeof(void*));
static_assert(alignof(h::Value)==alignof(void*));
static_assert(std::is_nothrow_copy_constructible_v<h::Value> &&
              std::is_nothrow_copy_assignable_v<h::Value> &&
              std::is_nothrow_move_constructible_v<h::Value> &&
              std::is_nothrow_move_assignable_v<h::Value> &&
              std::is_nothrow_destructible_v<h::Value>);

static bool armed=false;
static unsigned attempts=0,fail_at=0,checks=0,failures=0;
void *operator new(std::size_t n) {
  if(armed && ++attempts==fail_at)throw std::bad_alloc();
  if(void *p=std::malloc(n?n:1))return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void *p) noexcept {std::free(p);}
void operator delete[](void *p) noexcept {std::free(p);}
void operator delete(void *p,std::size_t) noexcept {std::free(p);}
void operator delete[](void *p,std::size_t) noexcept {std::free(p);}
void check(bool condition,const char *message) {
  ++checks;if(!condition){++failures;std::fprintf(stderr,"FAIL: %s\n",message);}
}
static bool equals(const h::Value &value,const std::array<float,4> &expected) {
  return value.valid() && value.rows()==2 && value.columns()==2 &&
    std::array<float,4>{value.data()[0],value.data()[1],value.data()[2],value.data()[3]}==expected;
}
int main() {
  const std::array<float,4> original{1,2,3,4},square{7,10,15,22};
  auto input=original;h::Value seed;
  h::Session producer;
  attempts=0;fail_at=0;armed=true;
  const auto read=producer.read(1,{input.data(),2,2,4},seed);
  armed=false;
  check(read && attempts==2,"nonempty read retains exactly owner and element allocation points");
  input.fill(99);
  check(equals(seed,original),"opaque value owns immutable read snapshot");
  attempts=0;fail_at=1;armed=true;
  {
    h::Value copy(seed),assigned;
    assigned=copy;
    h::Value moved(std::move(copy));
    const bool invalid=!copy.valid() && copy.rows()==0 && copy.columns()==0 && copy.data()==nullptr;
    copy=std::move(assigned);
    assigned=assigned;
    assigned=std::move(assigned);
    moved=moved;
    moved=std::move(moved);
    check(invalid && !assigned.valid(),"moved-from handles are empty; empty self operations safe");
    check(equals(copy,original) && equals(moved,original),"copy, move and valid self operations retain contents");
  }
  armed=false;
  check(attempts==0,"opaque ownership copy/move/destruction allocate nothing");
  check(equals(seed,original),"dropping copies does not release remaining owner");
  {
    h::Value output(seed),old(output);
    h::Session compute;
    check(static_cast<bool>(compute.gemm(1,output,output,h::Numeric::strict_f32,output)),
          "same output handle may also be both borrowed immutable inputs");
    check(equals(output,square) && equals(old,original),"replacement retains independent old-value identity");
  }
  for(unsigned n=1;n<=2;++n) {
    h::Value output(seed);
    h::Session compute;
    attempts=0;fail_at=n;armed=true;
    auto status=compute.gemm(1,output,output,h::Numeric::strict_f32,output);
    armed=false;
    check(status.code==h::Code::allocation_failure && attempts==n,
          "each original GEMM allocation point fails with bounded status");
    check(status.failed_frontier==1 && status.completed_frontier==0 && equals(output,original),
          "owner/element allocation failure preserves old output handle and exact frontier");
  }
  {
    h::Session empty;h::Value value;
    attempts=0;fail_at=0;armed=true;
    auto status=empty.read(1,{nullptr,0,8,0},value);
    armed=false;
    check(status && attempts==1 && value.valid() && value.rows()==0 && value.columns()==8,
          "empty value keeps one owner allocation and explicit shape");
  }
  std::atomic<bool> okay{true};
  std::array<std::thread,8> workers;
  for(auto &worker:workers)worker=std::thread([seed,&okay]{
    for(unsigned i=0;i<2000;++i) {
      h::Value a(seed),b(std::move(a));a=b;b=h::Value{};
      if(!equals(a,{1,2,3,4}))okay.store(false,std::memory_order_relaxed);
    }
  });
  for(auto &worker:workers)worker.join();
  check(okay.load() && equals(seed,original),"independent immutable handles safely retain/release across host threads");
  std::printf("Opaque private Value: %u checks, %u failures\n",checks,failures);
  return failures?1:0;
}
