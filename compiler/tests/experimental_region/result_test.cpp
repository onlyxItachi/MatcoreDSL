#if defined(MDSLC_RESULT_REVERSE_HEADERS)
#include <matcore/region.h>
#include <matcore/mdsl.h>
#else
#include <matcore/mdsl.h>
#include <matcore/region.h>
#endif
#include "closed_host_v1.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <utility>

namespace mdsl = matcore::mdsl;
namespace host = matcore::mdslc::runtime::closed_host_v1;
static bool reject_allocations = false;
static unsigned allocation_attempts = 0, checks = 0, failures = 0;
void *operator new(std::size_t bytes) {
  if (reject_allocations) { ++allocation_attempts; throw std::bad_alloc(); }
  if (void *p = std::malloc(bytes ? bytes : 1)) return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void check(bool okay, const char *text) {
  ++checks;
  if (!okay) { ++failures; std::fprintf(stderr, "FAIL: %s\n", text); }
}
static_assert(std::is_trivially_copyable_v<mdsl::Value>);
static_assert(std::is_trivially_destructible_v<mdsl::Value>);
static_assert(std::is_trivially_copyable_v<mdsl::Storage>);
static_assert(std::is_nothrow_move_constructible_v<mdsl::Result>);
static_assert(std::is_nothrow_move_assignable_v<mdsl::Result>);
static_assert(!std::is_copy_constructible_v<mdsl::Result>);
static_assert(!std::is_default_constructible_v<mdsl::Result>);
using LegacyGemm = void (*)(mdsl::out_arg, const mdsl::matrix_view &,
                            const mdsl::matrix_view &, mdsl::policy);
using RegionGemm = mdsl::Value (*)(mdsl::Value, mdsl::Value, mdsl::Numerics) noexcept;
static_assert(std::is_same_v<decltype(static_cast<LegacyGemm>(&mdsl::gemm)), LegacyGemm>);
static_assert(std::is_same_v<decltype(static_cast<RegionGemm>(&mdsl::gemm)), RegionGemm>);

mdsl::Result invocation(bool fail, std::array<float, 4> &out) {
  host::Session session;
  host::Value value;
  float a[4]{1, 2, 3, 4};
  check(bool(session.read(1, {a, 2, 2, 4}, value)), "snapshot before result");
  check(bool(session.publish(2, value, {out.data(), 2, 2, 4, host::Access::read_write})),
        "publication before result");
  check(bool(session.observe(3, {out.data(), 2, 2, 4})), "owning observation before result");
  if (fail) {
    check(session.read(4, {a, 2, 2, 3}, value).code == host::Code::insufficient_capacity,
          "late failure after earlier publication");
    check(!session.complete(5), "failure cannot become completed success");
  } else {
    check(bool(session.complete(4)), "completion frontier");
  }
  reject_allocations = true;
  auto result = std::move(session).takeResult({"result.mdsl", 12, 7});
  auto invalid = std::move(session).takeResult();
  check(!invalid && invalid.error() == mdsl::Error::already_complete,
        "invocation result can only be taken once");
  check(invalid.observation_count() == 0, "second take cannot duplicate observation ownership");
  check(!session.publish(6, value, {out.data(), 2, 2, 4, host::Access::read_write}),
        "retired Session cannot publish");
  reject_allocations = false;
  return result;
}
int main() {
  for (bool fail : {false, true}) {
    std::array<float, 4> out{};
    auto result = invocation(fail, out);
    check(result.ok() == !fail, "checked Result success/failure");
    check(result.error() == (fail ? mdsl::Error::insufficient_capacity : mdsl::Error::ok),
          "exact error classification survives transfer");
    check(result.publication_count() == 1 && result.observation_count() == 1,
          "completed effect prefix survives invocation destruction");
    check(fail ? result.failure_location().file != nullptr &&
                     result.failure_location().line == 12 && result.failure_location().column == 7
               : result.failure_location().file == nullptr,
          "source location attached only to failure without allocation");
    auto observation = result.observation(0);
    check(observation.valid() && observation.rows() == 2 && observation.columns() == 2,
          "owning observation geometry");
    out.fill(99);
    check(observation.data()[0] == 1 && observation.data()[3] == 4,
          "observation is immutable snapshot, not current resource view");
    reject_allocations = true;
    auto moved = std::move(result);
    check(!result && result.observation_count() == 0, "moved-from result is invalid and empty");
    check(moved.ok() == !fail && moved.observation(0).data()[2] == 3,
          "Result move preserves status and contents");
    check(!moved.observation(4).valid(), "out-of-range observation is invalid");
    result = std::move(moved);
    reject_allocations = false;
    check(result.ok() == !fail && !moved, "allocation-free move assignment");
  }
  host::Session unfinished;
  auto invalid = std::move(unfinished).takeResult();
  check(!invalid && invalid.error() == mdsl::Error::invalid_frontier,
        "missing completion cannot manufacture success");
  check(allocation_attempts == 0, "result transfer/moves never attempted global allocation");
  std::printf("Experimental owning Result: %u checks, %u failures\n", checks, failures);
  return failures ? 1 : 0;
}
