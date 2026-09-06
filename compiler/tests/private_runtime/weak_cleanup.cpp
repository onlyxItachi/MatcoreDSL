// Ordinary linked C++ conformance fixture, not source-admission authority.
// It preserves the independently discovered observation-unwind counterexample.
#include "closed_host_v1.h"
#include <cstdlib>
#include <new>
namespace host = matcore::mdslc::runtime::closed_host_v1;
static bool tracking = false;
static unsigned allocations = 0, fail_at = 0;
void *operator new(__SIZE_TYPE__ size) {
  if (tracking && ++allocations == fail_at) throw std::bad_alloc();
  if (auto *p = std::malloc(size ? size : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void *p) noexcept { std::free(p); }
void operator delete(void *p, __SIZE_TYPE__) noexcept { std::free(p); }

#ifdef NAMED_DEFINITION
namespace matcore::mdslc::runtime::closed_host_v1 {
struct ObservationAbiV2 { ~ObservationAbiV2(); };
ObservationAbiV2::~ObservationAbiV2() { std::_Exit(72); }
}
#else
extern "C" void forged_cleanup(void *)
    asm("_ZN7matcore5mdslc7runtime14closed_host_v116ObservationAbiV2D2Ev");
extern "C" void forged_cleanup(void *) { std::_Exit(71); }
#endif

static auto invoke(float &input, float &output) {
  host::Session session(host::Options{host::Candidate::generated_strict});
  host::Value a, c;
  session.read(1, {&input, 1, 1, 1, host::Access::read_only}, a);
  session.gemm(2, a, a, host::Numeric::strict_f32, c);
  session.publish(3, c, {&output, 1, 1, 1, host::Access::read_write});
  session.observe(4, {&output, 1, 1, 1, host::Access::read_only});
  session.complete(5);
  return static_cast<host::Session&&>(session).takeResult();
}
int main(int argc, char **) {
  float input = 3, output = -1;
  auto success = invoke(input, output);
  if (!success.ok() || output != 9 || success.publication_count() != 1 ||
      success.observation_count() != 1 || success.observation(0).data()[0] != 9)
    return 1;
  if (argc > 1) return 0; // Archive success control does not traverse cleanup.
  bool saw_failure = false, saw_published_failure = false, saw_success = false;
  for (fail_at = 1; fail_at != 20; ++fail_at) {
    output = -1;
    allocations = 0;
    tracking = true;
    auto result = invoke(input, output);
    tracking = false;
    if (result.ok()) {
      saw_success = true;
      if (output != 9 || result.observation_count() != 1 ||
          result.observation(0).data()[0] != 9) return 2;
    } else {
      saw_failure = true;
      if (result.error() != matcore::mdsl::Error::allocation_failure ||
          result.observation_count() != 0 || result.publication_count() > 1)
        return 3;
      if (result.publication_count()) {
        saw_published_failure = true;
        if (output != 9) return 4;
      } else if (output != -1) return 5;
    }
  }
  return saw_failure && saw_published_failure && saw_success ? 0 : 6;
}
