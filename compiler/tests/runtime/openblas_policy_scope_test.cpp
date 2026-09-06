#include "cpu_openblas.h"
#include <array>
#include <atomic>
#include <barrier>
#include <cfenv>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

namespace runtime = matcore::mdslc::runtime;
namespace planner = matcore::mdslc::planner;
namespace {
std::atomic<int> active{0}, maximum{0}, failures{0};
std::atomic<bool> armed{false}, pause_first{true};
std::mutex wait_mutex;
std::condition_variable entered;
thread_local bool inside_policy = false;
}

extern "C" int __real_openblas_set_num_threads_local(int);
// GNU link --wrap observes the real provider policy boundary; no production
// callback, mock arithmetic or weakened provider conformance. Hold the first
// concurrent scope long enough for all released callers to challenge exclusion.
extern "C" int __wrap_openblas_set_num_threads_local(int count) {
  if (!inside_policy) {
    inside_policy = true;
    const int current = ++active;
    int old = maximum.load();
    while (old < current && !maximum.compare_exchange_weak(old, current)) {}
    entered.notify_all();
    if (armed && pause_first.exchange(false)) {
      std::unique_lock lock(wait_mutex);
      entered.wait_for(lock, std::chrono::milliseconds(200), [] { return active > 1; });
    }
    return __real_openblas_set_num_threads_local(count);
  }
  const int previous = __real_openblas_set_num_threads_local(count);
  inside_policy = false;
  --active;
  return previous;
}

int main() {
  std::fesetenv(FE_DFL_ENV);
  if (!runtime::openblas_conformance_report_v1().conformant) {
    std::cerr << "real linked provider did not pass initial conformance\n";
    return 1;
  }
  constexpr int workers = 8;
  std::barrier start(workers);
  std::array<std::thread, workers> threads;
  armed = true;
  for (auto &thread : threads) thread = std::thread([&] {
    std::fesetenv(FE_DFL_ENV);
    const planner::CpuGemmProblemV1 problem{2, 2, 2};
    const float a[4]{1, 2, 3, 4}, b[4]{2, 1, 1, 2};
    start.arrive_and_wait();
    for (int iteration = 0; iteration < 16; ++iteration) {
      float out[4]{-1, -1, -1, -1}; std::uint32_t actual = 0;
      const auto status = runtime::execute_openblas_gemm_f32_v1(
          problem, a, b, out, 1, &actual);
      if (status != runtime::OpenBlasExecutionStatusV1::success || actual != 1 ||
          out[0] != 4 || out[1] != 5 || out[2] != 10 || out[3] != 11)
        ++failures;
    }
  });
  for (auto &thread : threads) thread.join();
  if (active != 0 || maximum != 1 || failures != 0) {
    std::cerr << "provider policy overlap: maximum=" << maximum
              << " active=" << active << " failed executions=" << failures << '\n';
    return 1;
  }
  std::cout << "128 concurrent requested GEMMs: one policy scope, one thread, correct output\n";
}
