// This TU intentionally uses a different libstdc++ vector/debug/ABI mode from
// the producer/runtime. Neither standard-library layout crosses the facade.
#include <vector>
#include <matcore/region.h>
#include <array>
#include <cstdio>
#include <thread>
#include <utility>
namespace mdsl = matcore::mdsl;
mdsl::Result produce_observation(float *, bool) noexcept;
int main() {
  unsigned failures = 0;
  auto *function = &produce_observation;
  mdsl::Observation retained;
  for (bool fail : {false, true}) {
    float output[4]{};
    auto result = function(output, fail);
    if (result.ok() == fail || result.observation_count() != 1 ||
        result.publication_count() != 1 || output[3] != 4) ++failures;
    if (fail && (result.error() != mdsl::Error::insufficient_capacity ||
                 result.failure_location().line != 7)) ++failures;
    retained = result.observation(0);
    output[3] = 99;
  }
  if (!retained.valid() || retained.rows() != 2 || retained.columns() != 2 ||
      retained.data()[3] != 4) ++failures;
  std::array<unsigned, 8> bad{};
  std::vector<std::thread> workers;
  for (unsigned i = 0; i < bad.size(); ++i) {
    workers.emplace_back([snapshot = retained, &bad, i] {
      for (unsigned j = 0; j < 1000; ++j) {
        auto one = snapshot;
        mdsl::Observation two;
        two = one;
        auto three = std::move(one);
        two = std::move(three);
        if (!two.valid() || two.data()[2] != 3 || one.valid() || three.valid()) ++bad[i];
      }
    });
  }
  retained = {};
  for (auto &worker : workers) worker.join();
  for (unsigned count : bad) failures += count;
  std::printf("Mixed standard-library host consumer: 8000 concurrent handle cycles, %u failures\n", failures);
  return failures ? 1 : 0;
}
