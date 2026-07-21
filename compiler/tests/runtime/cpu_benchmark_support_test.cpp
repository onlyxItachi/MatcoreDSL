#include "cpu_benchmark_support.h"

#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

int main() {
  using matcore::mdslc::test::checkedCpuBenchmarkWorkingSet;
  using matcore::mdslc::test::closeCpuBenchmarkResult;

  int failures = 0;
  const auto expect = [&failures](bool condition, const char *message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  };

  expect(closeCpuBenchmarkResult({1.0F, -2.0F}, {1.0F, -2.0F}),
         "equal finite results pass");
  expect(!closeCpuBenchmarkResult({1.0F}, {2.0F}),
         "finite mismatches fail");
  expect(!closeCpuBenchmarkResult(
             {std::numeric_limits<float>::quiet_NaN()}, {0.0F}),
         "NaN output fails");
  expect(!closeCpuBenchmarkResult(
             {0.0F}, {std::numeric_limits<float>::quiet_NaN()}),
         "NaN reference fails");
  expect(!closeCpuBenchmarkResult(
             {std::numeric_limits<float>::infinity()},
             {std::numeric_limits<float>::infinity()}),
         "infinite values fail even when their signs match");

  std::size_t bytes = 0;
  expect(checkedCpuBenchmarkWorkingSet(6, 6, 4, &bytes) &&
             bytes == 20 * sizeof(float),
         "small benchmark working set is counted exactly");
  expect(!checkedCpuBenchmarkWorkingSet(100'000'000, 1, 1, &bytes),
         "oversized benchmark working set is rejected");
  expect(!checkedCpuBenchmarkWorkingSet(
             std::numeric_limits<std::size_t>::max(), 1, 1, &bytes),
         "working-set arithmetic overflow is rejected");
  expect(!checkedCpuBenchmarkWorkingSet(1, 1, 1, nullptr),
         "null byte-count output is rejected");

  if (failures != 0) return 1;
  std::cout << "CPU benchmark support PASS\n";
  return 0;
}
