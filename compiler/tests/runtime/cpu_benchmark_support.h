#ifndef MATCORE_MDSLC_TESTS_CPU_BENCHMARK_SUPPORT_H
#define MATCORE_MDSLC_TESTS_CPU_BENCHMARK_SUPPORT_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace matcore::mdslc::test {

inline constexpr std::size_t kMaxCpuBenchmarkWorkingSetBytes =
    std::size_t{256} * 1024 * 1024;

inline bool checkedCpuBenchmarkWorkingSet(std::size_t lhs_elements,
                                          std::size_t rhs_elements,
                                          std::size_t output_elements,
                                          std::size_t *bytes) noexcept {
  if (bytes == nullptr) return false;
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (output_elements > maximum / 2) return false;
  const std::size_t two_outputs = output_elements * 2;
  if (lhs_elements > maximum - rhs_elements) return false;
  const std::size_t inputs = lhs_elements + rhs_elements;
  if (inputs > maximum - two_outputs) return false;
  const std::size_t elements = inputs + two_outputs;
  if (elements > maximum / sizeof(float)) return false;
  const std::size_t required = elements * sizeof(float);
  if (required > kMaxCpuBenchmarkWorkingSetBytes) return false;
  *bytes = required;
  return true;
}

inline bool closeCpuBenchmarkResult(
    const std::vector<float> &actual,
    const std::vector<float> &reference) noexcept {
  if (actual.size() != reference.size()) return false;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double actual_value = static_cast<double>(actual[index]);
    const double reference_value = static_cast<double>(reference[index]);
    if (!std::isfinite(actual_value) || !std::isfinite(reference_value))
      return false;
    const double difference = std::fabs(actual_value - reference_value);
    const double scale = std::max(1.0, std::fabs(reference_value));
    if (!std::isfinite(difference) || !(difference <= 2.0e-5 * scale))
      return false;
  }
  return true;
}

}  // namespace matcore::mdslc::test

#endif
