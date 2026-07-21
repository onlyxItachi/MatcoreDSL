#include "cpu_planner.h"
#include "cpu_benchmark_support.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <string_view>
#include <vector>

namespace {

using matcore::mdslc::planner::CpuGemmProblemV1;
using matcore::mdslc::planner::CpuGemmRequestV1;
using matcore::mdslc::planner::CpuPlanStatusV1;

bool parse_positive(const char *text, std::int64_t *value) {
  errno = 0;
  char *end = nullptr;
  const long long parsed = std::strtoll(text, &end, 10);
  if (errno == ERANGE || text == end || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<std::int64_t>::max())
    return false;
  *value = static_cast<std::int64_t>(parsed);
  return true;
}

bool checked_element_count(std::int64_t rows, std::int64_t columns,
                           std::size_t *count) {
  const auto unsigned_rows = static_cast<std::uint64_t>(rows);
  const auto unsigned_columns = static_cast<std::uint64_t>(columns);
  constexpr auto max_elements =
      std::numeric_limits<std::size_t>::max() / sizeof(float);
  if (unsigned_rows > max_elements / unsigned_columns) return false;
  *count = static_cast<std::size_t>(unsigned_rows * unsigned_columns);
  return true;
}

CpuGemmProblemV1 make_problem(std::int64_t m, std::int64_t k, std::int64_t n,
                              const float *a, const float *b,
                              const float *c) {
  std::uint32_t alignment =
      matcore::mdslc::planner::pointer_alignment_bytes(a);
  alignment = std::min(
      alignment, matcore::mdslc::planner::pointer_alignment_bytes(b));
  alignment = std::min(
      alignment, matcore::mdslc::planner::pointer_alignment_bytes(c));
  return {m,
          n,
          k,
          matcore::mdslc::planner::CpuScalarTypeV1::f32,
          matcore::mdslc::planner::CpuScalarTypeV1::f32,
          matcore::mdslc::planner::CpuLayoutV1::row_major_contiguous,
          alignment};
}

double checksum(const std::vector<float> &values) {
  double result = 0.0;
  for (float value : values) result += value;
  return result;
}

constexpr std::string_view compiler_identity() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#elif defined(_MSC_VER)
  return "msvc";
#else
  return "unknown";
#endif
}

}  // namespace

int main(int argc, char **argv) {
  std::int64_t m = 128;
  std::int64_t k = 128;
  std::int64_t n = 128;
  bool guard = false;
  int dimension_argument = 1;
  if (argc > 1 && std::string_view(argv[1]) == "--guard") {
    guard = true;
    ++dimension_argument;
  }
  const int dimension_count = argc - dimension_argument;
  if (dimension_count != 0 && dimension_count != 3) {
    std::cerr << "usage: matcore_cpu_planner_benchmark [--guard] [M K N]\n";
    return 2;
  }
  if (dimension_count == 3 &&
      (!parse_positive(argv[dimension_argument], &m) ||
       !parse_positive(argv[dimension_argument + 1], &k) ||
       !parse_positive(argv[dimension_argument + 2], &n))) {
    std::cerr << "M, K, and N must be positive integers\n";
    return 2;
  }

  std::size_t lhs_elements = 0;
  std::size_t rhs_elements = 0;
  std::size_t output_elements = 0;
  if (!checked_element_count(m, k, &lhs_elements) ||
      !checked_element_count(k, n, &rhs_elements) ||
      !checked_element_count(m, n, &output_elements)) {
    std::cerr << "matrix storage size overflows the address space\n";
    return 2;
  }
  std::size_t working_set_bytes = 0;
  if (!matcore::mdslc::test::checkedCpuBenchmarkWorkingSet(
          lhs_elements, rhs_elements, output_elements, &working_set_bytes)) {
    std::cerr << "benchmark working set exceeds the explicit 256 MiB limit\n";
    return 2;
  }
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> output;
  std::vector<float> reference;
  try {
    a.resize(lhs_elements);
    b.resize(rhs_elements);
    output.resize(output_elements);
    reference.resize(output_elements);
  } catch (const std::bad_alloc &) {
    std::cerr << "benchmark storage allocation failed for "
              << working_set_bytes << " bytes\n";
    return 2;
  }
  for (std::size_t index = 0; index < a.size(); ++index)
    a[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 16.0F;
  for (std::size_t index = 0; index < b.size(); ++index)
    b[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 16.0F;

  const auto capabilities =
      matcore::mdslc::planner::discover_cpu_capabilities_v1();
  const auto gemm_problem =
      make_problem(m, k, n, a.data(), b.data(), output.data());
  const auto reference_plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      gemm_problem, capabilities, CpuGemmRequestV1::force_reference);
  if (!matcore::mdslc::planner::execute_cpu_gemm_plan_v1(
          reference_plan, a.data(), b.data(), reference.data())) {
    std::cerr << "portable reference plan is unavailable\n";
    return 1;
  }

  const auto automatic_plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      gemm_problem, capabilities);
  char diagnostic[2048];
  matcore::mdslc::planner::format_cpu_gemm_plan_v1(
      automatic_plan, diagnostic, sizeof(diagnostic));
  std::cout << "compiler=" << compiler_identity() << " m=" << m
            << " k=" << k << " n=" << n
            << " guard=" << (guard ? "enabled" : "disabled") << '\n';
  std::cout << diagnostic << '\n';

  constexpr int warmup_iterations = 2;
  constexpr int measured_iterations = 9;
  constexpr double guard_max_median_ms = 5000.0;
  constexpr double guard_max_p95_ms = 10000.0;
  const CpuGemmRequestV1 requests[] = {
      CpuGemmRequestV1::force_reference,
      CpuGemmRequestV1::force_tiled,
      CpuGemmRequestV1::force_compiler_vectorized,
  };
  for (const auto request : requests) {
    const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
        gemm_problem, capabilities, request);
    if (plan.status != CpuPlanStatusV1::selected) {
      std::cout << "variant=unavailable reason=" << plan.selection_reason
                << '\n';
      continue;
    }
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
      matcore::mdslc::planner::execute_cpu_gemm_plan_v1(
          plan, a.data(), b.data(), output.data());
    std::vector<double> milliseconds;
    milliseconds.reserve(measured_iterations);
    for (int iteration = 0; iteration < measured_iterations; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      matcore::mdslc::planner::execute_cpu_gemm_plan_v1(
          plan, a.data(), b.data(), output.data());
      const auto end = std::chrono::steady_clock::now();
      milliseconds.push_back(
          std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    const bool correct =
        matcore::mdslc::test::closeCpuBenchmarkResult(output, reference);
    const double median = milliseconds[milliseconds.size() / 2];
    const double p95 = milliseconds.back();
    std::cout << std::fixed << std::setprecision(4)
              << "variant=" << plan.selected_id
              << " median_ms=" << median << " p95_ms=" << p95
              << " checksum=" << checksum(output)
              << " correctness=" << (correct ? "pass" : "FAIL") << '\n';
    if (!correct) return 1;
    if (guard &&
        (median > guard_max_median_ms || p95 > guard_max_p95_ms)) {
      std::cerr << "performance guard failed for " << plan.selected_id
                << ": median/p95 exceeded the generous absolute bound\n";
      return 1;
    }
  }
  return 0;
}
