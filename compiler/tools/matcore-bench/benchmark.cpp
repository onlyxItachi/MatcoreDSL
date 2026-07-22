#include "benchmark.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <new>
#include <numeric>
#include <ostream>
#include <sstream>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace matcore::mdslc::bench {
namespace {

constexpr std::uint64_t kColdCacheBytes = UINT64_C(64) * 1024 * 1024;
constexpr std::uint64_t kFullOracleOperationLimit = UINT64_C(16) * 1024 * 1024;
constexpr std::uint64_t kFullOracleElementLimit = UINT64_C(256) * 1024;
constexpr std::size_t kSampledOracleElements = 64;
constexpr std::uint64_t kMaximumAggregateRepetitions = 1'000'000;

bool checked_add(std::uint64_t lhs, std::uint64_t rhs,
                 std::uint64_t &result) noexcept {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) return false;
  result = lhs + rhs;
  return true;
}

bool checked_multiply(std::uint64_t lhs, std::uint64_t rhs,
                      std::uint64_t &result) noexcept {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool shape_counts(const GemmShapeV1 &shape, std::uint64_t &lhs,
                  std::uint64_t &rhs, std::uint64_t &output,
                  std::uint64_t &work) noexcept {
  if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0) return false;
  const auto m = static_cast<std::uint64_t>(shape.m);
  const auto n = static_cast<std::uint64_t>(shape.n);
  const auto k = static_cast<std::uint64_t>(shape.k);
  return checked_multiply(m, k, lhs) && checked_multiply(k, n, rhs) &&
         checked_multiply(m, n, output) && checked_multiply(output, k, work);
}

bool use_full_oracle(const GemmShapeV1 &shape) noexcept {
  std::uint64_t lhs = 0, rhs = 0, output = 0, work = 0;
  return shape_counts(shape, lhs, rhs, output, work) &&
         output <= kFullOracleElementLimit &&
         work <= kFullOracleOperationLimit;
}

class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  AlignedBuffer(std::size_t bytes, std::uint32_t alignment,
                bool exact_alignment = false) {
    reset(bytes, alignment, exact_alignment);
  }

  void reset(std::size_t bytes, std::uint32_t alignment,
             bool exact_alignment = false) {
    bytes_ = bytes;
    if (bytes == 0) {
      storage_.clear();
      data_ = nullptr;
      return;
    }
    const std::size_t usable_alignment = std::max<std::size_t>(alignment, 1);
    const std::size_t extra = usable_alignment * 2 + alignof(std::max_align_t);
    if (bytes > std::numeric_limits<std::size_t>::max() - extra)
      throw std::bad_alloc();
    storage_.assign(bytes + extra, std::byte{0});
    const auto begin = reinterpret_cast<std::uintptr_t>(storage_.data());
    std::uintptr_t candidate =
        (begin + usable_alignment - 1) & ~(usable_alignment - 1);
    if (exact_alignment && usable_alignment < 64 &&
        candidate % (usable_alignment * 2) == 0)
      candidate += usable_alignment;
    data_ = reinterpret_cast<std::byte *>(candidate);
  }

  std::byte *data() noexcept { return data_; }
  const std::byte *data() const noexcept { return data_; }
  std::size_t size() const noexcept { return bytes_; }
  std::span<std::byte> span() noexcept { return {data_, bytes_}; }

 private:
  std::vector<std::byte> storage_;
  std::byte *data_ = nullptr;
  std::size_t bytes_ = 0;
};

std::uint64_t splitmix64(std::uint64_t &state) noexcept {
  state += UINT64_C(0x9e3779b97f4a7c15);
  std::uint64_t value = state;
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

void fill_input(float *data, std::size_t count, std::uint64_t &state) noexcept {
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint32_t encoded =
        static_cast<std::uint32_t>(splitmix64(state) >> 40);
    data[index] = static_cast<float>(static_cast<std::int32_t>(encoded) -
                                     INT32_C(0x00800000)) /
                  static_cast<float>(INT32_C(0x01000000));
  }
}

double expected_element(const GemmShapeV1 &shape, const float *lhs,
                        const float *rhs, std::uint64_t linear_index,
                        double &absolute_product_sum) noexcept {
  const auto n = static_cast<std::uint64_t>(shape.n);
  const auto k = static_cast<std::uint64_t>(shape.k);
  const std::uint64_t row = linear_index / n;
  const std::uint64_t column = linear_index % n;
  double sum = 0.0;
  absolute_product_sum = 0.0;
  for (std::uint64_t p = 0; p < k; ++p) {
    const double product =
        static_cast<double>(lhs[row * k + p]) *
        static_cast<double>(rhs[p * n + column]);
    sum += product;
    absolute_product_sum += std::fabs(product);
  }
  return sum;
}

std::vector<std::uint64_t> oracle_indices(const GemmShapeV1 &shape,
                                          bool full) {
  const std::uint64_t output_count =
      static_cast<std::uint64_t>(shape.m) * static_cast<std::uint64_t>(shape.n);
  if (full) {
    std::vector<std::uint64_t> result(static_cast<std::size_t>(output_count));
    std::iota(result.begin(), result.end(), UINT64_C(0));
    return result;
  }
  const std::uint64_t count =
      std::min<std::uint64_t>(output_count, kSampledOracleElements);
  std::vector<std::uint64_t> result;
  result.reserve(static_cast<std::size_t>(count));
  if (count == 1) return {0};
  for (std::uint64_t sample = 0; sample < count; ++sample)
    result.push_back(sample * (output_count - 1) / (count - 1));
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

CorrectnessResultV1 verify_output(const GemmShapeV1 &shape, const float *lhs,
                                  const float *rhs, const float *output) {
  CorrectnessResultV1 result;
  const bool full = use_full_oracle(shape);
  result.oracle_mode = full ? "full-double" : "sampled-double-plus-checksum";
  const std::uint64_t output_count =
      static_cast<std::uint64_t>(shape.m) * static_cast<std::uint64_t>(shape.n);
  result.checked_elements = output_count;
  for (std::uint64_t index = 0; index < output_count; ++index) {
    if (!std::isfinite(output[index])) {
      result.reason = "output contains NaN or infinity";
      return result;
    }
    result.checksum += static_cast<double>(output[index]);
  }

  const auto indices = oracle_indices(shape, full);
  const double epsilon = static_cast<double>(std::numeric_limits<float>::epsilon());
  const double k_epsilon = static_cast<double>(shape.k) * epsilon;
  const double gamma = k_epsilon < 0.5 ? k_epsilon / (1.0 - k_epsilon) : 1.0;
  for (const std::uint64_t index : indices) {
    double absolute_product_sum = 0.0;
    const double expected =
        expected_element(shape, lhs, rhs, index, absolute_product_sum);
    const double difference = std::fabs(static_cast<double>(output[index]) - expected);
    const double allowed = std::max(1.0e-6, 8.0 * gamma * absolute_product_sum);
    result.maximum_absolute_error =
        std::max(result.maximum_absolute_error, difference);
    result.maximum_allowed_error = std::max(result.maximum_allowed_error, allowed);
    if (!(difference <= allowed)) {
      result.reason = "double-precision element oracle mismatch";
      return result;
    }
  }

  const auto m = static_cast<std::uint64_t>(shape.m);
  const auto n = static_cast<std::uint64_t>(shape.n);
  const auto k = static_cast<std::uint64_t>(shape.k);
  double checksum_bound = 0.0;
  for (std::uint64_t p = 0; p < k; ++p) {
    double lhs_sum = 0.0, rhs_sum = 0.0;
    double lhs_abs = 0.0, rhs_abs = 0.0;
    for (std::uint64_t i = 0; i < m; ++i) {
      const double value = static_cast<double>(lhs[i * k + p]);
      lhs_sum += value;
      lhs_abs += std::fabs(value);
    }
    for (std::uint64_t j = 0; j < n; ++j) {
      const double value = static_cast<double>(rhs[p * n + j]);
      rhs_sum += value;
      rhs_abs += std::fabs(value);
    }
    result.expected_checksum += lhs_sum * rhs_sum;
    checksum_bound += lhs_abs * rhs_abs;
  }
  const double checksum_allowed =
      std::max(1.0e-5, 8.0 * gamma * checksum_bound);
  if (!(std::fabs(result.checksum - result.expected_checksum) <=
        checksum_allowed)) {
    result.reason = "independent double-precision checksum mismatch";
    return result;
  }
  result.passed = true;
  result.reason = "independent double-precision oracle passed";
  return result;
}

void evict_cache(std::vector<std::byte> &buffer) noexcept {
  std::uint8_t checksum = 0;
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    const auto value = static_cast<std::uint8_t>(buffer[index]);
    checksum ^= value;
    buffer[index] = static_cast<std::byte>(value + 1U);
  }
  std::atomic_signal_fence(std::memory_order_seq_cst);
  if (checksum == UINT8_C(0xff) && !buffer.empty()) buffer[0] = std::byte{0};
}

std::string json_escape(std::string_view input) {
  std::ostringstream output;
  for (const unsigned char character : input) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

void json_string(std::ostream &output, std::string_view value) {
  output << '"' << json_escape(value) << '"';
}

std::string read_first_matching_line(const char *path,
                                     std::string_view prefix) {
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    if (line.starts_with(prefix)) {
      const auto separator = line.find(':');
      if (separator == std::string::npos) return line;
      const auto first = line.find_first_not_of(" \t", separator + 1);
      return first == std::string::npos ? std::string{} : line.substr(first);
    }
  }
  return "unknown";
}

std::string read_file_trimmed(const char *path) {
  std::ifstream input(path);
  std::string value;
  std::getline(input, value);
  while (!value.empty() &&
         (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' ||
          value.back() == '\t'))
    value.pop_back();
  return value.empty() ? "unknown" : value;
}

}  // namespace

std::string_view cache_mode_name_v1(CacheModeV1 mode) noexcept {
  return mode == CacheModeV1::cold ? "cold" : "hot";
}
std::string_view allocation_mode_name_v1(AllocationModeV1 mode) noexcept {
  return mode == AllocationModeV1::include_allocation ? "include-allocation"
                                                       : "reuse-workspace";
}
std::string_view packing_mode_name_v1(PackingModeV1 mode) noexcept {
  switch (mode) {
    case PackingModeV1::include: return "include-packing";
    case PackingModeV1::exclude: return "exclude-packing";
    case PackingModeV1::prepack_b: return "prepacked-b";
  }
  return "unknown";
}
std::string_view profile_name_v1(ProfileV1 profile) noexcept {
  switch (profile) {
    case ProfileV1::custom: return "custom";
    case ProfileV1::quick: return "quick";
    case ProfileV1::standard: return "standard";
    case ProfileV1::full: return "full";
  }
  return "unknown";
}

std::vector<GemmShapeV1> quick_profile_v1() {
  return {{1, 1, 1},       {2, 3, 2},       {16, 16, 16},
          {33, 35, 37},    {64, 7, 19},     {127, 129, 131},
          {128, 128, 128}, {256, 256, 256}};
}

std::vector<GemmShapeV1> standard_profile_v1() {
  std::vector<GemmShapeV1> result;
  for (const std::int64_t size : {4, 8, 16, 24, 32, 48, 64, 96, 128, 192,
                                  256, 384, 512, 768, 1024, 1536, 2048})
    result.push_back({size, size, size});
  const GemmShapeV1 rectangular[] = {
      {1, 4096, 4096},    {8, 4096, 4096},    {32, 4096, 4096},
      {4096, 32, 4096},  {4096, 4096, 64},   {64, 4096, 4096},
      {1024, 256, 4096}, {256, 1024, 4096},
  };
  result.insert(result.end(), std::begin(rectangular), std::end(rectangular));
  const GemmShapeV1 tails[] = {{31, 33, 35},   {63, 65, 67},
                               {127, 129, 131}, {255, 257, 259},
                               {511, 513, 515}};
  result.insert(result.end(), std::begin(tails), std::end(tails));
  return result;
}

std::vector<GemmShapeV1> full_profile_v1() {
  auto result = standard_profile_v1();
  result.push_back({4096, 4096, 4096});
  result.push_back({8192, 8192, 8192});
  return result;
}

bool validate_options_v1(BenchmarkOptionsV1 &options, std::string &error) {
  if (options.shapes.empty()) {
    switch (options.profile) {
      case ProfileV1::quick: options.shapes = quick_profile_v1(); break;
      case ProfileV1::standard: options.shapes = standard_profile_v1(); break;
      case ProfileV1::full: options.shapes = full_profile_v1(); break;
      case ProfileV1::custom:
        error = "custom profile requires --m, --n, and --k";
        return false;
    }
  }
  for (const auto &shape : options.shapes) {
    std::uint64_t lhs = 0, rhs = 0, output = 0, work = 0;
    if (!shape_counts(shape, lhs, rhs, output, work)) {
      error = "M, N, and K must be positive and fit uint64 arithmetic";
      return false;
    }
  }
  if (options.requested_threads == 0) {
    error = "thread count must be positive";
    return false;
  }
  if (options.measured_iterations == 0) {
    error = "measured iteration count must be positive";
    return false;
  }
  if (options.alignment_bytes < alignof(float) ||
      !std::has_single_bit(options.alignment_bytes)) {
    error = "alignment must be a power of two and at least 4 bytes";
    return false;
  }
  if (options.maximum_memory_bytes == 0) {
    error = "maximum memory limit must be positive";
    return false;
  }
  if (options.timer_floor_nanoseconds == 0) {
    error = "timer floor must be positive";
    return false;
  }
  if (options.packing_mode == PackingModeV1::prepack_b &&
      options.allocation_mode == AllocationModeV1::include_allocation) {
    error = "prepacked-B mode requires reusable workspace";
    return false;
  }
  return true;
}

bool checked_memory_requirement_v1(const GemmShapeV1 &shape,
                                   const RunnerPlanV1 &plan,
                                   const BenchmarkOptionsV1 &options,
                                   std::uint64_t &required_bytes,
                                   std::string &error) {
  std::uint64_t lhs = 0, rhs = 0, output = 0, work = 0;
  if (!shape_counts(shape, lhs, rhs, output, work)) {
    error = "matrix element count overflows";
    return false;
  }
  std::uint64_t elements = 0;
  if (!checked_add(lhs, rhs, elements) || !checked_add(elements, output, elements) ||
      !checked_multiply(elements, sizeof(float), required_bytes)) {
    error = "matrix byte count overflows";
    return false;
  }
  const std::uint64_t alignment_slack =
      static_cast<std::uint64_t>(options.alignment_bytes) * 6;
  if (!checked_add(required_bytes, alignment_slack, required_bytes) ||
      !checked_add(required_bytes, plan.workspace_bytes, required_bytes)) {
    error = "workspace byte count overflows";
    return false;
  }
  std::uint64_t workspace_slack = 0;
  if (!checked_multiply(std::max<std::uint32_t>(plan.workspace_alignment, 1U),
                        UINT64_C(2), workspace_slack) ||
      !checked_add(required_bytes, workspace_slack, required_bytes)) {
    error = "workspace alignment slack overflows";
    return false;
  }
  if (options.packing_mode == PackingModeV1::prepack_b &&
      (!checked_add(required_bytes, plan.prepacked_b_bytes, required_bytes) ||
       !checked_add(required_bytes, workspace_slack, required_bytes))) {
      error = "prepacked-B byte count overflows";
      return false;
  }
  if (options.cache_mode == CacheModeV1::cold &&
      !checked_add(required_bytes, kColdCacheBytes, required_bytes)) {
    error = "cold-cache buffer byte count overflows";
    return false;
  }
  if (required_bytes > options.maximum_memory_bytes) {
    error = "estimated benchmark allocation exceeds --max-memory-mib";
    return false;
  }
  if (required_bytes > std::numeric_limits<std::size_t>::max()) {
    error = "estimated benchmark allocation exceeds size_t";
    return false;
  }
  return true;
}

TimingStatisticsV1 summarize_timings_v1(std::vector<double> samples_seconds,
                                        std::uint64_t aggregate_repetitions,
                                        std::uint64_t timer_floor_nanoseconds) {
  TimingStatisticsV1 result;
  result.aggregate_repetitions = aggregate_repetitions;
  if (samples_seconds.empty() || aggregate_repetitions == 0) {
    result.rejection_reason = "no timing samples";
    return result;
  }
  for (const double sample : samples_seconds) {
    if (!std::isfinite(sample) || sample <= 0.0) {
      result.rejection_reason = "timer produced a non-positive sample";
      return result;
    }
  }
  std::sort(samples_seconds.begin(), samples_seconds.end());
  result.minimum_seconds = samples_seconds.front();
  result.median_seconds = samples_seconds[samples_seconds.size() / 2];
  const std::size_t p95_index =
      static_cast<std::size_t>(std::ceil(samples_seconds.size() * 0.95)) - 1;
  result.p95_seconds = samples_seconds[std::min(p95_index, samples_seconds.size() - 1)];
  for (const double sample : samples_seconds) {
    const double aggregate_ns =
        sample * static_cast<double>(aggregate_repetitions) * 1.0e9;
    if (aggregate_ns < static_cast<double>(timer_floor_nanoseconds)) {
      result.rejection_reason = "aggregate interval is below the timer floor";
      return result;
    }
  }
  result.valid = true;
  return result;
}

BenchmarkEnvironmentV1 discover_benchmark_environment_v1(
    const RunnerEnvironmentV1 &runner_environment) {
  BenchmarkEnvironmentV1 result;
#if defined(__linux__)
  result.os_family = "linux";
#elif defined(_WIN32)
  result.os_family = "windows";
#else
  result.os_family = "unknown";
#endif
#if defined(__x86_64__) || defined(_M_X64)
  result.architecture = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  result.architecture = "aarch64";
#else
  result.architecture = "unknown";
#endif
#if defined(__clang__)
  result.compiler = "clang " __clang_version__;
#elif defined(__GNUC__)
  result.compiler = "gcc " __VERSION__;
#elif defined(_MSC_VER)
  result.compiler = "msvc " + std::to_string(_MSC_VER);
#else
  result.compiler = "unknown";
#endif
#ifdef MATCORE_BENCH_COMPILER_FLAGS
  result.compiler_flags = MATCORE_BENCH_COMPILER_FLAGS;
#else
  result.compiler_flags = "unknown";
#endif
#ifdef MATCORE_BENCH_BUILD_TYPE
  result.build_type = MATCORE_BENCH_BUILD_TYPE;
#else
  result.build_type = "unknown";
#endif
#ifdef MATCORE_BENCH_SOURCE_COMMIT
  result.source_commit = MATCORE_BENCH_SOURCE_COMMIT;
#else
  result.source_commit = "unknown";
#endif
#if defined(__linux__)
  result.cpu_model = read_first_matching_line("/proc/cpuinfo", "model name");
  result.governor = read_file_trimmed(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  const std::string minimum_frequency = read_file_trimmed(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq");
  const std::string maximum_frequency = read_file_trimmed(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq");
  result.frequency_policy = "min_khz=" + minimum_frequency +
                            " max_khz=" + maximum_frequency;
  const std::string boost =
      read_file_trimmed("/sys/devices/system/cpu/cpufreq/boost");
  result.boost_state =
      boost == "1" ? "enabled" : boost == "0" ? "disabled" : boost;
  result.cpu_affinity =
      read_first_matching_line("/proc/self/status", "Cpus_allowed_list");
#else
  result.cpu_model = "unknown";
  result.governor = "unknown";
  result.frequency_policy = "unknown";
  result.boost_state = "unknown";
  result.cpu_affinity = "unknown";
#endif
  result.hardware_threads = std::thread::hardware_concurrency();
  result.timer_source = "std::chrono::steady_clock";
  const long double resolution_ns =
      static_cast<long double>(std::chrono::steady_clock::period::num) * 1.0e9L /
      static_cast<long double>(std::chrono::steady_clock::period::den);
  result.timer_resolution_nanoseconds =
      static_cast<std::uint64_t>(std::max<long double>(1.0L, std::ceil(resolution_ns)));
  result.timestamp_unix_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count();
  result.runner = runner_environment;
  return result;
}

bool run_benchmarks_v1(const BenchmarkOptionsV1 &unvalidated_options,
                       const GemmRunnerV1 &runner,
                       BenchmarkReportV1 &report, std::string &error) {
  BenchmarkOptionsV1 options = unvalidated_options;
  if (!validate_options_v1(options, error)) return false;
  report = {};
  report.options = options;
  report.environment = discover_benchmark_environment_v1(runner.environment());

  for (const auto &shape : options.shapes) {
    BenchmarkResultV1 result;
    result.shape = shape;
    result.requested_variant = options.requested_variant;
    result.cache_mode = options.cache_mode;
    result.allocation_mode = options.allocation_mode;
    result.packing_mode = options.packing_mode;
    result.plan = runner.plan(shape, options.alignment_bytes,
                              options.requested_threads,
                              options.requested_variant);
    if (!result.plan.legal) {
      error = "variant planning failed for " + std::to_string(shape.m) + "x" +
              std::to_string(shape.n) + "x" + std::to_string(shape.k) + ": " +
              result.plan.reason;
      return false;
    }
    if (result.plan.workspace_alignment == 0 ||
        !std::has_single_bit(result.plan.workspace_alignment)) {
      error = "runner returned an invalid workspace alignment";
      return false;
    }
    if (options.packing_mode == PackingModeV1::prepack_b &&
        !result.plan.supports_prepacked_b) {
      error = "selected variant does not support prepacked-B execution";
      return false;
    }
    if (!checked_memory_requirement_v1(shape, result.plan, options,
                                       result.estimated_memory_bytes, error))
      return false;

    std::uint64_t lhs_elements = 0, rhs_elements = 0, output_elements = 0, work = 0;
    if (!shape_counts(shape, lhs_elements, rhs_elements, output_elements, work)) {
      error = "matrix element count overflows";
      return false;
    }
    try {
      AlignedBuffer lhs(static_cast<std::size_t>(lhs_elements * sizeof(float)),
                        options.alignment_bytes, true);
      AlignedBuffer rhs(static_cast<std::size_t>(rhs_elements * sizeof(float)),
                        options.alignment_bytes, true);
      AlignedBuffer output(static_cast<std::size_t>(output_elements * sizeof(float)),
                           options.alignment_bytes, true);
      AlignedBuffer workspace(static_cast<std::size_t>(result.plan.workspace_bytes),
                              std::max(result.plan.workspace_alignment, 1U));
      AlignedBuffer prepacked_b(
          options.packing_mode == PackingModeV1::prepack_b
              ? static_cast<std::size_t>(result.plan.prepacked_b_bytes)
              : 0,
          std::max(result.plan.workspace_alignment, 1U));
      std::vector<std::byte> cold_cache(
          options.cache_mode == CacheModeV1::cold
              ? static_cast<std::size_t>(kColdCacheBytes)
              : 0);
      std::uint64_t random_state = options.seed ^
                                   static_cast<std::uint64_t>(shape.m) ^
                                   (static_cast<std::uint64_t>(shape.n) << 17) ^
                                   (static_cast<std::uint64_t>(shape.k) << 33);
      fill_input(reinterpret_cast<float *>(lhs.data()),
                 static_cast<std::size_t>(lhs_elements), random_state);
      fill_input(reinterpret_cast<float *>(rhs.data()),
                 static_cast<std::size_t>(rhs_elements), random_state);

      bool packing_prepared = false;
      if (options.packing_mode != PackingModeV1::include) {
        if (!runner.prepare(result.plan, shape,
                            reinterpret_cast<const float *>(lhs.data()),
                            reinterpret_cast<const float *>(rhs.data()),
                            workspace.span(),
                            prepacked_b.span(),
                            options.packing_mode == PackingModeV1::prepack_b,
                            error))
          return false;
        packing_prepared = true;
      }

      std::unique_ptr<AlignedBuffer> one_shot_output;
      std::unique_ptr<AlignedBuffer> one_shot_workspace;
      const auto invoke = [&]() -> bool {
        AlignedBuffer *active_output = &output;
        AlignedBuffer *active_workspace = &workspace;
        RunnerPlanV1 active_plan = result.plan;
        if (options.allocation_mode == AllocationModeV1::include_allocation) {
          active_plan = runner.plan(shape, options.alignment_bytes,
                                    options.requested_threads,
                                    options.requested_variant);
          if (!active_plan.legal || active_plan.selected_variant !=
                                        result.plan.selected_variant ||
              active_plan.workspace_bytes != result.plan.workspace_bytes ||
              active_plan.workspace_alignment != result.plan.workspace_alignment ||
              active_plan.prepacked_b_bytes != result.plan.prepacked_b_bytes) {
            error = "one-shot planning was not deterministic";
            return false;
          }
          one_shot_output = std::make_unique<AlignedBuffer>(
              static_cast<std::size_t>(output_elements * sizeof(float)),
              options.alignment_bytes, true);
          one_shot_workspace = std::make_unique<AlignedBuffer>(
              static_cast<std::size_t>(active_plan.workspace_bytes),
              std::max(active_plan.workspace_alignment, 1U));
          active_output = one_shot_output.get();
          active_workspace = one_shot_workspace.get();
        }
        bool prepared = packing_prepared;
        if (options.packing_mode == PackingModeV1::include) {
          if (!runner.prepare(active_plan, shape,
                              reinterpret_cast<const float *>(lhs.data()),
                              reinterpret_cast<const float *>(rhs.data()),
                              active_workspace->span(), prepacked_b.span(), false,
                              error))
            return false;
          prepared = true;
        }
        if (!runner.execute(active_plan, shape,
                            reinterpret_cast<const float *>(lhs.data()),
                            reinterpret_cast<const float *>(rhs.data()),
                            reinterpret_cast<float *>(active_output->data()),
                            active_workspace->span(), prepacked_b.span(), prepared,
                            error))
          return false;
        return runner.synchronize(error);
      };

      for (std::uint32_t warmup = 0; warmup < options.warmup_iterations; ++warmup) {
        if (options.cache_mode == CacheModeV1::cold) evict_cache(cold_cache);
        if (!invoke()) return false;
      }

      if (options.cache_mode == CacheModeV1::cold) evict_cache(cold_cache);
      const auto probe_begin = std::chrono::steady_clock::now();
      if (!invoke()) return false;
      const auto probe_end = std::chrono::steady_clock::now();
      const auto probe_ns = std::max<std::uint64_t>(
          1, static_cast<std::uint64_t>(
                 std::chrono::duration_cast<std::chrono::nanoseconds>(
                     probe_end - probe_begin)
                     .count()));
      std::uint64_t repetitions = 1;
      if (options.cache_mode == CacheModeV1::hot &&
          probe_ns < options.timer_floor_nanoseconds) {
        repetitions = std::min(
            kMaximumAggregateRepetitions,
            (options.timer_floor_nanoseconds + probe_ns - 1) / probe_ns);
        for (int calibration = 0; calibration < 4; ++calibration) {
          const auto begin = std::chrono::steady_clock::now();
          for (std::uint64_t repetition = 0; repetition < repetitions;
               ++repetition)
            if (!invoke()) return false;
          const auto end = std::chrono::steady_clock::now();
          const auto elapsed_ns = std::max<std::uint64_t>(
              1, static_cast<std::uint64_t>(
                     std::chrono::duration_cast<std::chrono::nanoseconds>(
                         end - begin)
                         .count()));
          if (elapsed_ns >= options.timer_floor_nanoseconds) break;
          const std::uint64_t multiplier =
              (options.timer_floor_nanoseconds + elapsed_ns - 1) / elapsed_ns;
          if (repetitions > kMaximumAggregateRepetitions /
                                std::max<std::uint64_t>(multiplier, 2)) {
            repetitions = kMaximumAggregateRepetitions;
            break;
          }
          repetitions *= std::max<std::uint64_t>(multiplier, 2);
        }
        if (repetitions <= kMaximumAggregateRepetitions / 2)
          repetitions *= 2;
      }

      std::vector<double> samples;
      samples.reserve(options.measured_iterations);
      for (std::uint32_t iteration = 0;
           iteration < options.measured_iterations; ++iteration) {
        if (options.cache_mode == CacheModeV1::cold) evict_cache(cold_cache);
        const auto begin = std::chrono::steady_clock::now();
        for (std::uint64_t repetition = 0; repetition < repetitions; ++repetition)
          if (!invoke()) return false;
        const auto end = std::chrono::steady_clock::now();
        const double aggregate_seconds =
            std::chrono::duration<double>(end - begin).count();
        samples.push_back(aggregate_seconds / static_cast<double>(repetitions));
      }
      result.timing = summarize_timings_v1(
          std::move(samples), repetitions, options.timer_floor_nanoseconds);
      const float *final_output =
          options.allocation_mode == AllocationModeV1::include_allocation
              ? reinterpret_cast<const float *>(one_shot_output->data())
              : reinterpret_cast<const float *>(output.data());
      result.correctness = verify_output(
          shape, reinterpret_cast<const float *>(lhs.data()),
          reinterpret_cast<const float *>(rhs.data()), final_output);
      if (!result.correctness.passed) {
        error = "correctness failed for " + result.plan.selected_variant + ": " +
                result.correctness.reason;
        return false;
      }
      if (result.timing.valid) {
        const long double operations =
            2.0L * static_cast<long double>(shape.m) *
            static_cast<long double>(shape.n) * static_cast<long double>(shape.k);
        result.gflops = static_cast<double>(
            operations / static_cast<long double>(result.timing.median_seconds) /
            1.0e9L);
      }
      report.results.push_back(std::move(result));
    } catch (const std::bad_alloc &) {
      error = "benchmark allocation failed despite the preflight memory bound";
      return false;
    }
  }
  return true;
}

void write_json_v1(const BenchmarkReportV1 &report, std::ostream &output) {
  output << std::setprecision(17);
  output << "{\n  \"schema\": \"matcore.benchmark.cpu.gemm\",\n"
         << "  \"version\": " << report.schema_version << ",\n"
         << "  \"operation\": ";
  json_string(output, report.operation);
  output << ",\n  \"dtype\": "; json_string(output, report.dtype);
  output << ",\n  \"accumulation_dtype\": ";
  json_string(output, report.accumulation_dtype);
  output << ",\n  \"layout\": "; json_string(output, report.layout);
  output << ",\n  \"environment\": {\n";
  const auto &environment = report.environment;
  const auto emit_environment = [&](std::string_view name,
                                    std::string_view value, bool comma = true) {
    output << "    \"" << name << "\": "; json_string(output, value);
    output << (comma ? ",\n" : "\n");
  };
  emit_environment("os_family", environment.os_family);
  emit_environment("architecture", environment.architecture);
  emit_environment("compiler", environment.compiler);
  emit_environment("compiler_flags", environment.compiler_flags);
  emit_environment("build_type", environment.build_type);
  emit_environment("cpu_model", environment.cpu_model);
  emit_environment("governor", environment.governor);
  emit_environment("frequency_policy", environment.frequency_policy);
  emit_environment("boost_state", environment.boost_state);
  emit_environment("cpu_affinity", environment.cpu_affinity);
  output << "    \"hardware_threads\": " << environment.hardware_threads
         << ",\n";
  emit_environment("source_commit", environment.source_commit);
  emit_environment("timer_source", environment.timer_source);
  output << "    \"timer_resolution_ns\": "
         << environment.timer_resolution_nanoseconds << ",\n"
         << "    \"timestamp_unix_seconds\": "
         << environment.timestamp_unix_seconds << ",\n";
  emit_environment("capability_record", environment.runner.capability_record);
  emit_environment("provider_name", environment.runner.provider_name);
  emit_environment("provider_version", environment.runner.provider_version);
  emit_environment("provider_config", environment.runner.provider_config, false);
  output << "  },\n  \"configuration\": {\n"
         << "    \"profile\": "; json_string(output, profile_name_v1(report.options.profile));
  output << ",\n    \"requested_variant\": ";
  json_string(output, report.options.requested_variant);
  output << ",\n    \"requested_threads\": " << report.options.requested_threads
         << ",\n    \"warmup_iterations\": " << report.options.warmup_iterations
         << ",\n    \"measured_iterations\": " << report.options.measured_iterations
         << ",\n    \"alignment_bytes\": " << report.options.alignment_bytes
         << ",\n    \"cache_mode\": "; json_string(output, cache_mode_name_v1(report.options.cache_mode));
  output << ",\n    \"allocation_mode\": ";
  json_string(output, allocation_mode_name_v1(report.options.allocation_mode));
  output << ",\n    \"packing_mode\": ";
  json_string(output, packing_mode_name_v1(report.options.packing_mode));
  output << ",\n    \"maximum_memory_bytes\": " << report.options.maximum_memory_bytes
         << ",\n    \"timer_floor_ns\": " << report.options.timer_floor_nanoseconds
         << ",\n    \"seed\": " << report.options.seed << "\n  },\n"
         << "  \"results\": [\n";
  for (std::size_t index = 0; index < report.results.size(); ++index) {
    const auto &result = report.results[index];
    output << "    {\n      \"m\": " << result.shape.m
           << ", \"n\": " << result.shape.n
           << ", \"k\": " << result.shape.k << ",\n"
           << "      \"requested_variant\": "; json_string(output, result.requested_variant);
    output << ",\n      \"selected_variant\": ";
    json_string(output, result.plan.selected_variant);
    output << ",\n      \"selection_reason\": "; json_string(output, result.plan.reason);
    output << ",\n      \"plan_diagnostic\": ";
    json_string(output, result.plan.diagnostic);
    output << ",\n      \"planner_mode\": ";
    json_string(output,
                result.requested_variant == "auto" ? "automatic" : "forced");
    output << ",\n      \"actual_threads\": " << result.plan.actual_threads
           << ",\n      \"workspace_bytes\": " << result.plan.workspace_bytes
           << ",\n      \"workspace_alignment\": " << result.plan.workspace_alignment
           << ",\n      \"prepacked_b_bytes\": " << result.plan.prepacked_b_bytes
           << ",\n      \"packing_required\": "
           << (result.plan.packing_required ? "true" : "false")
           << ",\n      \"supports_prepacked_b\": "
           << (result.plan.supports_prepacked_b ? "true" : "false")
           << ",\n      \"alignment_bytes\": " << report.options.alignment_bytes
           << ",\n      \"lhs_stride\": " << result.shape.k
           << ",\n      \"rhs_stride\": " << result.shape.n
           << ",\n      \"output_stride\": " << result.shape.n
           << ",\n      \"estimated_memory_bytes\": " << result.estimated_memory_bytes
           << ",\n      \"cache_mode\": "; json_string(output, cache_mode_name_v1(result.cache_mode));
    output << ",\n      \"allocation_mode\": ";
    json_string(output, allocation_mode_name_v1(result.allocation_mode));
    output << ",\n      \"packing_mode\": ";
    json_string(output, packing_mode_name_v1(result.packing_mode));
    output << ",\n      \"timing_valid\": "
           << (result.timing.valid ? "true" : "false")
           << ",\n      \"timing_rejection_reason\": ";
    json_string(output, result.timing.rejection_reason);
    output << ",\n      \"aggregate_repetitions\": "
           << result.timing.aggregate_repetitions
           << ",\n      \"minimum_seconds\": " << result.timing.minimum_seconds
           << ",\n      \"median_seconds\": " << result.timing.median_seconds
           << ",\n      \"p95_seconds\": " << result.timing.p95_seconds
           << ",\n      \"gflops\": " << result.gflops
           << ",\n      \"checksum\": " << result.correctness.checksum
           << ",\n      \"expected_checksum\": "
           << result.correctness.expected_checksum
           << ",\n      \"correctness\": "
           << (result.correctness.passed ? "true" : "false")
           << ",\n      \"oracle_mode\": "; json_string(output, result.correctness.oracle_mode);
    output << ",\n      \"checked_elements\": "
           << result.correctness.checked_elements
           << ",\n      \"maximum_absolute_error\": "
           << result.correctness.maximum_absolute_error
           << ",\n      \"maximum_allowed_error\": "
           << result.correctness.maximum_allowed_error
           << ",\n      \"correctness_reason\": ";
    json_string(output, result.correctness.reason);
    output << "\n    }" << (index + 1 == report.results.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
}

}  // namespace matcore::mdslc::bench
