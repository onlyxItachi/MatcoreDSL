#include "cpu_packed_avx512.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

namespace {

namespace runtime = matcore::mdslc::runtime;
namespace planner = matcore::mdslc::planner;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

class GuardedFloatBuffer {
 public:
  explicit GuardedFloatBuffer(std::size_t elements,
                              std::size_t float_offset = 0)
      : elements_(elements), storage_((elements + 80) * sizeof(float) + 64) {
    const auto raw = reinterpret_cast<std::uintptr_t>(storage_.data());
    const auto aligned = (raw + 63U) & ~std::uintptr_t{63U};
    data_ = reinterpret_cast<float *>(aligned) + 16 + float_offset;
    std::fill(data_ - kGuardFloats, data_, kGuardValue);
    std::fill(data_ + elements_, data_ + elements_ + kGuardFloats,
              kGuardValue);
  }

  float *data() noexcept { return data_; }
  const float *data() const noexcept { return data_; }
  std::size_t size() const noexcept { return elements_; }

  bool guards_intact() const noexcept {
    for (std::size_t index = 0; index < kGuardFloats; ++index) {
      if (data_[-static_cast<std::ptrdiff_t>(kGuardFloats) +
                static_cast<std::ptrdiff_t>(index)] != kGuardValue ||
          data_[elements_ + index] != kGuardValue) {
        return false;
      }
    }
    return true;
  }

 private:
  static constexpr std::size_t kGuardFloats = 8;
  static constexpr float kGuardValue = -918273.5F;
  std::size_t elements_ = 0;
  std::vector<std::byte> storage_;
  float *data_ = nullptr;
};

class GuardedWorkspace {
 public:
  explicit GuardedWorkspace(std::size_t bytes, std::size_t byte_offset = 0)
      : bytes_(bytes), storage_(bytes + 192) {
    const auto raw = reinterpret_cast<std::uintptr_t>(storage_.data());
    const auto aligned = (raw + 63U) & ~std::uintptr_t{63U};
    data_ = reinterpret_cast<std::byte *>(aligned) + byte_offset;
    std::fill(data_ + bytes_, data_ + bytes_ + kGuardBytes, kGuardValue);
  }

  void *data() noexcept { return data_; }
  std::size_t size() const noexcept { return bytes_; }
  bool guard_intact() const noexcept {
    return std::all_of(data_ + bytes_, data_ + bytes_ + kGuardBytes,
                       [](std::byte value) { return value == kGuardValue; });
  }

 private:
  static constexpr std::size_t kGuardBytes = 32;
  static constexpr std::byte kGuardValue{0x5a};
  std::size_t bytes_ = 0;
  std::vector<std::byte> storage_;
  std::byte *data_ = nullptr;
};

planner::CpuGemmProblemV1 problem(std::size_t m, std::size_t k, std::size_t n,
                                  std::uint32_t alignment = 64) {
  return {static_cast<std::int64_t>(m),
          static_cast<std::int64_t>(n),
          static_cast<std::int64_t>(k),
          planner::CpuScalarTypeV1::f32,
          planner::CpuScalarTypeV1::f32,
          planner::CpuLayoutV1::row_major_contiguous,
          alignment};
}

void fill_inputs(GuardedFloatBuffer &lhs, GuardedFloatBuffer &rhs,
                 std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
  for (std::size_t index = 0; index < lhs.size(); ++index)
    lhs.data()[index] = distribution(generator);
  for (std::size_t index = 0; index < rhs.size(); ++index)
    rhs.data()[index] = distribution(generator);
}

std::vector<double> oracle(const float *lhs, const float *rhs, std::size_t m,
                           std::size_t k, std::size_t n) {
  std::vector<double> result(m * n, 0.0);
  for (std::size_t row = 0; row < m; ++row)
    for (std::size_t column = 0; column < n; ++column)
      for (std::size_t p = 0; p < k; ++p)
        result[row * n + column] +=
            static_cast<double>(lhs[row * k + p]) *
            static_cast<double>(rhs[p * n + column]);
  return result;
}

bool close_to_oracle(const float *actual, const std::vector<double> &expected) {
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const double value = static_cast<double>(actual[index]);
    const double scale = std::max(1.0, std::fabs(expected[index]));
    if (!std::isfinite(value) ||
        std::fabs(value - expected[index]) > 1.0e-4 * scale) {
      return false;
    }
  }
  return true;
}

void run_shape(std::size_t m, std::size_t k, std::size_t n,
               std::size_t offset, std::uint32_t seed) {
  const auto gemm_problem = problem(m, k, n, offset == 0 ? 64U : 4U);
  GuardedFloatBuffer lhs(m * k, offset);
  GuardedFloatBuffer rhs(k * n, offset);
  GuardedFloatBuffer out(m * n, offset);
  fill_inputs(lhs, rhs, seed);
  std::fill(out.data(), out.data() + out.size(), -77.0F);
  const auto expected = oracle(lhs.data(), rhs.data(), m, k, n);

  runtime::CpuPackedGemmWorkspaceRequirementsV1 requirements;
  expect(runtime::cpu_packed_avx512_workspace_requirements_v1(
             gemm_problem,
             runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
             &requirements) == runtime::CpuPackedGemmStatusV1::success,
         "AVX-512 workspace query succeeds");
  GuardedWorkspace workspace(requirements.total_bytes);
  expect(runtime::cpu_execute_packed_avx512_v1(
             gemm_problem, lhs.data(), rhs.data(), out.data(), workspace.data(),
             workspace.size()) == runtime::CpuPackedGemmStatusV1::success,
         "AVX-512 packed execution succeeds");
  expect(close_to_oracle(out.data(), expected),
         "AVX-512 result matches independent double oracle");
  expect(lhs.guards_intact() && rhs.guards_intact() && out.guards_intact() &&
             workspace.guard_intact(),
         "AVX-512 tails and alignment preserve every guard");
}

void representative_and_randomized_correctness() {
  struct Shape {
    std::size_t m;
    std::size_t k;
    std::size_t n;
  };
  constexpr std::array<Shape, 13> shapes{{
      {1, 1, 1},       {2, 3, 2},       {4, 8, 16},
      {3, 5, 17},      {5, 19, 7},      {16, 16, 16},
      {31, 35, 33},    {33, 37, 35},    {63, 67, 65},
      {127, 131, 129}, {129, 17, 257},  {7, 257, 19},
      {130, 259, 258},
  }};
  std::uint32_t seed = 100;
  for (const auto &shape : shapes) {
    run_shape(shape.m, shape.k, shape.n, 0, seed++);
    run_shape(shape.m, shape.k, shape.n, 1, seed++);
  }

  std::mt19937 generator(0x4d435035U);
  std::uniform_int_distribution<std::size_t> m_distribution(1, 39);
  std::uniform_int_distribution<std::size_t> k_distribution(1, 43);
  std::uniform_int_distribution<std::size_t> n_distribution(1, 47);
  for (std::uint32_t iteration = 0; iteration < 24; ++iteration) {
    run_shape(m_distribution(generator), k_distribution(generator),
              n_distribution(generator), iteration % 2, 1000U + iteration);
  }
}

void prepacked_interoperability() {
  constexpr std::size_t m = 35;
  constexpr std::size_t k = 259;
  constexpr std::size_t n = 259;
  const auto gemm_problem = problem(m, k, n);
  GuardedFloatBuffer lhs(m * k);
  GuardedFloatBuffer rhs(k * n);
  GuardedFloatBuffer out(m * n);
  fill_inputs(lhs, rhs, 77);
  const auto expected = oracle(lhs.data(), rhs.data(), m, k, n);

  runtime::CpuPackedGemmWorkspaceRequirementsV1 storage_requirements;
  (void)runtime::cpu_packed_avx512_prepacked_b_requirements_v1(
      gemm_problem, &storage_requirements);
  GuardedWorkspace packed_storage(storage_requirements.total_bytes);
  runtime::CpuPackedBViewV1 view;
  expect(runtime::cpu_prepare_packed_b_avx2_v1(
             gemm_problem, rhs.data(), packed_storage.data(),
             packed_storage.size(), &view) ==
             runtime::CpuPackedGemmStatusV1::success,
         "AVX2 preparation creates the common packed-B v1 view");

  runtime::CpuPackedGemmWorkspaceRequirementsV1 execute_requirements;
  (void)runtime::cpu_packed_avx512_workspace_requirements_v1(
      gemm_problem,
      runtime::CpuPackedGemmWorkspaceModeV1::transient_a_with_prepacked_b,
      &execute_requirements);
  GuardedWorkspace workspace(execute_requirements.total_bytes);
  expect(runtime::cpu_execute_packed_avx512_prepacked_b_v1(
             gemm_problem, lhs.data(), out.data(), view, workspace.data(),
             workspace.size()) == runtime::CpuPackedGemmStatusV1::success,
         "AVX-512 consumes an AVX2-prepared common packed-B view");
  expect(close_to_oracle(out.data(), expected),
         "cross-ISA prepacked execution matches the oracle");

  GuardedWorkspace second_storage(storage_requirements.total_bytes);
  runtime::CpuPackedBViewV1 second_view;
  expect(runtime::cpu_prepare_packed_b_avx512_v1(
             gemm_problem, rhs.data(), second_storage.data(),
             second_storage.size(), &second_view) ==
             runtime::CpuPackedGemmStatusV1::success,
         "AVX-512 preparation creates the common packed-B v1 view");
  std::fill(out.data(), out.data() + out.size(), -12.0F);
  expect(runtime::cpu_execute_packed_avx2_prepacked_b_v1(
             gemm_problem, lhs.data(), out.data(), second_view,
             workspace.data(), workspace.size()) ==
             runtime::CpuPackedGemmStatusV1::success,
         "AVX2 consumes an AVX-512-prepared common packed-B view");
  expect(close_to_oracle(out.data(), expected),
         "reverse cross-ISA prepacked execution matches the oracle");
}

void rejection_contract() {
  constexpr std::size_t m = 17;
  constexpr std::size_t k = 19;
  constexpr std::size_t n = 23;
  const auto gemm_problem = problem(m, k, n);
  GuardedFloatBuffer lhs(m * k);
  GuardedFloatBuffer rhs(k * n);
  GuardedFloatBuffer out(m * n);
  std::fill(out.data(), out.data() + out.size(), 123.0F);
  const std::vector<float> unchanged(out.data(), out.data() + out.size());

  runtime::CpuPackedGemmWorkspaceRequirementsV1 requirements;
  (void)runtime::cpu_packed_avx512_workspace_requirements_v1(
      gemm_problem,
      runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
      &requirements);
  GuardedWorkspace workspace(requirements.total_bytes);
  GuardedWorkspace misaligned_workspace(requirements.total_bytes, 1);

  expect(runtime::cpu_execute_packed_avx512_v1(
             gemm_problem, lhs.data(), rhs.data(), out.data(), workspace.data(),
             requirements.total_bytes - 1) ==
             runtime::CpuPackedGemmStatusV1::workspace_insufficient &&
             std::equal(out.data(), out.data() + out.size(), unchanged.begin()),
         "insufficient workspace fails before output mutation");
  expect(runtime::cpu_execute_packed_avx512_v1(
             gemm_problem, lhs.data(), rhs.data(), out.data(),
             misaligned_workspace.data(), misaligned_workspace.size()) ==
             runtime::CpuPackedGemmStatusV1::workspace_misaligned &&
             std::equal(out.data(), out.data() + out.size(), unchanged.begin()),
         "misaligned workspace fails before output mutation");
  expect(runtime::cpu_execute_packed_avx512_v1(
             gemm_problem, lhs.data(), rhs.data(), lhs.data(), workspace.data(),
             workspace.size()) == runtime::CpuPackedGemmStatusV1::alias_violation,
         "output/input alias fails before packing");

  auto invalid = gemm_problem;
  invalid.m = 0;
  expect(runtime::cpu_execute_packed_avx512_v1(
             invalid, lhs.data(), rhs.data(), out.data(), workspace.data(),
             workspace.size()) == runtime::CpuPackedGemmStatusV1::invalid_problem &&
             std::equal(out.data(), out.data() + out.size(), unchanged.begin()),
         "invalid dimensions fail before ISA dispatch or output mutation");
  invalid = gemm_problem;
  invalid.m = std::numeric_limits<std::int64_t>::max();
  invalid.k = std::numeric_limits<std::int64_t>::max();
  expect(runtime::cpu_packed_avx512_workspace_requirements_v1(
             invalid,
             runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
             &requirements) == runtime::CpuPackedGemmStatusV1::arithmetic_overflow,
         "AVX-512 workspace arithmetic fails closed");
}

}  // namespace

int main() {
  expect(runtime::cpu_packed_avx512_build_available_v1(),
         "AVX-512 backend is function-target compiled on x86-64");
  expect(runtime::cpu_packed_avx512_runtime_usable_v1(),
         "validation host exposes OS-usable AVX-512F and FMA");
  if (!runtime::cpu_packed_avx512_runtime_usable_v1()) return 1;

  representative_and_randomized_correctness();
  prepacked_interoperability();
  rejection_contract();
  if (failures != 0) return 1;
  std::cout << "native packed AVX-512/FMA GEMM: all tests passed\n";
  return 0;
}
