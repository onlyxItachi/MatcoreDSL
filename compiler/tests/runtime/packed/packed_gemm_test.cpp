#include "cpu_gemm_backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
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
  const void *data() const noexcept { return data_; }
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
    if (!std::isfinite(value) || !std::isfinite(expected[index])) return false;
    const double scale = std::max(1.0, std::fabs(expected[index]));
    if (std::fabs(value - expected[index]) > 1.0e-4 * scale) return false;
  }
  return true;
}

void run_transient_shape(std::size_t m, std::size_t k, std::size_t n,
                         std::size_t input_float_offset, std::uint32_t seed) {
  const std::uint32_t alignment = input_float_offset == 0 ? 64U : 4U;
  const auto gemm_problem = problem(m, k, n, alignment);
  GuardedFloatBuffer lhs(m * k, input_float_offset);
  GuardedFloatBuffer rhs(k * n, input_float_offset);
  GuardedFloatBuffer out(m * n, input_float_offset);
  fill_inputs(lhs, rhs, seed);
  std::fill(out.data(), out.data() + out.size(), -71.0F);
  const auto expected = oracle(lhs.data(), rhs.data(), m, k, n);

  runtime::CpuPackedGemmWorkspaceRequirementsV1 requirements;
  const auto query = runtime::cpu_packed_avx2_workspace_requirements_v1(
      gemm_problem,
      runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
      &requirements);
  expect(query == runtime::CpuPackedGemmStatusV1::success,
         "transient workspace query succeeds");
  expect(requirements.version == runtime::kCpuPackedGemmBackendVersionV1 &&
             requirements.alignment_bytes == 64 &&
             requirements.total_bytes != 0 &&
             requirements.packed_a_bytes != 0 &&
             requirements.packed_b_bytes != 0 &&
             requirements.packed_b_offset % 64 == 0,
         "transient workspace contract is explicit and aligned");
  GuardedWorkspace workspace(requirements.total_bytes);
  const auto status = runtime::cpu_execute_packed_avx2_v1(
      gemm_problem, lhs.data(), rhs.data(), out.data(), workspace.data(),
      workspace.size());
  expect(status == runtime::CpuPackedGemmStatusV1::success,
         "packed AVX2 transient execution succeeds");
  expect(close_to_oracle(out.data(), expected),
         "packed AVX2 transient result matches double oracle");
  expect(lhs.guards_intact() && rhs.guards_intact() && out.guards_intact() &&
             workspace.guard_intact(),
         "packed AVX2 transient execution preserves every guard");
}

void representative_and_randomized_correctness() {
  struct Shape {
    std::size_t m;
    std::size_t k;
    std::size_t n;
  };
  const std::array<Shape, 15> shapes{{
      {1, 1, 1},       {2, 3, 2},       {4, 8, 16},
      {3, 5, 17},      {5, 19, 7},      {16, 16, 16},
      {31, 35, 33},    {33, 37, 35},    {63, 67, 65},
      {127, 131, 129}, {129, 17, 257},  {7, 257, 19},
      {130, 259, 258}, {257, 33, 15},    {8, 513, 32},
  }};
  std::uint32_t seed = 1;
  for (const auto &shape : shapes) {
    run_transient_shape(shape.m, shape.k, shape.n, 0, seed++);
    run_transient_shape(shape.m, shape.k, shape.n, 1, seed++);
  }

  std::mt19937 generator(0x4d435034U);
  std::uniform_int_distribution<std::size_t> m_distribution(1, 39);
  std::uniform_int_distribution<std::size_t> k_distribution(1, 43);
  std::uniform_int_distribution<std::size_t> n_distribution(1, 47);
  for (std::uint32_t iteration = 0; iteration < 40; ++iteration) {
    run_transient_shape(m_distribution(generator), k_distribution(generator),
                        n_distribution(generator), iteration % 2,
                        1000U + iteration);
  }
}

void prepacked_b_correctness() {
  constexpr std::size_t m = 35;
  constexpr std::size_t k = 259;
  constexpr std::size_t n = 259;
  const auto gemm_problem = problem(m, k, n);
  GuardedFloatBuffer lhs(m * k);
  GuardedFloatBuffer rhs(k * n);
  GuardedFloatBuffer out(m * n);
  fill_inputs(lhs, rhs, 77);

  runtime::CpuPackedGemmWorkspaceRequirementsV1 storage_requirements;
  expect(runtime::cpu_packed_avx2_prepacked_b_requirements_v1(
             gemm_problem, &storage_requirements) ==
             runtime::CpuPackedGemmStatusV1::success,
         "prepacked-B storage query succeeds");
  GuardedWorkspace packed_storage(storage_requirements.total_bytes);
  runtime::CpuPackedBViewV1 packed_view;
  expect(runtime::cpu_prepare_packed_b_avx2_v1(
             gemm_problem, rhs.data(), packed_storage.data(),
             packed_storage.size(), &packed_view) ==
             runtime::CpuPackedGemmStatusV1::success,
         "prepacked-B preparation succeeds");
  expect(packed_view.version == runtime::kCpuPackedGemmBackendVersionV1 &&
             packed_view.source_data == rhs.data() &&
             packed_view.packed_data == packed_storage.data() &&
             packed_view.packed_elements ==
                 storage_requirements.packed_b_bytes / sizeof(float) &&
             packed_view.provenance != 0,
         "prepacked-B view records stable provenance and exact storage");

  runtime::CpuPackedGemmWorkspaceRequirementsV1 execute_requirements;
  expect(runtime::cpu_packed_avx2_workspace_requirements_v1(
             gemm_problem,
             runtime::CpuPackedGemmWorkspaceModeV1::
                 transient_a_with_prepacked_b,
             &execute_requirements) ==
             runtime::CpuPackedGemmStatusV1::success &&
             execute_requirements.packed_b_bytes == 0,
         "prepacked-B execution requires only transient A storage");
  GuardedWorkspace workspace(execute_requirements.total_bytes);

  for (std::uint32_t repetition = 0; repetition < 3; ++repetition) {
    for (std::size_t index = 0; index < lhs.size(); ++index) {
      const auto encoded = static_cast<std::int32_t>(
          (index * 17U + repetition * 13U) % 37U);
      lhs.data()[index] = static_cast<float>(encoded - 18) / 32.0F;
    }
    const auto expected = oracle(lhs.data(), rhs.data(), m, k, n);
    std::fill(out.data(), out.data() + out.size(), -19.0F);
    expect(runtime::cpu_execute_packed_avx2_prepacked_b_v1(
               gemm_problem, lhs.data(), out.data(), packed_view,
               workspace.data(), workspace.size()) ==
               runtime::CpuPackedGemmStatusV1::success,
           "repeated prepacked-B execution succeeds");
    expect(close_to_oracle(out.data(), expected),
           "repeated prepacked-B result matches double oracle");
  }
  expect(lhs.guards_intact() && rhs.guards_intact() && out.guards_intact() &&
             packed_storage.guard_intact() && workspace.guard_intact(),
         "prepacked-B preparation and reuse preserve every guard");
}

void workspace_and_rejection_contract() {
  constexpr std::size_t m = 17;
  constexpr std::size_t k = 19;
  constexpr std::size_t n = 23;
  const auto gemm_problem = problem(m, k, n);
  GuardedFloatBuffer lhs(m * k);
  GuardedFloatBuffer rhs(k * n);
  GuardedFloatBuffer out(m * n);
  fill_inputs(lhs, rhs, 91);
  std::fill(out.data(), out.data() + out.size(), 123.0F);
  const std::vector<float> unchanged(out.data(), out.data() + out.size());

  runtime::CpuPackedGemmWorkspaceRequirementsV1 requirements;
  expect(runtime::cpu_packed_avx2_workspace_requirements_v1(
             gemm_problem,
             runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
             &requirements) == runtime::CpuPackedGemmStatusV1::success,
         "adversarial workspace query succeeds");
  GuardedWorkspace workspace(requirements.total_bytes);
  GuardedWorkspace misaligned_workspace(requirements.total_bytes, 1);

  expect(runtime::cpu_execute_packed_avx2_v1(
             gemm_problem, lhs.data(), rhs.data(), out.data(), workspace.data(),
             requirements.total_bytes - 1) ==
             runtime::CpuPackedGemmStatusV1::workspace_insufficient &&
             std::equal(out.data(), out.data() + out.size(), unchanged.begin()),
         "insufficient workspace fails before output mutation");
  expect(runtime::cpu_execute_packed_avx2_v1(
             gemm_problem, lhs.data(), rhs.data(), out.data(),
             misaligned_workspace.data(), misaligned_workspace.size()) ==
             runtime::CpuPackedGemmStatusV1::workspace_misaligned &&
             std::equal(out.data(), out.data() + out.size(), unchanged.begin()),
         "misaligned workspace fails before output mutation");
  expect(runtime::cpu_execute_packed_avx2_v1(
             gemm_problem, lhs.data(), rhs.data(), out.data(), nullptr,
             requirements.total_bytes) ==
             runtime::CpuPackedGemmStatusV1::null_pointer &&
             std::equal(out.data(), out.data() + out.size(), unchanged.begin()),
         "null workspace fails before output mutation");

  const auto square = problem(16, 16, 16);
  GuardedFloatBuffer square_lhs(16 * 16);
  GuardedFloatBuffer square_rhs(16 * 16);
  runtime::CpuPackedGemmWorkspaceRequirementsV1 square_requirements;
  (void)runtime::cpu_packed_avx2_workspace_requirements_v1(
      square, runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
      &square_requirements);
  GuardedWorkspace square_workspace(square_requirements.total_bytes);
  expect(runtime::cpu_execute_packed_avx2_v1(
             square, square_lhs.data(), square_rhs.data(), square_lhs.data(),
             square_workspace.data(), square_workspace.size()) ==
             runtime::CpuPackedGemmStatusV1::alias_violation,
         "output/input alias is rejected");
  const auto square_low_alignment = problem(16, 16, 16, alignof(float));
  expect(runtime::cpu_execute_packed_avx2_v1(
             square_low_alignment, square_lhs.data(), square_rhs.data(),
             square_lhs.data() + 1, square_workspace.data(),
             square_workspace.size()) ==
             runtime::CpuPackedGemmStatusV1::alias_violation,
         "partial output/input overlap is rejected");
  expect(runtime::cpu_execute_packed_avx2_v1(
             square, square_lhs.data(), square_rhs.data(), out.data(),
             square_lhs.data(), square_requirements.total_bytes) ==
             runtime::CpuPackedGemmStatusV1::alias_violation,
         "workspace/input alias is rejected before packing");

  GuardedFloatBuffer misaligned_lhs(m * k, 1);
  GuardedFloatBuffer misaligned_rhs(k * n, 1);
  GuardedFloatBuffer misaligned_out(m * n, 1);
  expect(runtime::cpu_execute_packed_avx2_v1(
             gemm_problem, misaligned_lhs.data(), misaligned_rhs.data(),
             misaligned_out.data(), workspace.data(), workspace.size()) ==
             runtime::CpuPackedGemmStatusV1::invalid_pointer_alignment,
         "actual pointers must satisfy the planner's declared alignment");

  auto invalid = gemm_problem;
  invalid.m = 0;
  runtime::CpuPackedGemmWorkspaceRequirementsV1 untouched;
  untouched.total_bytes = 9123;
  expect(runtime::cpu_packed_avx2_workspace_requirements_v1(
             invalid,
             runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
             &untouched) == runtime::CpuPackedGemmStatusV1::invalid_problem &&
             untouched.total_bytes == 9123,
         "invalid workspace query leaves caller storage unchanged");
  invalid = gemm_problem;
  invalid.m = std::numeric_limits<std::int64_t>::max();
  invalid.k = std::numeric_limits<std::int64_t>::max();
  expect(runtime::cpu_packed_avx2_workspace_requirements_v1(
             invalid,
             runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
             &untouched) == runtime::CpuPackedGemmStatusV1::arithmetic_overflow,
         "workspace arithmetic overflow is rejected");

  runtime::CpuPackedGemmWorkspaceRequirementsV1 packed_requirements;
  (void)runtime::cpu_packed_avx2_prepacked_b_requirements_v1(
      gemm_problem, &packed_requirements);
  GuardedWorkspace packed_storage(packed_requirements.total_bytes);
  runtime::CpuPackedBViewV1 view;
  view.version = 77;
  expect(runtime::cpu_prepare_packed_b_avx2_v1(
             gemm_problem, rhs.data(), packed_storage.data(),
             packed_requirements.total_bytes - 1, &view) ==
             runtime::CpuPackedGemmStatusV1::workspace_insufficient &&
             view.version == 77,
         "failed prepack leaves the view unchanged");
  expect(runtime::cpu_prepare_packed_b_avx2_v1(
             gemm_problem, rhs.data(), rhs.data(), packed_requirements.total_bytes,
             &view) == runtime::CpuPackedGemmStatusV1::alias_violation,
         "prepacked storage may not overlap its source");
  expect(runtime::cpu_prepare_packed_b_avx2_v1(
             gemm_problem, rhs.data(), packed_storage.data(),
             packed_storage.size(), &view) ==
             runtime::CpuPackedGemmStatusV1::success,
         "valid prepack follows rejected attempts");
  auto corrupted = view;
  corrupted.provenance ^= UINT64_C(1);
  runtime::CpuPackedGemmWorkspaceRequirementsV1 a_requirements;
  (void)runtime::cpu_packed_avx2_workspace_requirements_v1(
      gemm_problem,
      runtime::CpuPackedGemmWorkspaceModeV1::transient_a_with_prepacked_b,
      &a_requirements);
  GuardedWorkspace a_workspace(a_requirements.total_bytes);
  expect(runtime::cpu_execute_packed_avx2_prepacked_b_v1(
             gemm_problem, lhs.data(),
             static_cast<float *>(packed_storage.data()), view,
             a_workspace.data(), a_workspace.size()) ==
             runtime::CpuPackedGemmStatusV1::alias_violation,
         "output may not overlap persistent packed-B storage");
  expect(runtime::cpu_execute_packed_avx2_prepacked_b_v1(
             gemm_problem, lhs.data(), out.data(), corrupted,
             a_workspace.data(), a_workspace.size()) ==
             runtime::CpuPackedGemmStatusV1::invalid_prepacked_b &&
             std::equal(out.data(), out.data() + out.size(), unchanged.begin()),
         "corrupt prepacked provenance fails before output mutation");
}

}  // namespace

int main() {
  expect(runtime::cpu_packed_avx2_build_available_v1(),
         "AVX2 backend is compiled on the validation target");
  expect(runtime::cpu_packed_avx2_runtime_usable_v1(),
         "validation host exposes usable AVX2/FMA");
  if (!runtime::cpu_packed_avx2_runtime_usable_v1()) return 1;

  representative_and_randomized_correctness();
  prepacked_b_correctness();
  workspace_and_rejection_contract();
  if (failures != 0) return 1;
  std::cout << "native packed AVX2/FMA GEMM: all tests passed\n";
  return 0;
}
