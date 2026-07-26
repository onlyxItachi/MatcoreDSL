#include "cpu_parallel_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

namespace {

namespace planner = matcore::mdslc::planner;
namespace runtime = matcore::mdslc::runtime;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes, std::size_t alignment = 64)
      : storage_(bytes + alignment), bytes_(bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(storage_.data());
    const auto aligned = (address + alignment - 1U) & ~(alignment - 1U);
    data_ = reinterpret_cast<std::byte *>(aligned);
  }
  void *data() noexcept { return data_; }
  const void *data() const noexcept { return data_; }
  float *floats() noexcept { return reinterpret_cast<float *>(data_); }
  const float *floats() const noexcept {
    return reinterpret_cast<const float *>(data_);
  }
  std::size_t size() const noexcept { return bytes_; }

 private:
  std::vector<std::byte> storage_;
  std::size_t bytes_ = 0;
  std::byte *data_ = nullptr;
};

planner::CpuGemmProblemV1 problem(std::size_t m, std::size_t k,
                                  std::size_t n,
                                  std::uint32_t alignment = alignof(float)) {
  return {static_cast<std::int64_t>(m), static_cast<std::int64_t>(n),
          static_cast<std::int64_t>(k), planner::CpuScalarTypeV1::f32,
          planner::CpuScalarTypeV1::f32,
          planner::CpuLayoutV1::row_major_contiguous, alignment};
}

void fill(std::vector<float> &values, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
  for (float &value : values) value = distribution(generator);
}

void fill(float *values, std::size_t count, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
  for (std::size_t index = 0; index < count; ++index)
    values[index] = distribution(generator);
}

std::vector<double> oracle(const std::vector<float> &lhs,
                           const std::vector<float> &rhs, std::size_t m,
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

bool close(const std::vector<float> &actual,
           const std::vector<double> &expected) {
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double scale = std::max(1.0, std::fabs(expected[index]));
    if (!std::isfinite(actual[index]) ||
        std::fabs(static_cast<double>(actual[index]) - expected[index]) >
            2.0e-4 * scale) {
      return false;
    }
  }
  return true;
}

bool close(const float *actual, std::size_t count,
           const std::vector<double> &expected) {
  for (std::size_t index = 0; index < count; ++index) {
    const double scale = std::max(1.0, std::fabs(expected[index]));
    if (!std::isfinite(actual[index]) ||
        std::fabs(static_cast<double>(actual[index]) - expected[index]) >
            2.0e-4 * scale) {
      return false;
    }
  }
  return true;
}

bool guards_unchanged(const float *storage, std::size_t total_elements,
                      std::size_t guard_elements,
                      float sentinel) {
  return std::all_of(storage, storage + guard_elements,
                     [sentinel](float value) { return value == sentinel; }) &&
         std::all_of(storage + total_elements - guard_elements,
                     storage + total_elements,
                     [sentinel](float value) { return value == sentinel; });
}

std::unique_ptr<runtime::CpuExecutionContextV1> make_context() {
  runtime::CpuExecutionStatusV1 status{};
  auto result = runtime::CpuExecutionContextV1::create(
      {runtime::kCpuExecutionContextVersionV1, 4, 4}, &status);
  expect(status == runtime::CpuExecutionStatusV1::success && result != nullptr,
         "four-worker execution context is created");
  return result;
}

void run_parallel_correctness() {
  if (!runtime::cpu_packed_avx2_runtime_usable_v1()) {
    std::cout << "parallel packed AVX2 runtime unavailable: test skipped\n";
    return;
  }
  constexpr std::size_t m = 385;
  constexpr std::size_t k = 67;
  constexpr std::size_t n = 65;
  const auto gemm_problem = problem(m, k, n);
  std::vector<float> lhs(m * k);
  std::vector<float> rhs(k * n);
  std::vector<float> out(m * n, -9.0F);
  fill(lhs, 41);
  fill(rhs, 42);
  const auto expected = oracle(lhs, rhs, m, k, n);
  auto context = make_context();

  runtime::CpuParallelGemmWorkspaceRequirementsV1 requirements;
  runtime::CpuPackedGemmWorkspaceRequirementsV1 packed_b_requirements;
  expect(runtime::cpu_packed_avx2_prepacked_b_requirements_v1(
             gemm_problem, &packed_b_requirements) ==
             runtime::CpuPackedGemmStatusV1::success,
         "shared packed-B requirement query succeeds");
  expect(runtime::cpu_parallel_packed_avx2_workspace_requirements_v1(
             gemm_problem, 4, &requirements) ==
             runtime::CpuParallelGemmStatusV1::success,
         "parallel workspace query succeeds");
  expect(requirements.execution_threads == 4 &&
             requirements.alignment_bytes == 64 &&
             requirements.shared_packed_b_bytes ==
                 packed_b_requirements.total_bytes &&
             requirements.worker_region_offset % 64 == 0 &&
             requirements.per_worker_bytes != 0 &&
             requirements.per_worker_stride_bytes % 64 == 0 &&
             requirements.total_bytes ==
                 requirements.worker_region_offset +
                     requirements.per_worker_stride_bytes * 4,
         "parallel workspace exposes one shared B and isolated A slices");
  AlignedBuffer workspace(requirements.total_bytes);

  const auto workers_started = context->info().actual_worker_count;
  for (std::uint64_t repetition = 1; repetition <= 4; ++repetition) {
    std::fill(out.begin(), out.end(), -9.0F);
    runtime::CpuParallelGemmReportV1 report;
    const auto status = runtime::cpu_execute_parallel_packed_avx2_v1(
        *context, gemm_problem, lhs.data(), rhs.data(), out.data(),
        workspace.data(), workspace.size(), 4,
        runtime::CpuProviderNestingPolicyV1::native_only, &report);
    expect(status == runtime::CpuParallelGemmStatusV1::success,
           "parallel packed execution succeeds");
    expect(report.actual_threads == 4 && report.macro_tile_count == 4 &&
               report.row_task_count == 4 &&
               report.column_task_count == 1 &&
               report.task_count == 4 &&
               report.shared_packed_b_bytes ==
                   requirements.shared_packed_b_bytes &&
               report.context_submission == repetition,
           "parallel report exposes threads, shared B, tiles, and reuse");
    expect(close(out, expected),
           "parallel packed result matches double-precision oracle");
  }
  expect(context->info().actual_worker_count == workers_started &&
             context->info().completed_submissions == 4,
         "repeated GEMM execution reuses persistent workers");

  const auto tiny = problem(16, 17, 19);
  std::vector<float> tiny_lhs(16 * 17);
  std::vector<float> tiny_rhs(17 * 19);
  std::vector<float> tiny_out(16 * 19, 0.0F);
  fill(tiny_lhs, 7);
  fill(tiny_rhs, 8);
  runtime::CpuParallelGemmWorkspaceRequirementsV1 tiny_requirements;
  expect(runtime::cpu_parallel_packed_avx2_workspace_requirements_v1(
             tiny, 1, &tiny_requirements) ==
             runtime::CpuParallelGemmStatusV1::success,
         "tiny workspace query succeeds for one selected worker");
  AlignedBuffer tiny_workspace(tiny_requirements.total_bytes);
  runtime::CpuParallelGemmReportV1 tiny_report;
  expect(runtime::cpu_execute_parallel_packed_avx2_v1(
             *context, tiny, tiny_lhs.data(), tiny_rhs.data(), tiny_out.data(),
             tiny_workspace.data(), tiny_workspace.size(), 4,
             runtime::CpuProviderNestingPolicyV1::native_only, &tiny_report) ==
             runtime::CpuParallelGemmStatusV1::success &&
             tiny_report.actual_threads == 1 &&
             tiny_report.macro_tile_count == 1 &&
             tiny_report.row_task_count == 1 &&
             tiny_report.column_task_count == 1 &&
             tiny_report.task_count == 1,
         "tiny GEMM is deterministically reduced to one worker");
}

void run_balanced_partition_correctness() {
  if (!runtime::cpu_packed_avx2_runtime_usable_v1()) return;
  struct TestCase {
    std::size_t m;
    std::size_t k;
    std::size_t n;
    std::size_t expected_macro_tiles;
    std::size_t expected_row_tasks;
    std::size_t expected_column_tasks;
    std::string_view label;
  };
  constexpr TestCase cases[] = {
      {64, 257, 1024, 1, 1, 4, "short-wide sub-MC"},
      {129, 1024, 512, 2, 2, 2, "post-MC task wave"},
      {63, 261, 1024, 1, 1, 4, "short-wide M/K-tail"},
  };
  constexpr std::size_t guard_elements = 16;
  constexpr float guard_sentinel = 937.25F;
  auto context = make_context();

  std::uint64_t submission = 0;
  for (const auto &test_case : cases) {
    const auto gemm_problem =
        problem(test_case.m, test_case.k, test_case.n, 64);
    const std::size_t lhs_elements = test_case.m * test_case.k;
    const std::size_t rhs_elements = test_case.k * test_case.n;
    const std::size_t out_elements = test_case.m * test_case.n;
    const std::size_t guarded_out_elements =
        out_elements + 2 * guard_elements;
    AlignedBuffer lhs_storage(lhs_elements * sizeof(float));
    AlignedBuffer rhs_storage(rhs_elements * sizeof(float));
    AlignedBuffer out_storage(guarded_out_elements * sizeof(float));
    float *lhs = lhs_storage.floats();
    float *rhs = rhs_storage.floats();
    float *guarded_out = out_storage.floats();
    float *out = guarded_out + guard_elements;
    fill(lhs, lhs_elements, static_cast<std::uint32_t>(901 + submission * 2));
    fill(rhs, rhs_elements, static_cast<std::uint32_t>(902 + submission * 2));
    std::fill(guarded_out, guarded_out + guarded_out_elements, guard_sentinel);
    const auto expected =
        oracle(lhs, rhs, test_case.m, test_case.k, test_case.n);

    runtime::CpuParallelGemmWorkspaceRequirementsV1 requirements;
    expect(runtime::cpu_parallel_packed_avx2_workspace_requirements_v1(
               gemm_problem, 4, &requirements) ==
               runtime::CpuParallelGemmStatusV1::success &&
               requirements.execution_threads == 4,
           "adaptive partition workspace reserves four isolated workers");
    AlignedBuffer workspace(requirements.total_bytes);
    runtime::CpuParallelGemmReportV1 report;
    const auto status = runtime::cpu_execute_parallel_packed_avx2_v1(
        *context, gemm_problem, lhs, rhs, out, workspace.data(),
        workspace.size(), 4, runtime::CpuProviderNestingPolicyV1::native_only,
        &report);
    ++submission;
    expect(status == runtime::CpuParallelGemmStatusV1::success,
           test_case.label);
    expect(report.actual_threads == 4 &&
               report.macro_tile_count == test_case.expected_macro_tiles &&
               report.row_task_count == test_case.expected_row_tasks &&
               report.column_task_count ==
                   test_case.expected_column_tasks &&
               report.task_count == 4 &&
               report.context_submission == submission,
           "adaptive partition uses four workers despite an imbalanced "
           "legacy macro-tile wave");
    expect(close(out, out_elements, expected),
           "adaptive partition result matches double-precision oracle");
    expect(guards_unchanged(guarded_out, guarded_out_elements, guard_elements,
                            guard_sentinel),
           "adaptive partition owns only its disjoint output rows");
  }
}

void run_low_alignment_row_only_correctness() {
  if (!runtime::cpu_packed_avx2_runtime_usable_v1()) return;
  constexpr std::size_t m = 64;
  constexpr std::size_t k = 65;
  constexpr std::size_t n = 512;
  const auto gemm_problem = problem(m, k, n, alignof(float));
  AlignedBuffer lhs_storage(m * k * sizeof(float));
  AlignedBuffer rhs_storage(k * n * sizeof(float));
  AlignedBuffer out_storage((m * n + 2) * sizeof(float));
  float *lhs = lhs_storage.floats();
  float *rhs = rhs_storage.floats();
  float *out_storage_begin = out_storage.floats();
  float *out = out_storage_begin + 1;
  fill(lhs, m * k, 1201);
  fill(rhs, k * n, 1202);
  std::fill(out_storage_begin, out_storage_begin + m * n + 2, 19.0F);
  const auto expected = oracle(lhs, rhs, m, k, n);

  auto context = make_context();
  runtime::CpuParallelGemmWorkspaceRequirementsV1 requirements;
  expect(runtime::cpu_parallel_packed_avx2_workspace_requirements_v1(
             gemm_problem, 1, &requirements) ==
             runtime::CpuParallelGemmStatusV1::success,
         "low-alignment row-only workspace query succeeds");
  AlignedBuffer workspace(requirements.total_bytes);
  runtime::CpuParallelGemmReportV1 report;
  expect(runtime::cpu_execute_parallel_packed_avx2_v1(
             *context, gemm_problem, lhs, rhs, out, workspace.data(),
             workspace.size(), 4,
             runtime::CpuProviderNestingPolicyV1::native_only, &report) ==
             runtime::CpuParallelGemmStatusV1::success &&
             report.actual_threads == 1 && report.macro_tile_count == 1 &&
             report.row_task_count == 1 &&
             report.column_task_count == 1 &&
             report.task_count == 1,
         "low-alignment output remains on the fail-closed row-only path");
  expect(close(out, m * n, expected),
         "low-alignment row-only result matches double-precision oracle");
  expect(out_storage_begin[0] == 19.0F &&
             out_storage_begin[m * n + 1] == 19.0F,
         "low-alignment row-only execution preserves output guards");
}

void rejection_preserves_output() {
  if (!runtime::cpu_packed_avx2_runtime_usable_v1()) return;
  const auto gemm_problem = problem(257, 33, 35);
  std::vector<float> lhs(257 * 33, 0.25F);
  std::vector<float> rhs(33 * 35, -0.5F);
  std::vector<float> out(257 * 35, 123.0F);
  const std::vector<float> sentinel = out;
  auto context = make_context();
  runtime::CpuParallelGemmWorkspaceRequirementsV1 requirements;
  expect(runtime::cpu_parallel_packed_avx2_workspace_requirements_v1(
             gemm_problem, 3, &requirements) ==
             runtime::CpuParallelGemmStatusV1::success,
         "rejection fixture workspace query succeeds");
  AlignedBuffer workspace(requirements.total_bytes);
  runtime::CpuParallelGemmReportV1 report;

  expect(runtime::cpu_execute_parallel_packed_avx2_v1(
             *context, gemm_problem, lhs.data(), rhs.data(), out.data(),
             workspace.data(), workspace.size() - 1, 3,
             runtime::CpuProviderNestingPolicyV1::native_only, &report) ==
             runtime::CpuParallelGemmStatusV1::workspace_insufficient &&
             out == sentinel,
         "insufficient workspace rejects before output mutation");
  expect(runtime::cpu_execute_parallel_packed_avx2_v1(
             *context, gemm_problem, lhs.data(), rhs.data(), out.data(),
             workspace.data(), workspace.size(), 5,
             runtime::CpuProviderNestingPolicyV1::native_only, &report) ==
             runtime::CpuParallelGemmStatusV1::invalid_thread_count &&
             out == sentinel,
         "request above context ceiling rejects before output mutation");
  expect(runtime::cpu_execute_parallel_packed_avx2_v1(
             *context, gemm_problem, lhs.data(), rhs.data(), out.data(),
             workspace.data(), workspace.size(), 3,
             runtime::CpuProviderNestingPolicyV1::external_provider_active,
             &report) ==
             runtime::CpuParallelGemmStatusV1::nested_parallelism_rejected &&
             out == sentinel,
         "nested provider/native pool rejects before output mutation");
  expect(runtime::cpu_execute_parallel_packed_avx2_v1(
             *context, gemm_problem, lhs.data(), rhs.data(), lhs.data(),
             workspace.data(), workspace.size(), 3,
             runtime::CpuProviderNestingPolicyV1::native_only, &report) ==
             runtime::CpuParallelGemmStatusV1::alias_violation,
         "parallel output/input alias is rejected");
  expect(context->info().completed_submissions == 0,
         "preflight rejection submits no worker tasks");

  const auto submacro_problem = problem(64, 257, 1024, 64);
  AlignedBuffer submacro_lhs(64 * 257 * sizeof(float));
  AlignedBuffer submacro_rhs(257 * 1024 * sizeof(float));
  AlignedBuffer submacro_out(64 * 1024 * sizeof(float));
  std::fill(submacro_lhs.floats(), submacro_lhs.floats() + 64 * 257, 0.125F);
  std::fill(submacro_rhs.floats(), submacro_rhs.floats() + 257 * 1024,
            -0.25F);
  std::fill(submacro_out.floats(), submacro_out.floats() + 64 * 1024, 73.0F);
  runtime::CpuParallelGemmWorkspaceRequirementsV1 submacro_requirements;
  expect(runtime::cpu_parallel_packed_avx2_workspace_requirements_v1(
             submacro_problem, 4, &submacro_requirements) ==
             runtime::CpuParallelGemmStatusV1::success,
         "sub-MC rejection fixture workspace query succeeds");
  AlignedBuffer submacro_workspace(submacro_requirements.total_bytes);
  expect(runtime::cpu_execute_parallel_packed_avx2_v1(
             *context, submacro_problem, submacro_lhs.floats(),
             submacro_rhs.floats(), submacro_out.floats(),
             submacro_workspace.data(),
             submacro_workspace.size() - 1, 4,
             runtime::CpuProviderNestingPolicyV1::native_only, &report) ==
             runtime::CpuParallelGemmStatusV1::workspace_insufficient &&
             std::all_of(submacro_out.floats(),
                         submacro_out.floats() + 64 * 1024,
                         [](float value) { return value == 73.0F; }) &&
             context->info().completed_submissions == 0,
         "sub-MC four-worker execution fails before mutation or dispatch when "
         "workspace is insufficient");
}

void run_parallel_avx512_correctness() {
  if (!runtime::cpu_packed_avx512_runtime_usable_v1()) {
    std::cout << "parallel packed AVX-512 runtime unavailable: test skipped\n";
    return;
  }
  constexpr std::size_t m = 257;
  constexpr std::size_t k = 35;
  constexpr std::size_t n = 33;
  const auto gemm_problem = problem(m, k, n);
  std::vector<float> lhs(m * k);
  std::vector<float> rhs(k * n);
  std::vector<float> out(m * n, -3.0F);
  fill(lhs, 511);
  fill(rhs, 512);
  const auto expected = oracle(lhs, rhs, m, k, n);
  auto context = make_context();
  runtime::CpuParallelGemmWorkspaceRequirementsV1 requirements;
  expect(runtime::cpu_parallel_packed_avx512_workspace_requirements_v1(
             gemm_problem, 3, &requirements) ==
             runtime::CpuParallelGemmStatusV1::success,
         "parallel AVX-512 workspace query succeeds");
  AlignedBuffer workspace(requirements.total_bytes);
  runtime::CpuParallelGemmReportV1 report;
  expect(runtime::cpu_execute_parallel_packed_avx512_v1(
             *context, gemm_problem, lhs.data(), rhs.data(), out.data(),
             workspace.data(), workspace.size(), 3,
             runtime::CpuProviderNestingPolicyV1::native_only, &report) ==
             runtime::CpuParallelGemmStatusV1::success &&
             report.actual_threads == 3 && report.macro_tile_count == 3,
         "parallel AVX-512 packed execution succeeds on usable hardware");
  expect(close(out, expected),
         "parallel AVX-512 result matches double-precision oracle");

  struct TwoDimensionalCase {
    std::size_t m;
    std::size_t k;
    std::size_t n;
    std::size_t macro_tiles;
    std::size_t row_tasks;
    std::size_t column_tasks;
  };
  constexpr TwoDimensionalCase two_dimensional_cases[] = {
      {64, 257, 1024, 1, 1, 4},
      {129, 1024, 512, 2, 2, 2},
  };
  constexpr std::size_t guard_elements = 16;
  constexpr float guard_sentinel = -481.5F;
  for (const auto &test_case : two_dimensional_cases) {
    const auto two_dimensional_problem =
        problem(test_case.m, test_case.k, test_case.n, 64);
    const std::size_t lhs_elements = test_case.m * test_case.k;
    const std::size_t rhs_elements = test_case.k * test_case.n;
    const std::size_t out_elements = test_case.m * test_case.n;
    const std::size_t guarded_out_elements =
        out_elements + 2 * guard_elements;
    AlignedBuffer lhs_storage(lhs_elements * sizeof(float));
    AlignedBuffer rhs_storage(rhs_elements * sizeof(float));
    AlignedBuffer out_storage(guarded_out_elements * sizeof(float));
    float *two_dimensional_lhs = lhs_storage.floats();
    float *two_dimensional_rhs = rhs_storage.floats();
    float *guarded_out = out_storage.floats();
    float *two_dimensional_out = guarded_out + guard_elements;
    fill(two_dimensional_lhs, lhs_elements,
         static_cast<std::uint32_t>(1600 + test_case.m));
    fill(two_dimensional_rhs, rhs_elements,
         static_cast<std::uint32_t>(1700 + test_case.m));
    std::fill(guarded_out, guarded_out + guarded_out_elements,
              guard_sentinel);
    const auto two_dimensional_expected =
        oracle(two_dimensional_lhs, two_dimensional_rhs, test_case.m,
               test_case.k, test_case.n);
    runtime::CpuParallelGemmWorkspaceRequirementsV1
        two_dimensional_requirements;
    expect(runtime::cpu_parallel_packed_avx512_workspace_requirements_v1(
               two_dimensional_problem, 4, &two_dimensional_requirements) ==
               runtime::CpuParallelGemmStatusV1::success,
           "parallel AVX-512 2-D workspace query succeeds");
    AlignedBuffer two_dimensional_workspace(
        two_dimensional_requirements.total_bytes);
    runtime::CpuParallelGemmReportV1 two_dimensional_report;
    const auto two_dimensional_status =
        runtime::cpu_execute_parallel_packed_avx512_v1(
            *context, two_dimensional_problem, two_dimensional_lhs,
            two_dimensional_rhs, two_dimensional_out,
            two_dimensional_workspace.data(),
            two_dimensional_workspace.size(), 4,
            runtime::CpuProviderNestingPolicyV1::native_only,
            &two_dimensional_report);
    expect(two_dimensional_status ==
               runtime::CpuParallelGemmStatusV1::success,
           "parallel AVX-512 2-D execution succeeds");
    expect(two_dimensional_report.actual_threads == 4 &&
               two_dimensional_report.macro_tile_count ==
                   test_case.macro_tiles &&
               two_dimensional_report.row_task_count ==
                   test_case.row_tasks &&
               two_dimensional_report.column_task_count ==
                   test_case.column_tasks &&
               two_dimensional_report.task_count == 4,
           "parallel AVX-512 reports authenticated 2-D output partitions");
    expect(close(two_dimensional_out, out_elements,
                 two_dimensional_expected),
           "parallel AVX-512 2-D result matches double-precision oracle");
    expect(guards_unchanged(guarded_out, guarded_out_elements, guard_elements,
                            guard_sentinel),
           "parallel AVX-512 2-D execution preserves output guards");
  }
}

}  // namespace

int main() {
  run_parallel_correctness();
  run_balanced_partition_correctness();
  run_low_alignment_row_only_correctness();
  rejection_preserves_output();
  run_parallel_avx512_correctness();
  if (failures != 0) {
    std::cerr << failures << " parallel packed GEMM checks failed\n";
    return 1;
  }
  std::cout << "parallel packed AVX2 GEMM PASS\n";
  return 0;
}
