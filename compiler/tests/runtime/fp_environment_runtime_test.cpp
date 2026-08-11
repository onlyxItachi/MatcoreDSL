#include "matcore/runtime_c.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <string_view>
#include <vector>

#if defined(__linux__) && defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

#if defined(__linux__) && defined(__x86_64__)
class ScopedMxcsr {
 public:
  ScopedMxcsr() noexcept : saved_(_mm_getcsr()) {}
  ScopedMxcsr(const ScopedMxcsr &) = delete;
  ScopedMxcsr &operator=(const ScopedMxcsr &) = delete;
  ~ScopedMxcsr() { _mm_setcsr(saved_); }
  std::uint32_t saved() const noexcept { return saved_; }
  void set(std::uint32_t value) noexcept { _mm_setcsr(value); }

 private:
  std::uint32_t saved_ = 0;
};
#endif

class AlignedBytes {
 public:
  explicit AlignedBytes(std::size_t bytes)
      : bytes_(std::max<std::size_t>(bytes, 1U)),
        data_(static_cast<std::byte *>(::operator new(
            bytes_, std::align_val_t(64), std::nothrow))) {}
  AlignedBytes(const AlignedBytes &) = delete;
  AlignedBytes &operator=(const AlignedBytes &) = delete;
  ~AlignedBytes() { ::operator delete(data_, std::align_val_t(64)); }
  std::byte *data() const noexcept { return data_; }
  std::size_t size() const noexcept { return bytes_; }
  bool valid() const noexcept { return data_ != nullptr; }

 private:
  std::size_t bytes_ = 0;
  std::byte *data_ = nullptr;
};

matcore_tensor_desc_v0 tensor(void *data, std::int64_t rows,
                              std::int64_t columns,
                              matcore_mutability_v0 mutability) {
  matcore_tensor_desc_v0 result{};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = sizeof(result);
  result.data = data;
  result.dtype = MATCORE_DTYPE_F32_V0;
  result.rank = 2;
  result.dims[0] = rows;
  result.dims[1] = columns;
  result.strides[0] = columns;
  result.strides[1] = 1;
  result.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  result.mutability = mutability;
  return result;
}

matcore_policy_v0 policy() {
  matcore_policy_v0 result{};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = sizeof(result);
  result.target = MATCORE_TARGET_CPU_V0;
  result.fallback = MATCORE_FALLBACK_ERROR_V0;
  return result;
}

matcore_cpu_gemm_execution_options_v1 options(
    matcore_cpu_gemm_request_v2 request) {
  matcore_cpu_gemm_execution_options_v1 result{};
  result.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  result.struct_size = sizeof(result);
  result.request = request;
  result.requested_threads = 1;
  return result;
}

matcore_cpu_gemm_plan_report_v2 empty_report() {
  matcore_cpu_gemm_plan_report_v2 result{};
  result.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V2;
  result.struct_size = sizeof(result);
  return result;
}

matcore_packed_b_desc_v1 empty_packed_b() {
  matcore_packed_b_desc_v1 result{};
  result.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  result.struct_size = sizeof(result);
  return result;
}

struct Fixture {
  static constexpr std::int64_t m = 4;
  static constexpr std::int64_t n = 16;
  static constexpr std::int64_t k = 3;
  alignas(64) std::array<float, m * k> lhs{};
  alignas(64) std::array<float, k * n> rhs{};
  alignas(64) std::array<float, m * n> out{};
  matcore_tensor_desc_v0 out_desc;
  matcore_tensor_desc_v0 lhs_desc;
  matcore_tensor_desc_v0 rhs_desc;
  matcore_policy_v0 cpu_policy = policy();

  Fixture()
      : out_desc(tensor(out.data(), m, n,
                        MATCORE_MUTABILITY_READ_WRITE_V0)),
        lhs_desc(tensor(lhs.data(), m, k,
                        MATCORE_MUTABILITY_READ_ONLY_V0)),
        rhs_desc(tensor(rhs.data(), k, n,
                        MATCORE_MUTABILITY_READ_ONLY_V0)) {
    for (std::size_t index = 0; index < lhs.size(); ++index)
      lhs[index] = static_cast<float>((index % 7U) + 1U);
    for (std::size_t index = 0; index < rhs.size(); ++index)
      rhs[index] = static_cast<float>((index % 5U) + 1U);
    out.fill(-17.0F);
  }

  bool output_unchanged() const {
    return std::all_of(out.begin(), out.end(),
                       [](float value) { return value == -17.0F; });
  }
};

void public_execution_rejects_before_mutation() {
#if defined(__linux__) && defined(__x86_64__)
  Fixture fixture;
  ScopedMxcsr scope;
  scope.set(scope.saved() | (1U << 15U));

  expect(matcore_runtime_gemm_f32_v0(
             nullptr, &fixture.lhs_desc, &fixture.rhs_desc,
             &fixture.cpu_policy)
             .code == MATCORE_STATUS_INVALID_ARGUMENT_V0,
         "descriptor validation errors retain priority over FP-state rejection");
  auto rejected = matcore_runtime_gemm_f32_v0(
      &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
      &fixture.cpu_policy);
  expect(rejected.code ==
             MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0 &&
             fixture.output_unchanged(),
         "v0 GEMM rejects FTZ before output mutation");

  auto reference = options(MATCORE_CPU_GEMM_REQUEST_FORCE_REFERENCE_V2);
  auto report = empty_report();
  rejected = matcore_runtime_gemm_f32_execute_v1(
      &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
      &fixture.cpu_policy, &reference, nullptr, 0, &report);
  expect(rejected.code ==
             MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0 &&
             fixture.output_unchanged() && report.plan_status == 0,
         "workspace-aware v1 GEMM rejects FTZ before output/report mutation");
#endif
}

void prepacked_paths_reject_before_storage_mutation() {
#if defined(__linux__) && defined(__x86_64__)
  Fixture fixture;
  auto packed_options =
      options(MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V2);
  matcore_gemm_prepacked_b_requirements_v1 requirements{};
  requirements.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  requirements.struct_size = sizeof(requirements);
  const auto query = matcore_runtime_gemm_f32_prepacked_b_size_v1(
      &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
      &fixture.cpu_policy, &packed_options, &requirements);
  expect(query.code == MATCORE_STATUS_OK_V0,
         "prepacked-B requirements are available for the physical AVX2 host");
  if (query.code != MATCORE_STATUS_OK_V0) return;

  AlignedBytes packed(static_cast<std::size_t>(requirements.packed_b_bytes));
  AlignedBytes workspace(
      static_cast<std::size_t>(requirements.execution_workspace_bytes));
  expect(packed.valid() && workspace.valid(),
         "caller-owned packed and transient storage allocate");
  if (!packed.valid() || !workspace.valid()) return;
  std::memset(packed.data(), 0x5A, packed.size());
  std::memset(workspace.data(), 0x6B, workspace.size());
  const std::vector<std::byte> packed_before(packed.data(),
                                             packed.data() + packed.size());
  const std::vector<std::byte> workspace_before(
      workspace.data(), workspace.data() + workspace.size());
  auto packed_desc = empty_packed_b();

  matcore_gemm_workspace_requirements_v1 direct_requirements{};
  direct_requirements.abi_version = MATCORE_RUNTIME_EXECUTION_ABI_VERSION_V1;
  direct_requirements.struct_size = sizeof(direct_requirements);
  auto direct_query_report = empty_report();
  expect(matcore_runtime_gemm_f32_workspace_size_v1(
             &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
             &fixture.cpu_policy, &packed_options, &direct_requirements,
             &direct_query_report)
             .code == MATCORE_STATUS_OK_V0,
         "direct packed-v1 workspace requirement is available");
  AlignedBytes direct_workspace(
      static_cast<std::size_t>(direct_requirements.workspace_bytes));
  expect(direct_workspace.valid(),
         "direct packed-v1 caller workspace allocates");
  if (!direct_workspace.valid()) return;
  std::memset(direct_workspace.data(), 0x4C, direct_workspace.size());
  const std::vector<std::byte> direct_workspace_before(
      direct_workspace.data(),
      direct_workspace.data() + direct_workspace.size());
  auto direct_report = empty_report();
  {
    ScopedMxcsr scope;
    scope.set(scope.saved() | (1U << 15U));
    const auto rejected = matcore_runtime_gemm_f32_execute_v1(
        &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
        &fixture.cpu_policy, &packed_options, direct_workspace.data(),
        direct_workspace.size(), &direct_report);
    expect(rejected.code ==
               MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0 &&
               fixture.output_unchanged() &&
               std::equal(direct_workspace_before.begin(),
                          direct_workspace_before.end(),
                          direct_workspace.data()),
           "direct packed-v1 rejects FTZ before packing/output/workspace mutation");
  }

  {
    ScopedMxcsr scope;
    scope.set(scope.saved() | (1U << 15U));
    const auto rejected = matcore_runtime_gemm_f32_prepack_b_v1(
        &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
        &fixture.cpu_policy, &packed_options, packed.data(), packed.size(),
        &packed_desc);
    expect(rejected.code ==
               MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0 &&
               std::equal(packed_before.begin(), packed_before.end(),
                          packed.data()) &&
               packed_desc.source_data == nullptr,
           "prepack rejects FTZ before packed storage or descriptor mutation");
  }

  expect(matcore_runtime_gemm_f32_prepack_b_v1(
             &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
             &fixture.cpu_policy, &packed_options, packed.data(), packed.size(),
             &packed_desc)
             .code == MATCORE_STATUS_OK_V0,
         "clean environment prepares B for execution-path guard test");
  std::memset(workspace.data(), 0x6B, workspace.size());
  auto report = empty_report();
  {
    ScopedMxcsr scope;
    scope.set(scope.saved() | (1U << 15U));
    const auto rejected = matcore_runtime_gemm_f32_execute_prepacked_b_v1(
        &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
        &fixture.cpu_policy, &packed_options, &packed_desc, workspace.data(),
        workspace.size(), &report);
    expect(rejected.code ==
               MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0 &&
               fixture.output_unchanged() &&
               std::equal(workspace_before.begin(), workspace_before.end(),
                          workspace.data()),
           "prepacked execution rejects FTZ before output/workspace mutation");
  }
#endif
}

void status_flags_are_legality_neutral() {
#if defined(__linux__) && defined(__x86_64__)
  Fixture fixture;
  ScopedMxcsr scope;
  scope.set((scope.saved() & ~0x3FU) | 0x3FU);
  const auto result = matcore_runtime_gemm_f32_v0(
      &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
      &fixture.cpu_policy);
  expect(result.code == MATCORE_STATUS_OK_V0 && !fixture.output_unchanged(),
         "raised MXCSR status flags do not reject otherwise legal execution");
#endif
}

void context_creation_authenticates_worker_environment() {
#if defined(__linux__) && defined(__x86_64__)
  matcore_cpu_execution_context_options_v1 options{};
  options.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  options.struct_size = sizeof(options);
  options.requested_threads = 2;
  options.affinity_policy = MATCORE_CPU_AFFINITY_NONE_V1;
  options.numa_policy = MATCORE_CPU_NUMA_SINGLE_NODE_V1;
  options.smt_policy = MATCORE_CPU_SMT_ALLOW_V1;
  matcore_cpu_execution_context_report_v1 report{};
  report.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  report.struct_size = sizeof(report);
  matcore_cpu_execution_context_v1 *context = nullptr;
  ScopedMxcsr scope;
  scope.set(scope.saved() | (1U << 15U));
  const auto rejected = matcore_runtime_cpu_execution_context_create_v1(
      &options, &context, &report);
  expect(rejected.code ==
             MATCORE_STATUS_UNSUPPORTED_FLOATING_POINT_ENVIRONMENT_V0 &&
             context == nullptr,
         "persistent workers inheriting FTZ are rejected during creation");
#endif
}

}  // namespace

int main() {
  public_execution_rejects_before_mutation();
  prepacked_paths_reject_before_storage_mutation();
  status_flags_are_legality_neutral();
  context_creation_authenticates_worker_environment();
  if (failures != 0) return 1;
  std::cout << "runtime floating-point environment guards PASS\n";
  return 0;
}
