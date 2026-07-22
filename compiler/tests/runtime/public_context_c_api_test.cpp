#include "matcore/runtime_c.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <string_view>
#include <vector>

// This additive declaration is integrated into runtime_c.h by the integration
// owner. Keeping the test-side declaration here lets this isolated lane prove
// the implementation without editing the shared public header concurrently.
extern "C" MATCORE_RUNTIME_API matcore_status_v0
matcore_runtime_cpu_execution_context_query_v1(
    matcore_cpu_execution_context_v1 *context,
    matcore_cpu_execution_context_report_v1 *report) noexcept;

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

matcore_cpu_capabilities_v2 empty_capabilities() {
  matcore_cpu_capabilities_v2 result{};
  result.abi_version = MATCORE_RUNTIME_CAPABILITY_ABI_VERSION_V2;
  result.struct_size = sizeof(result);
  return result;
}

matcore_cpu_execution_context_report_v1 empty_context_report() {
  matcore_cpu_execution_context_report_v1 result{};
  result.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  result.struct_size = sizeof(result);
  return result;
}

matcore_cpu_gemm_plan_report_v3 empty_plan_report() {
  matcore_cpu_gemm_plan_report_v3 result{};
  result.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V3;
  result.struct_size = sizeof(result);
  return result;
}

matcore_gemm_workspace_requirements_v2 empty_workspace_requirements() {
  matcore_gemm_workspace_requirements_v2 result{};
  result.abi_version = MATCORE_RUNTIME_EXECUTION_OPTIONS_ABI_VERSION_V2;
  result.struct_size = sizeof(result);
  return result;
}

matcore_policy_v0 cpu_policy() {
  matcore_policy_v0 result{};
  result.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  result.struct_size = sizeof(result);
  result.target = MATCORE_TARGET_CPU_V0;
  result.fallback = MATCORE_FALLBACK_ERROR_V0;
  return result;
}

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

matcore_cpu_gemm_execution_options_v2 execution_options(
    matcore_cpu_gemm_request_v3 request, std::uint32_t threads,
    matcore_cpu_affinity_policy_v1 affinity = MATCORE_CPU_AFFINITY_NONE_V1,
    matcore_cpu_numa_policy_v1 numa = MATCORE_CPU_NUMA_SINGLE_NODE_V1) {
  matcore_cpu_gemm_execution_options_v2 result{};
  result.abi_version = MATCORE_RUNTIME_EXECUTION_OPTIONS_ABI_VERSION_V2;
  result.struct_size = sizeof(result);
  result.request = request;
  result.requested_threads = threads;
  result.affinity_policy = affinity;
  result.numa_policy = numa;
  return result;
}

class AlignedStorage {
 public:
  explicit AlignedStorage(std::size_t bytes)
      : bytes_(std::max<std::size_t>(bytes, 1U)),
        data_(::operator new(bytes_, std::align_val_t(64), std::nothrow)) {}
  AlignedStorage(const AlignedStorage &) = delete;
  AlignedStorage &operator=(const AlignedStorage &) = delete;
  ~AlignedStorage() { ::operator delete(data_, std::align_val_t(64)); }
  void *data() const { return data_; }
  std::size_t size() const { return bytes_; }

 private:
  std::size_t bytes_ = 0;
  void *data_ = nullptr;
};

struct GemmFixture {
  GemmFixture(std::int64_t input_m, std::int64_t input_n,
              std::int64_t input_k)
      : m(input_m), n(input_n), k(input_k),
        lhs(static_cast<std::size_t>(m * k)),
        rhs(static_cast<std::size_t>(k * n)),
        out(static_cast<std::size_t>(m * n), -777.0F),
        out_desc(tensor(out.data(), m, n, MATCORE_MUTABILITY_READ_WRITE_V0)),
        lhs_desc(tensor(lhs.data(), m, k, MATCORE_MUTABILITY_READ_ONLY_V0)),
        rhs_desc(tensor(rhs.data(), k, n, MATCORE_MUTABILITY_READ_ONLY_V0)) {
    for (std::size_t index = 0; index < lhs.size(); ++index)
      lhs[index] = static_cast<float>(static_cast<int>(index % 11U) - 5) /
                   8.0F;
    for (std::size_t index = 0; index < rhs.size(); ++index)
      rhs[index] = static_cast<float>(static_cast<int>(index % 13U) - 6) /
                   8.0F;
  }

  void reset(float value = -777.0F) {
    std::fill(out.begin(), out.end(), value);
  }

  bool unchanged(float value = -777.0F) const {
    return std::all_of(out.begin(), out.end(),
                       [value](float element) { return element == value; });
  }

  bool correct() const {
    for (std::int64_t row = 0; row < m; ++row) {
      for (std::int64_t column = 0; column < n; ++column) {
        double expected = 0.0;
        for (std::int64_t depth = 0; depth < k; ++depth) {
          expected += static_cast<double>(
                          lhs[static_cast<std::size_t>(row * k + depth)]) *
                      static_cast<double>(
                          rhs[static_cast<std::size_t>(depth * n + column)]);
        }
        const double actual =
            out[static_cast<std::size_t>(row * n + column)];
        const double tolerance = 2.0e-4 * std::max(1.0, std::abs(expected));
        if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
          return false;
      }
    }
    return true;
  }

  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
  std::vector<float> lhs;
  std::vector<float> rhs;
  std::vector<float> out;
  matcore_tensor_desc_v0 out_desc;
  matcore_tensor_desc_v0 lhs_desc;
  matcore_tensor_desc_v0 rhs_desc;
};

bool same_capabilities(const matcore_cpu_capabilities_v2 &left,
                       const matcore_cpu_capabilities_v2 &right) {
  return left.abi_version == right.abi_version &&
         left.struct_size == right.struct_size &&
         left.architecture == right.architecture &&
         left.detection_complete == right.detection_complete &&
         left.hardware_known_features == right.hardware_known_features &&
         left.hardware_available_features == right.hardware_available_features &&
         left.os_known_features == right.os_known_features &&
         left.os_available_features == right.os_available_features &&
         left.compiler_known_features == right.compiler_known_features &&
         left.compiler_available_features == right.compiler_available_features &&
         left.implementation_known_features ==
             right.implementation_known_features &&
         left.implementation_available_features ==
             right.implementation_available_features &&
         left.runtime_validation_known_features ==
             right.runtime_validation_known_features &&
         left.runtime_validated_features == right.runtime_validated_features &&
         left.os_xstate_mask == right.os_xstate_mask &&
         left.usable_vector_bits == right.usable_vector_bits &&
         left.os_xstate_mask_known == right.os_xstate_mask_known &&
         left.amx_permission_known == right.amx_permission_known &&
         left.amx_permission_granted == right.amx_permission_granted;
}

void capability_contract(matcore_cpu_capabilities_v2 *detected) {
  auto first = empty_capabilities();
  auto second = empty_capabilities();
  expect(matcore_runtime_query_cpu_capabilities_v2(&first).code ==
             MATCORE_STATUS_OK_V0,
         "capability v2 query succeeds");
  expect(matcore_runtime_query_cpu_capabilities_v2(&second).code ==
             MATCORE_STATUS_OK_V0 &&
             same_capabilities(first, second),
         "capability v2 query is deterministic");
  const std::uint64_t portable = MATCORE_CPU_FEATURE_PORTABLE_SCALAR_F32_V2;
  expect((first.implementation_available_features & portable) != 0 &&
             (first.runtime_validated_features & portable) != 0,
         "portable implementation and runtime validation are explicit");
  expect((first.runtime_validated_features &
          ~first.implementation_available_features) == 0,
         "runtime validation never exceeds implementation availability");

  auto malformed = empty_capabilities();
  malformed.reserved[0] = 1;
  expect(matcore_runtime_query_cpu_capabilities_v2(&malformed).code ==
             MATCORE_STATUS_INVALID_ARGUMENT_V0 &&
             malformed.reserved[0] == 1,
         "capability query rejects nonzero output fields without mutation");
  *detected = first;
}

matcore_cpu_execution_context_v1 *create_context(
    matcore_cpu_execution_context_report_v1 *created_report) {
  matcore_cpu_execution_context_options_v1 options{};
  options.abi_version = MATCORE_RUNTIME_EXECUTION_CONTEXT_ABI_VERSION_V1;
  options.struct_size = sizeof(options);
  options.requested_threads = 4;
  options.affinity_policy = MATCORE_CPU_AFFINITY_NONE_V1;
  options.numa_policy = MATCORE_CPU_NUMA_SINGLE_NODE_V1;
  matcore_cpu_execution_context_v1 *context = nullptr;
  *created_report = empty_context_report();
  const auto result = matcore_runtime_cpu_execution_context_create_v1(
      &options, &context, created_report);
  expect(result.code == MATCORE_STATUS_OK_V0 && context != nullptr,
         "public execution context is created");
  expect(created_report->requested_threads == 4 &&
             created_report->actual_worker_count > 0 &&
             created_report->actual_worker_count <= 4 &&
             created_report->persistent_worker_count ==
                 created_report->actual_worker_count &&
             created_report->execution_generation == 0,
         "creation report exposes fixed persistent worker state");

  auto invalid_options = options;
  invalid_options.reserved[0] = 1;
  matcore_cpu_execution_context_v1 *invalid_context = nullptr;
  auto invalid_report = empty_context_report();
  expect(matcore_runtime_cpu_execution_context_create_v1(
             &invalid_options, &invalid_context, &invalid_report)
                 .code == MATCORE_STATUS_INVALID_ARGUMENT_V0 &&
             invalid_context == nullptr && invalid_report.requested_threads == 0,
         "context creation rejects reserved options without publishing state");
  return context;
}

bool run_variant(matcore_cpu_execution_context_v1 *context,
                 GemmFixture &fixture,
                 matcore_cpu_gemm_request_v3 request,
                 std::uint32_t threads,
                 matcore_cpu_gemm_plan_report_v3 *executed_report = nullptr) {
  const auto options = execution_options(request, threads);
  auto requirements = empty_workspace_requirements();
  auto query_report = empty_plan_report();
  const auto policy = cpu_policy();
  const auto query = matcore_runtime_gemm_f32_context_workspace_size_v2(
      context, &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
      &policy, &options, &requirements, &query_report);
  if (query.code != MATCORE_STATUS_OK_V0) return false;
  AlignedStorage workspace(static_cast<std::size_t>(requirements.workspace_bytes));
  expect(workspace.data() != nullptr, "test workspace allocation succeeds");
  if (workspace.data() == nullptr) return false;
  auto report = empty_plan_report();
  const auto execution = matcore_runtime_gemm_f32_execute_context_v2(
      context, &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
      &policy, &options, workspace.data(),
      static_cast<std::size_t>(requirements.workspace_bytes), &report);
  expect(execution.code == MATCORE_STATUS_OK_V0,
         "forced public context variant executes");
  expect(report.plan_status == MATCORE_CPU_PLAN_STATUS_SELECTED_V1 &&
             report.selected_stable_id != nullptr &&
             requirements.selected_stable_id != nullptr &&
             std::strcmp(report.selected_stable_id,
                         requirements.selected_stable_id) == 0,
         "workspace query and execution select the same stable variant");
  expect(fixture.correct(), "forced public context variant is numerically correct");
  if (executed_report != nullptr) *executed_report = report;
  return execution.code == MATCORE_STATUS_OK_V0;
}

void strict_preflight_and_old_abi(matcore_cpu_execution_context_v1 *context) {
  GemmFixture fixture(33, 35, 37);
  const auto policy = cpu_policy();
  auto options = execution_options(
      MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V3, 1);
  auto requirements = empty_workspace_requirements();
  auto report = empty_plan_report();
  const auto query = matcore_runtime_gemm_f32_context_workspace_size_v2(
      context, &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
      &policy, &options, &requirements, &report);
  if (query.code == MATCORE_STATUS_OK_V0) {
    expect(requirements.workspace_bytes > 0 &&
               requirements.workspace_alignment >= 32,
           "packed query exposes explicit caller workspace");
    AlignedStorage workspace(
        static_cast<std::size_t>(requirements.workspace_bytes) + 64U);
    fixture.reset();
    auto too_small_report = empty_plan_report();
    expect(matcore_runtime_gemm_f32_execute_context_v2(
               context, &fixture.out_desc, &fixture.lhs_desc,
               &fixture.rhs_desc, &policy, &options, workspace.data(),
               static_cast<std::size_t>(requirements.workspace_bytes - 1U),
               &too_small_report)
                   .code == MATCORE_STATUS_INSUFFICIENT_WORKSPACE_V0 &&
               fixture.unchanged(),
           "insufficient workspace fails before output mutation");

    fixture.reset();
    auto misaligned_report = empty_plan_report();
    auto *misaligned = static_cast<std::byte *>(workspace.data()) + 1;
    expect(matcore_runtime_gemm_f32_execute_context_v2(
               context, &fixture.out_desc, &fixture.lhs_desc,
               &fixture.rhs_desc, &policy, &options, misaligned,
               static_cast<std::size_t>(requirements.workspace_bytes),
               &misaligned_report)
                   .code == MATCORE_STATUS_INVALID_ALIGNMENT_V0 &&
               fixture.unchanged(),
           "misaligned workspace fails before output mutation");
  }

  fixture.reset();
  auto unavailable = execution_options(
      MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PARALLEL_AVX2_FMA_V3, 1);
  auto unavailable_requirements = empty_workspace_requirements();
  auto unavailable_report = empty_plan_report();
  expect(matcore_runtime_gemm_f32_context_workspace_size_v2(
             context, &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
             &policy, &unavailable, &unavailable_requirements,
             &unavailable_report)
                 .code == MATCORE_STATUS_UNAVAILABLE_VARIANT_V0 &&
             unavailable_report.plan_status ==
                 MATCORE_CPU_PLAN_STATUS_FORCED_VARIANT_ILLEGAL_V1 &&
             fixture.unchanged(),
         "forced illegal parallel variant fails without fallback or output mutation");

  auto invalid = options;
  invalid.reserved[0] = 1;
  auto invalid_requirements = empty_workspace_requirements();
  auto invalid_report = empty_plan_report();
  expect(matcore_runtime_gemm_f32_context_workspace_size_v2(
             context, &fixture.out_desc, &fixture.lhs_desc, &fixture.rhs_desc,
             &policy, &invalid, &invalid_requirements, &invalid_report)
                 .code == MATCORE_STATUS_INVALID_ARGUMENT_V0 &&
             invalid_report.planner_version == 0 &&
             invalid_requirements.workspace_bytes == 0,
         "v2 option reserved fields fail before output report mutation");

  GemmFixture legacy(7, 5, 9);
  expect(matcore_runtime_gemm_f32_v0(&legacy.out_desc, &legacy.lhs_desc,
                                     &legacy.rhs_desc, &policy)
                 .code == MATCORE_STATUS_OK_V0 &&
             legacy.correct(),
         "pre-existing one-shot C ABI remains compatible");
}

void runtime_variants_and_generation(
    matcore_cpu_execution_context_v1 *context,
    const matcore_cpu_execution_context_report_v1 &created,
    const matcore_cpu_capabilities_v2 &capabilities) {
  const std::uint64_t avx2_fma =
      MATCORE_CPU_FEATURE_AVX2_V2 | MATCORE_CPU_FEATURE_FMA_V2;
  if ((capabilities.runtime_validated_features & avx2_fma) == avx2_fma) {
    GemmFixture packed_avx2(33, 35, 37);
    expect(run_variant(
               context, packed_avx2,
               MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX2_FMA_V3, 1),
           "physically runtime-validated packed AVX2 path is forceable");
  }

  const std::uint64_t avx512_fma =
      MATCORE_CPU_FEATURE_AVX512F_V2 | MATCORE_CPU_FEATURE_FMA_V2;
  if ((capabilities.runtime_validated_features & avx512_fma) == avx512_fma) {
    GemmFixture packed_avx512(35, 33, 37);
    expect(run_variant(
               context, packed_avx512,
               MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PACKED_AVX512_FMA_V3, 1),
           "physically runtime-validated packed AVX-512 path is forceable");
  }

  auto before = empty_context_report();
  expect(matcore_runtime_cpu_execution_context_query_v1(context, &before).code ==
             MATCORE_STATUS_OK_V0 &&
             before.execution_generation == 0,
         "nonparallel executions do not submit persistent native workers");

  if (created.actual_worker_count >= 2 &&
      (capabilities.runtime_validated_features & avx2_fma) == avx2_fma) {
    GemmFixture parallel(257, 128, 128);
    expect(run_variant(
               context, parallel,
               MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PARALLEL_AVX2_FMA_V3, 2),
           "parallel AVX2 executes through the public persistent context");
    parallel.reset();
    expect(run_variant(
               context, parallel,
               MATCORE_CPU_GEMM_REQUEST_FORCE_NATIVE_PARALLEL_AVX2_FMA_V3, 2),
           "the same public persistent context executes repeatedly");
    auto after = empty_context_report();
    expect(matcore_runtime_cpu_execution_context_query_v1(context, &after).code ==
               MATCORE_STATUS_OK_V0 &&
               after.execution_generation == 2 &&
               after.persistent_worker_count == before.persistent_worker_count,
           "public query proves submissions advance without worker recreation");

    GemmFixture provider(64, 64, 64);
    const bool provider_ran = run_variant(
        context, provider,
        MATCORE_CPU_GEMM_REQUEST_FORCE_EXTERNAL_OPENBLAS_V3, 2);
    auto after_provider = empty_context_report();
    expect(matcore_runtime_cpu_execution_context_query_v1(
               context, &after_provider)
                   .code == MATCORE_STATUS_OK_V0,
           "context remains queryable after provider attempt");
    if (provider_ran) {
      expect(after_provider.execution_generation == after.execution_generation,
             "OpenBLAS executes outside the native pool without nesting");
    }
  }

  auto malformed = empty_context_report();
  malformed.reserved[0] = 1;
  expect(matcore_runtime_cpu_execution_context_query_v1(context, &malformed)
                 .code == MATCORE_STATUS_INVALID_ARGUMENT_V0 &&
             malformed.reserved[0] == 1,
         "context query enforces strict empty report input");
}

}  // namespace

int main() {
  matcore_cpu_capabilities_v2 capabilities{};
  capability_contract(&capabilities);
  matcore_cpu_execution_context_report_v1 created{};
  matcore_cpu_execution_context_v1 *context = create_context(&created);
  if (context != nullptr) {
    strict_preflight_and_old_abi(context);
    runtime_variants_and_generation(context, created, capabilities);
    expect(matcore_runtime_cpu_execution_context_destroy_v1(context).code ==
               MATCORE_STATUS_OK_V0,
           "public execution context destroys and joins workers");
  }
  expect(matcore_runtime_cpu_execution_context_destroy_v1(nullptr).code ==
             MATCORE_STATUS_INVALID_EXECUTION_CONTEXT_V0,
         "null context destruction is rejected cleanly");
  if (failures != 0) {
    std::cerr << failures << " public context C ABI checks failed\n";
    return 1;
  }
  std::cout << "public context C ABI PASS\n";
  return 0;
}
