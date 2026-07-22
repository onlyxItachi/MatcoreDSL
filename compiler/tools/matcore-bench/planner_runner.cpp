#include "benchmark.h"

#include "cpu_backend_registry.h"
#include "cpu_capability_v2.h"
#include "cpu_execution_context.h"
#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"
#include "cpu_packed_avx512.h"
#include "cpu_parallel_gemm.h"
#include "cpu_planner_v3.h"
#include "cpu_planner_v3_resources.h"
#include "cpu_topology_v1.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

namespace matcore::mdslc::bench {
namespace {

namespace planner = matcore::mdslc::planner;
namespace platform = matcore::mdslc::platform;
namespace runtime = matcore::mdslc::runtime;

planner::CpuGemmProblemV1 problem(const GemmShapeV1 &shape,
                                 std::uint32_t alignment) noexcept {
  return {shape.m,
          shape.n,
          shape.k,
          planner::CpuScalarTypeV1::f32,
          planner::CpuScalarTypeV1::f32,
          planner::CpuLayoutV1::row_major_contiguous,
          alignment};
}

bool request_for(std::string_view stable_id,
                 planner::CpuGemmRequestV3 &request) noexcept {
  if (stable_id == "auto") {
    request = planner::CpuGemmRequestV3::automatic;
  } else if (stable_id == "cpu.reference.f32.v1") {
    request = planner::CpuGemmRequestV3::force_reference;
  } else if (stable_id == "cpu.tiled.f32.v1") {
    request = planner::CpuGemmRequestV3::force_tiled;
  } else if (stable_id == "cpu.compiler-vectorized.avx2-fma.f32.v1") {
    request = planner::CpuGemmRequestV3::force_compiler_vectorized;
  } else if (stable_id == "cpu.external.openblas.f32.v1") {
    request = planner::CpuGemmRequestV3::force_external_openblas;
  } else if (stable_id == "cpu.native-packed.avx2-fma.f32.v1") {
    request = planner::CpuGemmRequestV3::force_native_packed_avx2_fma;
  } else if (stable_id == "cpu.native-packed.avx512-fma.f32.v1") {
    request = planner::CpuGemmRequestV3::force_native_packed_avx512_fma;
  } else if (stable_id == "cpu.native-parallel.avx2-fma.f32.v1") {
    request = planner::CpuGemmRequestV3::force_native_parallel_avx2_fma;
  } else if (stable_id == "cpu.native-parallel.avx512-fma.f32.v1") {
    request = planner::CpuGemmRequestV3::force_native_parallel_avx512_fma;
  } else {
    return false;
  }
  return true;
}

planner::CpuGemmRequestV1 legacy_request(
    planner::CpuGemmVariantV3 variant) noexcept {
  switch (variant) {
    case planner::CpuGemmVariantV3::reference:
      return planner::CpuGemmRequestV1::force_reference;
    case planner::CpuGemmVariantV3::tiled:
      return planner::CpuGemmRequestV1::force_tiled;
    case planner::CpuGemmVariantV3::compiler_vectorized:
      return planner::CpuGemmRequestV1::force_compiler_vectorized;
    case planner::CpuGemmVariantV3::external_openblas:
    case planner::CpuGemmVariantV3::native_packed_avx2_fma:
    case planner::CpuGemmVariantV3::native_packed_avx512_fma:
    case planner::CpuGemmVariantV3::native_parallel_avx2_fma:
    case planner::CpuGemmVariantV3::native_parallel_avx512_fma:
      return planner::CpuGemmRequestV1::force_reference;
  }
  return planner::CpuGemmRequestV1::force_reference;
}

struct ComputeOnlyPackedLayout {
  std::size_t packed_a_offset = 0;
  std::size_t packed_a_bytes = 0;
  std::size_t packed_b_offset = 0;
  std::size_t packed_b_bytes = 0;
  std::size_t total_bytes = 0;
};

bool checked_multiply(std::size_t lhs, std::size_t rhs,
                      std::size_t &result) noexcept {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool checked_add(std::size_t lhs, std::size_t rhs,
                 std::size_t &result) noexcept {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
  result = lhs + rhs;
  return true;
}

bool checked_round_up(std::size_t value, std::size_t multiple,
                      std::size_t &result) noexcept {
  if (multiple == 0) return false;
  const std::size_t remainder = value % multiple;
  return remainder == 0
             ? (result = value, true)
             : checked_add(value, multiple - remainder, result);
}

bool compute_only_layout(const GemmShapeV1 &shape,
                         ComputeOnlyPackedLayout &result) noexcept {
  if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0 ||
      static_cast<std::uint64_t>(shape.m) >
          std::numeric_limits<std::size_t>::max() ||
      static_cast<std::uint64_t>(shape.n) >
          std::numeric_limits<std::size_t>::max() ||
      static_cast<std::uint64_t>(shape.k) >
          std::numeric_limits<std::size_t>::max())
    return false;
  const auto m = static_cast<std::size_t>(shape.m);
  const auto n = static_cast<std::size_t>(shape.n);
  const auto k = static_cast<std::size_t>(shape.k);
  std::size_t padded_m = 0;
  std::size_t padded_n = 0;
  std::size_t packed_a_elements = 0;
  std::size_t packed_b_elements = 0;
  ComputeOnlyPackedLayout layout;
  if (!checked_round_up(m, runtime::kCpuPackedGemmMrV1, padded_m) ||
      !checked_round_up(n, runtime::kCpuPackedGemmNrV1, padded_n) ||
      !checked_multiply(padded_m, k, packed_a_elements) ||
      !checked_multiply(padded_n, k, packed_b_elements) ||
      !checked_multiply(packed_a_elements, sizeof(float),
                        layout.packed_a_bytes) ||
      !checked_multiply(packed_b_elements, sizeof(float),
                        layout.packed_b_bytes) ||
      !checked_round_up(layout.packed_a_bytes,
                        runtime::kCpuPackedGemmWorkspaceAlignmentV1,
                        layout.packed_b_offset) ||
      !checked_add(layout.packed_b_offset, layout.packed_b_bytes,
                   layout.total_bytes))
    return false;
  result = layout;
  return true;
}

void pack_compute_only_a(const GemmShapeV1 &shape, const float *lhs,
                         float *packed_a) noexcept {
  const auto m = static_cast<std::size_t>(shape.m);
  const auto k = static_cast<std::size_t>(shape.k);
  std::size_t destination = 0;
  for (std::size_t row = 0; row < m;
       row += runtime::kCpuPackedGemmMrV1) {
    for (std::size_t p = 0; p < k; ++p) {
      for (std::size_t lane = 0; lane < runtime::kCpuPackedGemmMrV1; ++lane) {
        packed_a[destination++] =
            row + lane < m ? lhs[(row + lane) * k + p] : 0.0F;
      }
    }
  }
}

void pack_compute_only_b(const GemmShapeV1 &shape, const float *rhs,
                         float *packed_b) noexcept {
  const auto n = static_cast<std::size_t>(shape.n);
  const auto k = static_cast<std::size_t>(shape.k);
  std::size_t destination = 0;
  for (std::size_t column = 0; column < n;
       column += runtime::kCpuPackedGemmNrV1) {
    for (std::size_t p = 0; p < k; ++p) {
      for (std::size_t lane = 0; lane < runtime::kCpuPackedGemmNrV1; ++lane) {
        packed_b[destination++] =
            column + lane < n ? rhs[p * n + column + lane] : 0.0F;
      }
    }
  }
}

bool numerical_packed_self_test(bool avx512) noexcept {
  if (avx512 ? !runtime::cpu_packed_avx512_runtime_usable_v1()
             : !runtime::cpu_packed_avx2_runtime_usable_v1())
    return false;
  const auto test_problem = problem({2, 3, 2}, 64);
  runtime::CpuPackedGemmWorkspaceRequirementsV1 requirements;
  const auto query = avx512
      ? runtime::cpu_packed_avx512_workspace_requirements_v1(
            test_problem,
            runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
            &requirements)
      : runtime::cpu_packed_avx2_workspace_requirements_v1(
            test_problem,
            runtime::CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
            &requirements);
  alignas(64) std::array<std::byte, 4096> workspace{};
  if (query != runtime::CpuPackedGemmStatusV1::success ||
      requirements.total_bytes > workspace.size())
    return false;
  alignas(64) const std::array<float, 4> lhs{{1.0F, 2.0F, 3.0F, 4.0F}};
  alignas(64) const std::array<float, 6> rhs{
      {5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F}};
  alignas(64) std::array<float, 6> output{};
  const auto status = avx512
      ? runtime::cpu_execute_packed_avx512_v1(
            test_problem, lhs.data(), rhs.data(), output.data(),
            workspace.data(), workspace.size())
      : runtime::cpu_execute_packed_avx2_v1(
            test_problem, lhs.data(), rhs.data(), output.data(),
            workspace.data(), workspace.size());
  const std::array<float, 6> expected{
      {21.0F, 24.0F, 27.0F, 47.0F, 54.0F, 61.0F}};
  if (status != runtime::CpuPackedGemmStatusV1::success) return false;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (!std::isfinite(output[index]) || output[index] != expected[index])
      return false;
  }
  return true;
}

platform::CpuImplementationAvailabilityV2 benchmark_implementation_evidence() noexcept {
  platform::CpuImplementationAvailabilityV2 result;
  result.compiled.known = platform::kKnownCpuFeatureBitsV2;
  result.runtime_validated.known = platform::kKnownCpuFeatureBitsV2;
  const auto portable =
      platform::feature_bit(platform::CpuFeatureV2::portable_scalar_f32);
  result.compiled.available = portable;
  result.runtime_validated.available = portable;
  if (runtime::cpu_packed_avx2_build_available_v1()) {
    result.compiled.available |=
        platform::feature_bit(platform::CpuFeatureV2::avx2) |
        platform::feature_bit(platform::CpuFeatureV2::fma);
  }
  if (numerical_packed_self_test(false)) {
    result.runtime_validated.available |=
        platform::feature_bit(platform::CpuFeatureV2::avx2) |
        platform::feature_bit(platform::CpuFeatureV2::fma);
  }
  if (runtime::cpu_packed_avx512_build_available_v1()) {
    result.compiled.available |=
        platform::feature_bit(platform::CpuFeatureV2::avx512f) |
        platform::feature_bit(platform::CpuFeatureV2::fma);
  }
  if (numerical_packed_self_test(true)) {
    result.runtime_validated.available |=
        platform::feature_bit(platform::CpuFeatureV2::avx512f) |
        platform::feature_bit(platform::CpuFeatureV2::fma);
  }
  return result;
}

platform::CpuTopologyV1 discover_benchmark_topology(
    platform::ArchitectureKindV1 architecture) {
#if defined(__linux__)
  (void)architecture;
  return platform::discover_linux_cpu_topology_v1();
#else
  platform::CpuTopologyV1 result;
  result.architecture = architecture;
  return result;
#endif
}

std::uint32_t inherited_available_processor_count() noexcept {
#if defined(__linux__)
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (::sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
    const int count = CPU_COUNT(&allowed);
    if (count > 0) return static_cast<std::uint32_t>(count);
  }
#endif
  return std::max<std::uint32_t>(std::thread::hardware_concurrency(), 1);
}

bool is_single_packed(planner::CpuGemmVariantV3 variant) noexcept {
  return variant == planner::CpuGemmVariantV3::native_packed_avx2_fma ||
         variant == planner::CpuGemmVariantV3::native_packed_avx512_fma;
}

bool is_parallel_packed(planner::CpuGemmVariantV3 variant) noexcept {
  return variant == planner::CpuGemmVariantV3::native_parallel_avx2_fma ||
         variant == planner::CpuGemmVariantV3::native_parallel_avx512_fma;
}

bool is_avx512_packed(planner::CpuGemmVariantV3 variant) noexcept {
  return variant == planner::CpuGemmVariantV3::native_packed_avx512_fma ||
         variant == planner::CpuGemmVariantV3::native_parallel_avx512_fma;
}

struct BackendPlanState final : RunnerPlanStateV1 {
  BackendPlanState(planner::CpuGemmPlanV3 value, PackingModeV1 packing,
                   ComputeOnlyPackedLayout compute_layout)
      : plan(std::move(value)),
        packing_mode(packing),
        compute_only_layout(compute_layout) {}
  planner::CpuGemmPlanV3 plan;
  PackingModeV1 packing_mode = PackingModeV1::include;
  ComputeOnlyPackedLayout compute_only_layout;
  mutable runtime::CpuPackedBViewV1 packed_b;
  mutable bool packed_b_ready = false;
  mutable const float *compute_only_lhs = nullptr;
  mutable const float *compute_only_rhs = nullptr;
  mutable const std::byte *compute_only_workspace = nullptr;
  mutable bool compute_only_ready = false;
};

class PlannerRunner final : public GemmRunnerV1 {
 public:
  PlannerRunner()
      : capabilities_(platform::discover_cpu_capabilities_v2(
            benchmark_implementation_evidence())),
        topology_(discover_benchmark_topology(capabilities_.architecture)) {
    available_processors_ = inherited_available_processor_count();
    std::uint32_t capacity = platform::logical_cpu_count_v1(topology_);
    if (capacity != 0)
      available_processors_ = std::min(available_processors_, capacity);
    if (capacity == 0) capacity = available_processors_;
    capacity = std::min(capacity, available_processors_);
    capacity = std::max<std::uint32_t>(capacity, 1);
    runtime::CpuExecutionContextConfigV1 config;
    config.requested_threads = capacity;
    config.maximum_threads = capacity;
    runtime::CpuExecutionStatusV1 status{};
    execution_context_ = runtime::CpuExecutionContextV1::create(config, &status);
    execution_context_status_ = status;
  }

  RunnerEnvironmentV1 environment() const override {
    const auto probe_problem = problem({1, 1, 1}, alignof(float));
    const auto baseline =
        runtime::discover_cpu_gemm_implementation_resources_v1(probe_problem,
                                                                1);
    runtime::CpuRuntimeValidationEvidenceV1 evidence;
    evidence.packed_avx2_f32_runtime_validated =
        platform::has_runtime_validated_feature_v2(
            capabilities_, platform::CpuFeatureV2::avx2) &&
        platform::has_runtime_validated_feature_v2(
            capabilities_, platform::CpuFeatureV2::fma);
    evidence.packed_avx512_f32_runtime_validated =
        platform::has_runtime_validated_feature_v2(
            capabilities_, platform::CpuFeatureV2::avx512f) &&
        platform::has_runtime_validated_feature_v2(
            capabilities_, platform::CpuFeatureV2::fma);
    const auto resources = runtime::augment_cpu_gemm_implementation_resources_v2(
        probe_problem, baseline, execution_context_.get(), evidence);
    planner::CpuThreadPolicyV1 thread_policy;
    const auto probe = planner::plan_cpu_gemm_v3(
        probe_problem, capabilities_, topology_, thread_policy, resources,
        planner::CpuGemmRequestV3::automatic, available_processors_);
    std::array<char, 8192> diagnostic{};
    planner::format_cpu_gemm_plan_v3(probe, diagnostic.data(),
                                     diagnostic.size());
    const auto provider = runtime::openblas_provider_info_v1();
    RunnerEnvironmentV1 result;
    result.capability_record = platform::format_cpu_capabilities_v2(capabilities_);
    result.capability_record += "\nplanner_probe=";
    result.capability_record += diagnostic.data();
    result.capability_runtime_validation_source =
        "benchmark-process numerical self-test: tiny aligned packed GEMM per "
        "usable implemented ISA; every emitted measurement is independently "
        "checked by a double-precision oracle; evidence is local to this "
        "benchmark run";
    result.topology_record = platform::format_cpu_topology_v1(topology_);
    result.capability_record_version = capabilities_.version;
    result.topology_record_version = topology_.version;
    const auto topology_validation = platform::validate_cpu_topology_v1(topology_);
    result.topology_discovery_complete =
        topology_validation.valid && topology_.discovery_complete;
    result.logical_processors = platform::logical_cpu_count_v1(topology_);
    result.physical_cores = platform::physical_core_count_v1(topology_);
    result.numa_nodes = platform::numa_node_count_v1(topology_);
    if (execution_context_) {
      const auto info = execution_context_->info();
      result.persistent_execution_context = info.accepting_work;
      result.execution_context_workers = info.actual_worker_count;
      result.execution_context_workers_started = info.workers_started;
      result.execution_context_submissions = info.completed_submissions;
    }
    result.available_processors = available_processors_;
    result.worker_affinity_applied = false;
    result.worker_affinity_source =
        "inherited process affinity/cpuset constrains worker capacity; "
        "individual workers are not pinned by this benchmark backend";
    if (!execution_context_) {
      result.capability_record += "\nexecution_context_status=";
      result.capability_record +=
          runtime::cpu_execution_status_message_v1(execution_context_status_);
    }
    result.provider_name = provider.linked ? "OpenBLAS" : "unavailable";
    result.provider_version = provider.package_version;
    result.provider_config = provider.runtime_config;
    return result;
  }

  std::vector<std::string> variant_ids() const override {
    std::vector<std::string> result;
    for (const auto &record : planner::kCpuGemmVariantRegistryV3)
      result.emplace_back(record.stable_id);
    return result;
  }

  RunnerPlanV1 plan(const GemmShapeV1 &shape,
                    std::uint32_t minimum_alignment,
                    std::uint32_t requested_threads,
                    std::string_view requested_variant,
                    PackingModeV1 packing_mode, SmtPolicyV2 smt_policy,
                    AffinityPolicyV2 affinity_policy) const override {
    RunnerPlanV1 result;
    result.smt_policy = std::string(smt_policy_name_v2(smt_policy));
    result.affinity_policy =
        std::string(affinity_policy_name_v2(affinity_policy));
    result.worker_affinity_applied = false;
    result.affinity_diagnostic =
        "inherited process mask only; benchmark worker binding is unavailable";
    planner::CpuGemmRequestV3 request{};
    if (!request_for(requested_variant, request)) {
      result.reason = "requested stable variant ID is not registered";
      return result;
    }
    const auto gemm_problem = problem(shape, minimum_alignment);
    const auto baseline =
        runtime::discover_cpu_gemm_implementation_resources_v1(
            gemm_problem, requested_threads);
    runtime::CpuRuntimeValidationEvidenceV1 evidence;
    evidence.packed_avx2_f32_runtime_validated =
        platform::has_runtime_validated_feature_v2(
            capabilities_, platform::CpuFeatureV2::avx2) &&
        platform::has_runtime_validated_feature_v2(
            capabilities_, platform::CpuFeatureV2::fma);
    evidence.packed_avx512_f32_runtime_validated =
        platform::has_runtime_validated_feature_v2(
            capabilities_, platform::CpuFeatureV2::avx512f) &&
        platform::has_runtime_validated_feature_v2(
            capabilities_, platform::CpuFeatureV2::fma);
    const auto resources = runtime::augment_cpu_gemm_implementation_resources_v2(
        gemm_problem, baseline, execution_context_.get(), evidence);
    planner::CpuThreadPolicyV1 thread_policy;
    thread_policy.requested_threads = requested_threads;
    thread_policy.allow_smt = smt_policy == SmtPolicyV2::allow_smt;
    if (execution_context_)
      thread_policy.maximum_threads =
          execution_context_->info().actual_worker_count;
    auto selected = planner::plan_cpu_gemm_v3(
        gemm_problem, capabilities_, topology_, thread_policy, resources,
        request, available_processors_);
    result.legal = selected.status == planner::CpuPlanStatusV1::selected;
    result.planner_version = selected.planner_version;
    result.selected_variant = std::string(selected.selected_id);
    result.reason = std::string(selected.selection_reason);
    if (!result.legal && request != planner::CpuGemmRequestV3::automatic) {
      const std::size_t candidate_index =
          static_cast<std::size_t>(request) - 1U;
      if (candidate_index < selected.candidates.size() &&
          !selected.candidates[candidate_index].reason.empty())
        result.reason = std::string(selected.candidates[candidate_index].reason);
    }
    if (result.legal && packing_mode == PackingModeV1::exclude &&
        selected.selected_variant !=
            planner::CpuGemmVariantV3::native_packed_avx2_fma) {
      result.legal = false;
      result.reason =
          "--exclude-packing is a diagnostic implemented only for "
          "cpu.native-packed.avx2-fma.f32.v1; other implementations expose "
          "only complete calls with no separable benchmark-managed packing";
    }
    if (result.legal && affinity_policy != AffinityPolicyV2::none) {
      result.legal = false;
      result.reason =
          "requested benchmark worker-affinity policy is not implemented by "
          "the current execution-context backend; inherited process affinity "
          "is reported separately and is not worker binding";
    }
    ComputeOnlyPackedLayout compute_layout;
    if (result.legal) {
      const std::size_t selected_index =
          static_cast<std::size_t>(selected.selected_variant);
      auto &candidate = selected.candidates[selected_index];
      result.actual_threads = candidate.actual_threads;
      result.workspace_bytes = candidate.required_workspace_bytes;
      result.shared_workspace_bytes = candidate.shared_workspace_bytes;
      result.per_worker_workspace_bytes = candidate.per_worker_workspace_bytes;
      result.workspace_alignment = candidate.required_workspace_alignment;
      result.packing_required = is_single_packed(selected.selected_variant) ||
                                is_parallel_packed(selected.selected_variant);
      result.supports_prepacked_b = is_single_packed(selected.selected_variant);
      result.persistent_execution_context =
          is_parallel_packed(selected.selected_variant);
      if (selected.selected_variant ==
              planner::CpuGemmVariantV3::native_packed_avx2_fma &&
          packing_mode == PackingModeV1::exclude) {
        if (!compute_only_layout(shape, compute_layout)) {
          result.legal = false;
          result.reason =
              "native packed compute-only workspace requirement overflows";
        } else {
          result.workspace_bytes = compute_layout.total_bytes;
          result.shared_workspace_bytes = compute_layout.total_bytes;
          result.per_worker_workspace_bytes = 0;
          result.workspace_alignment = static_cast<std::uint32_t>(
              runtime::kCpuPackedGemmWorkspaceAlignmentV1);
          candidate.required_workspace_bytes = compute_layout.total_bytes;
          candidate.required_workspace_alignment = result.workspace_alignment;
          result.timing_scope =
              "packed-compute-only: A and B packing prepared before timing; "
              "timed region includes AVX2/FMA microkernel dispatch, tail "
              "handling, and output stores";
          result.complete_implementation_comparison = false;
        }
      } else if (is_single_packed(selected.selected_variant) &&
                 packing_mode == PackingModeV1::prepack_b) {
        runtime::CpuPackedGemmWorkspaceRequirementsV1 packed_requirements;
        const bool avx512 = is_avx512_packed(selected.selected_variant);
        const auto packed_status = avx512
            ? runtime::cpu_packed_avx512_prepacked_b_requirements_v1(
                  gemm_problem, &packed_requirements)
            : runtime::cpu_packed_avx2_prepacked_b_requirements_v1(
                  gemm_problem, &packed_requirements);
        runtime::CpuPackedGemmWorkspaceRequirementsV1 execute_requirements;
        const auto execute_status = avx512
            ? runtime::cpu_packed_avx512_workspace_requirements_v1(
                  gemm_problem,
                  runtime::CpuPackedGemmWorkspaceModeV1::
                      transient_a_with_prepacked_b,
                  &execute_requirements)
            : runtime::cpu_packed_avx2_workspace_requirements_v1(
                  gemm_problem,
                  runtime::CpuPackedGemmWorkspaceModeV1::
                      transient_a_with_prepacked_b,
                  &execute_requirements);
        if (packed_status != runtime::CpuPackedGemmStatusV1::success ||
            execute_status != runtime::CpuPackedGemmStatusV1::success) {
          result.legal = false;
          result.reason = "native packed prepacked-B requirement query failed";
        } else {
          result.prepacked_b_bytes = packed_requirements.total_bytes;
          result.workspace_bytes = execute_requirements.total_bytes;
          result.shared_workspace_bytes = execute_requirements.total_bytes;
          result.per_worker_workspace_bytes = 0;
          result.workspace_alignment = static_cast<std::uint32_t>(
              execute_requirements.alignment_bytes);
          selected.candidates[selected_index].required_workspace_bytes =
              execute_requirements.total_bytes;
          selected.candidates[selected_index].required_workspace_alignment =
              static_cast<std::uint32_t>(execute_requirements.alignment_bytes);
          result.timing_scope =
              "prepacked-B execution: B packing prepared before timing; timed "
              "region includes transient A packing, selected packed-ISA "
              "compute, tail handling, and output stores";
        }
      } else if (is_parallel_packed(selected.selected_variant)) {
        result.timing_scope =
            "persistent-context parallel packed execution: timed region "
            "includes shared B packing, per-worker A packing, worker "
            "submission, selected packed-ISA compute, synchronization, tail "
            "handling, and output stores; context creation is outside timing";
      } else if (is_single_packed(selected.selected_variant)) {
        result.timing_scope =
            "packed execution: timed region includes transient A and B "
            "packing, selected packed-ISA compute, tail handling, and output "
            "stores";
      } else if (selected.selected_variant ==
                 planner::CpuGemmVariantV3::external_openblas) {
        result.timing_scope =
            "complete OpenBLAS CBLAS call: provider-internal packing is opaque "
            "and remains inside the timed region";
      } else {
        result.timing_scope =
            "complete implementation call: selected backend has no explicit "
            "benchmark-managed packing stage";
      }
      if (result.legal)
        result.state = std::make_shared<BackendPlanState>(
            selected, packing_mode, compute_layout);
    }
    std::array<char, 8192> diagnostic{};
    planner::format_cpu_gemm_plan_v3(selected, diagnostic.data(),
                                     diagnostic.size());
    result.diagnostic = diagnostic.data();
    if (result.legal && !result.timing_scope.empty()) {
      result.diagnostic += "\nbenchmark_timing_scope=";
      result.diagnostic += result.timing_scope;
    }
    result.diagnostic += "\nbenchmark_smt_policy=";
    result.diagnostic += result.smt_policy;
    result.diagnostic += " benchmark_affinity_policy=";
    result.diagnostic += result.affinity_policy;
    result.diagnostic += " worker_affinity_applied=false available_processors=";
    result.diagnostic += std::to_string(available_processors_);
    if (!result.legal && result.reason.empty())
      result.reason = "planner found no legal variant";
    return result;
  }

  bool prepare(const RunnerPlanV1 &plan, const GemmShapeV1 &shape,
               const float *lhs, const float *rhs,
               std::span<std::byte> workspace,
               std::span<std::byte> prepacked_b_storage, bool prepack_b,
               std::string &error) const override {
    if (workspace.size() < plan.workspace_bytes) {
      error = "runner workspace is smaller than its declared requirement";
      return false;
    }
    if (prepacked_b_storage.size() < plan.prepacked_b_bytes) {
      error = "prepacked-B storage is smaller than its declared requirement";
      return false;
    }
    const auto state =
        std::dynamic_pointer_cast<const BackendPlanState>(plan.state);
    if (!state) {
      error = "selected backend plan state is absent or incompatible";
      return false;
    }
    if (state->packing_mode == PackingModeV1::exclude &&
        state->plan.selected_variant ==
            planner::CpuGemmVariantV3::native_packed_avx2_fma) {
      state->compute_only_lhs = nullptr;
      state->compute_only_rhs = nullptr;
      state->compute_only_workspace = nullptr;
      state->compute_only_ready = false;
      const auto &layout = state->compute_only_layout;
      if (prepack_b || lhs == nullptr || rhs == nullptr ||
          workspace.data() == nullptr || layout.total_bytes == 0 ||
          workspace.size() < layout.total_bytes ||
          !runtime::cpu_packed_avx2_runtime_usable_v1() ||
          reinterpret_cast<std::uintptr_t>(workspace.data()) %
                  runtime::kCpuPackedGemmWorkspaceAlignmentV1 !=
              0) {
        error = "native packed compute-only preparation received invalid storage";
        return false;
      }
      auto *packed_a = reinterpret_cast<float *>(
          workspace.data() + layout.packed_a_offset);
      auto *packed_b = reinterpret_cast<float *>(
          workspace.data() + layout.packed_b_offset);
      pack_compute_only_a(shape, lhs, packed_a);
      pack_compute_only_b(shape, rhs, packed_b);
      state->compute_only_lhs = lhs;
      state->compute_only_rhs = rhs;
      state->compute_only_workspace = workspace.data();
      state->compute_only_ready = true;
      return true;
    }
    if (!prepack_b) return true;
    if (!is_single_packed(state->plan.selected_variant)) {
      error = "selected implementation does not support prepacked-B";
      return false;
    }
    state->packed_b_ready = false;
    const auto packed_status = is_avx512_packed(state->plan.selected_variant)
        ? runtime::cpu_prepare_packed_b_avx512_v1(
              problem(shape, state->plan.problem.minimum_alignment_bytes), rhs,
              prepacked_b_storage.data(), prepacked_b_storage.size(),
              &state->packed_b)
        : runtime::cpu_prepare_packed_b_avx2_v1(
              problem(shape, state->plan.problem.minimum_alignment_bytes), rhs,
              prepacked_b_storage.data(), prepacked_b_storage.size(),
              &state->packed_b);
    if (packed_status != runtime::CpuPackedGemmStatusV1::success) {
      error = std::string(runtime::cpu_packed_gemm_status_message_v1(
          packed_status));
      return false;
    }
    state->packed_b_ready = true;
    return true;
  }

  bool execute(const RunnerPlanV1 &plan, const GemmShapeV1 &shape,
               const float *lhs, const float *rhs, float *output,
               std::span<std::byte> workspace,
               std::span<const std::byte> prepacked_b_storage,
               bool packing_is_prepared,
               std::string &error) const override {
    if (workspace.size() < plan.workspace_bytes ||
        prepacked_b_storage.size() < plan.prepacked_b_bytes) {
      error = "execution storage is smaller than the declared requirement";
      return false;
    }
    const auto state =
        std::dynamic_pointer_cast<const BackendPlanState>(plan.state);
    if (!state || state->plan.problem.m != shape.m ||
        state->plan.problem.n != shape.n || state->plan.problem.k != shape.k ||
        state->plan.selected_id != plan.selected_variant ||
        pointer_alignment(lhs, rhs, output) <
            state->plan.problem.minimum_alignment_bytes) {
      error = "prepared backend plan does not match execution buffers";
      return false;
    }

    switch (state->plan.selected_variant) {
      case planner::CpuGemmVariantV3::reference:
      case planner::CpuGemmVariantV3::tiled:
      case planner::CpuGemmVariantV3::compiler_vectorized: {
        const auto legacy = planner::plan_cpu_gemm_v1(
            state->plan.problem, state->plan.baseline_capabilities,
            legacy_request(state->plan.selected_variant));
        if (!planner::execute_cpu_gemm_plan_v1(legacy, lhs, rhs, output)) {
          error = "selected legacy implementation failed";
          return false;
        }
        return true;
      }
      case planner::CpuGemmVariantV3::external_openblas: {
        std::uint32_t actual_threads = 0;
        const auto provider_status = runtime::execute_openblas_gemm_f32_v1(
            state->plan.problem, lhs, rhs, output,
            plan.actual_threads, &actual_threads);
        if (provider_status != runtime::OpenBlasExecutionStatusV1::success ||
            actual_threads != plan.actual_threads) {
          error = "OpenBLAS failed or did not honor the planned thread count";
          return false;
        }
        return true;
      }
      case planner::CpuGemmVariantV3::native_packed_avx2_fma: {
        if (state->packing_mode == PackingModeV1::exclude) {
          if (!packing_is_prepared || !state->compute_only_ready ||
              state->compute_only_lhs != lhs || state->compute_only_rhs != rhs ||
              state->compute_only_workspace != workspace.data()) {
            error =
                "native packed compute-only execution was not prepared for "
                "these exact buffers";
            return false;
          }
          const auto m = static_cast<std::size_t>(shape.m);
          const auto n = static_cast<std::size_t>(shape.n);
          const auto k = static_cast<std::size_t>(shape.k);
          const auto &layout = state->compute_only_layout;
          const auto *packed_a = reinterpret_cast<const float *>(
              workspace.data() + layout.packed_a_offset);
          const auto *packed_b = reinterpret_cast<const float *>(
              workspace.data() + layout.packed_b_offset);
          for (std::size_t row = 0; row < m;
               row += runtime::kCpuPackedGemmMrV1) {
            const auto rows = static_cast<std::uint32_t>(
                std::min(runtime::kCpuPackedGemmMrV1, m - row));
            const float *a_panel =
                packed_a + (row / runtime::kCpuPackedGemmMrV1) * k *
                               runtime::kCpuPackedGemmMrV1;
            for (std::size_t column = 0; column < n;
                 column += runtime::kCpuPackedGemmNrV1) {
              const auto columns = static_cast<std::uint32_t>(
                  std::min(runtime::kCpuPackedGemmNrV1, n - column));
              const float *b_panel =
                  packed_b + (column / runtime::kCpuPackedGemmNrV1) * k *
                                 runtime::kCpuPackedGemmNrV1;
              runtime::detail::
                  matcore_cpu_packed_avx2_4x16_microkernel_f32_v1(
                      a_panel, b_panel, k, output + row * n + column, n, rows,
                      columns, false);
            }
          }
          return true;
        }
        runtime::CpuPackedGemmStatusV1 packed_status;
        if (state->packing_mode == PackingModeV1::prepack_b) {
          if (!packing_is_prepared || !state->packed_b_ready) {
            error = "prepacked-B execution was not prepared";
            return false;
          }
          packed_status = runtime::cpu_execute_packed_avx2_prepacked_b_v1(
              state->plan.problem, lhs, output, state->packed_b,
              workspace.data(), workspace.size());
        } else {
          packed_status = runtime::cpu_execute_packed_avx2_v1(
              state->plan.problem, lhs, rhs, output, workspace.data(),
              workspace.size());
        }
        if (packed_status != runtime::CpuPackedGemmStatusV1::success) {
          error = std::string(
              runtime::cpu_packed_gemm_status_message_v1(packed_status));
          return false;
        }
        return true;
      }
      case planner::CpuGemmVariantV3::native_packed_avx512_fma: {
        runtime::CpuPackedGemmStatusV1 packed_status;
        if (state->packing_mode == PackingModeV1::prepack_b) {
          if (!packing_is_prepared || !state->packed_b_ready) {
            error = "prepacked-B AVX-512 execution was not prepared";
            return false;
          }
          packed_status = runtime::cpu_execute_packed_avx512_prepacked_b_v1(
              state->plan.problem, lhs, output, state->packed_b,
              workspace.data(), workspace.size());
        } else {
          packed_status = runtime::cpu_execute_packed_avx512_v1(
              state->plan.problem, lhs, rhs, output, workspace.data(),
              workspace.size());
        }
        if (packed_status != runtime::CpuPackedGemmStatusV1::success) {
          error = std::string(
              runtime::cpu_packed_gemm_status_message_v1(packed_status));
          return false;
        }
        return true;
      }
      case planner::CpuGemmVariantV3::native_parallel_avx2_fma:
      case planner::CpuGemmVariantV3::native_parallel_avx512_fma: {
        if (!execution_context_) {
          error = "persistent CPU execution context is unavailable";
          return false;
        }
        runtime::CpuParallelGemmReportV1 execution_report;
        const auto status =
            state->plan.selected_variant ==
                    planner::CpuGemmVariantV3::native_parallel_avx512_fma
                ? runtime::cpu_execute_parallel_packed_avx512_v1(
                      *execution_context_, state->plan.problem, lhs, rhs,
                      output, workspace.data(), workspace.size(),
                      plan.actual_threads,
                      runtime::CpuProviderNestingPolicyV1::native_only,
                      &execution_report)
                : runtime::cpu_execute_parallel_packed_avx2_v1(
                      *execution_context_, state->plan.problem, lhs, rhs,
                      output, workspace.data(), workspace.size(),
                      plan.actual_threads,
                      runtime::CpuProviderNestingPolicyV1::native_only,
                      &execution_report);
        if (status != runtime::CpuParallelGemmStatusV1::success) {
          error = runtime::cpu_parallel_gemm_status_message_v1(status);
          return false;
        }
        if (execution_report.actual_threads != plan.actual_threads ||
            execution_report.workspace_bytes != plan.workspace_bytes ||
            execution_report.shared_packed_b_bytes >
                plan.shared_workspace_bytes ||
            execution_report.per_worker_workspace_bytes >
                plan.per_worker_workspace_bytes) {
          error =
              "parallel execution report disagrees with the benchmark plan";
          return false;
        }
        return true;
      }
    }
    error = "selected backend variant is unknown";
    return false;
  }

  bool synchronize(std::string &) const override { return true; }

 private:
  static std::uint32_t pointer_alignment(const float *lhs, const float *rhs,
                                         const float *output) noexcept {
    std::uint32_t alignment = planner::pointer_alignment_bytes(lhs);
    alignment = std::min(alignment, planner::pointer_alignment_bytes(rhs));
    alignment = std::min(alignment, planner::pointer_alignment_bytes(output));
    return alignment;
  }

  platform::CpuCapabilitiesV2 capabilities_;
  platform::CpuTopologyV1 topology_;
  mutable std::unique_ptr<runtime::CpuExecutionContextV1> execution_context_;
  runtime::CpuExecutionStatusV1 execution_context_status_ =
      runtime::CpuExecutionStatusV1::invalid_configuration;
  std::uint32_t available_processors_ = 1;
};

}  // namespace

std::unique_ptr<GemmRunnerV1> make_planner_runner_v1() {
  return std::make_unique<PlannerRunner>();
}

}  // namespace matcore::mdslc::bench
