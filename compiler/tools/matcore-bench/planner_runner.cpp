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
#include "cpu_runtime_validation.h"
#include "cpu_topology_v1.h"
#include "thread_affinity_v1.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
  auto result = platform::discover_host_cpu_topology_v1();
  // Preserve the diagnostic architecture on unsupported hosts while leaving
  // the record incomplete so placement remains fail closed.
  if (result.architecture == platform::ArchitectureKindV1::unknown)
    result.architecture = architecture;
  return result;
}

std::string format_cpu_ids(const std::vector<std::uint32_t> &cpu_ids) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < cpu_ids.size(); ++index) {
    if (index != 0) output << ',';
    output << cpu_ids[index];
  }
  output << ']';
  return output.str();
}

const platform::CpuLogicalProcessorV1 *find_logical_processor(
    const platform::CpuTopologyV1 &topology,
    std::uint32_t logical_cpu) noexcept {
  const auto found = std::find_if(
      topology.logical_processors.begin(), topology.logical_processors.end(),
      [logical_cpu](const platform::CpuLogicalProcessorV1 &processor) {
        return processor.logical_cpu == logical_cpu;
      });
  return found == topology.logical_processors.end() ? nullptr : &*found;
}

bool same_physical_core(
    const platform::CpuLogicalProcessorV1 &left,
    const platform::CpuLogicalProcessorV1 &right) noexcept {
  return left.package_id != platform::kUnknownTopologyIdV1 &&
         left.core_id != platform::kUnknownTopologyIdV1 &&
         right.package_id != platform::kUnknownTopologyIdV1 &&
         right.core_id != platform::kUnknownTopologyIdV1 &&
         left.package_id == right.package_id && left.core_id == right.core_id;
}

struct BenchmarkCallerSpareV1 {
  bool selected = false;
  bool dedicated_physical_core = false;
  std::uint32_t logical_cpu = platform::kUnknownTopologyIdV1;
  std::uint32_t numa_node = platform::kUnknownTopologyIdV1;
  std::vector<std::uint32_t> reserved_logical_cpus;
};

BenchmarkCallerSpareV1 select_benchmark_caller_spare_v1(
    const platform::CpuTopologyV1 &topology,
    const platform::CpuPlacementPlanV1 &worker_placement) {
  BenchmarkCallerSpareV1 result;
  const std::uint32_t worker_node =
      worker_placement.numa_nodes.size() == 1
          ? worker_placement.numa_nodes.front()
          : platform::kUnknownTopologyIdV1;
  const auto is_worker_cpu = [&](std::uint32_t logical_cpu) {
    return std::find(worker_placement.logical_cpus.begin(),
                     worker_placement.logical_cpus.end(),
                     logical_cpu) != worker_placement.logical_cpus.end();
  };
  const auto worker_uses_core = [&](
                                    const platform::CpuLogicalProcessorV1 &candidate) {
    return std::any_of(
        worker_placement.logical_cpus.begin(),
        worker_placement.logical_cpus.end(),
        [&](std::uint32_t worker_cpu) {
          const auto *worker = find_logical_processor(topology, worker_cpu);
          return worker != nullptr && same_physical_core(candidate, *worker);
        });
  };
  const auto select_highest = [&](bool require_same_node,
                                  bool require_unused_core)
      -> const platform::CpuLogicalProcessorV1 * {
    const platform::CpuLogicalProcessorV1 *selected = nullptr;
    for (const auto &candidate : topology.logical_processors) {
      if (!candidate.online || is_worker_cpu(candidate.logical_cpu) ||
          (require_same_node &&
           (worker_node == platform::kUnknownTopologyIdV1 ||
            candidate.numa_node_id != worker_node)) ||
          (require_unused_core &&
           (candidate.package_id == platform::kUnknownTopologyIdV1 ||
            candidate.core_id == platform::kUnknownTopologyIdV1 ||
            worker_uses_core(candidate))))
        continue;
      if (selected == nullptr || candidate.logical_cpu > selected->logical_cpu)
        selected = &candidate;
    }
    return selected;
  };

  const platform::CpuLogicalProcessorV1 *selected =
      select_highest(true, true);
  if (selected != nullptr) {
    result.dedicated_physical_core = true;
  } else {
    selected = select_highest(true, false);
    if (selected == nullptr) selected = select_highest(false, false);
  }
  if (selected == nullptr) return result;

  result.selected = true;
  result.logical_cpu = selected->logical_cpu;
  result.numa_node = selected->numa_node_id;
  if (result.dedicated_physical_core) {
    for (const auto &processor : topology.logical_processors) {
      if (processor.online && same_physical_core(*selected, processor))
        result.reserved_logical_cpus.push_back(processor.logical_cpu);
    }
  }
  if (result.reserved_logical_cpus.empty())
    result.reserved_logical_cpus.push_back(result.logical_cpu);
  std::sort(result.reserved_logical_cpus.begin(),
            result.reserved_logical_cpus.end());
  return result;
}

platform::CpuAffinityPolicyV1 platform_affinity_policy(
    AffinityPolicyV2 policy) noexcept {
  switch (policy) {
    case AffinityPolicyV2::none:
    case AffinityPolicyV2::compact:
      return platform::CpuAffinityPolicyV1::compact;
    case AffinityPolicyV2::scatter:
      return platform::CpuAffinityPolicyV1::scatter;
    case AffinityPolicyV2::local_first:
      return platform::CpuAffinityPolicyV1::local_first;
  }
  return platform::CpuAffinityPolicyV1::compact;
}

platform::CpuSmtPolicyV1 platform_smt_policy(SmtPolicyV2 policy) noexcept {
  return policy == SmtPolicyV2::allow_smt
             ? platform::CpuSmtPolicyV1::allow_smt
             : platform::CpuSmtPolicyV1::physical_cores_only;
}

planner::CpuPlannerPlacementEvidenceV1 unbound_placement_evidence(
    const platform::CpuTopologyV1 &topology) noexcept {
  planner::CpuPlannerPlacementEvidenceV1 evidence;
  const auto validation = platform::validate_cpu_topology_v1(topology);
  if (!validation || !topology.discovery_complete ||
      topology.numa_nodes.size() != 1 || topology.numa_nodes.front().node_id ==
                                               platform::kUnknownTopologyIdV1)
    return evidence;

  evidence.evidence_complete = true;
  evidence.affinity_requested = false;
  evidence.affinity_applied = false;
  evidence.affinity = platform::CpuAffinityPolicyV1::compact;
  evidence.numa = planner::CpuPlannerNumaPolicyV1::single_node;
  evidence.local_logical_processor_capacity =
      platform::logical_cpu_count_v1(topology);
  evidence.local_physical_core_capacity =
      platform::physical_core_count_v1(topology);
  evidence.selected_numa_node_count = 1;
  evidence.selected_numa_nodes[0] = topology.numa_nodes.front().node_id;
  evidence.crosses_numa_nodes = false;
  evidence.caller_first_touch_required = false;
  if (evidence.local_logical_processor_capacity == 0 ||
      evidence.local_physical_core_capacity == 0)
    evidence.evidence_complete = false;
  return evidence;
}

planner::CpuPlannerPlacementEvidenceV1 bound_placement_evidence(
    const platform::CpuTopologyV1 &topology,
    const platform::CpuPlacementPlanV1 &placement) noexcept {
  planner::CpuPlannerPlacementEvidenceV1 evidence;
  if (placement.status != platform::CpuPlacementStatusV1::selected ||
      placement.logical_cpus.empty() || placement.numa_nodes.empty() ||
      placement.crosses_numa_nodes || placement.numa_nodes.size() != 1)
    return evidence;

  const auto selected =
      platform::restrict_cpu_topology_v1(topology, placement.logical_cpus);
  if (!selected) return evidence;
  evidence.evidence_complete = true;
  evidence.affinity_requested = true;
  evidence.affinity_applied = true;
  evidence.affinity = placement.affinity;
  evidence.numa = placement.affinity == platform::CpuAffinityPolicyV1::local_first
                      ? planner::CpuPlannerNumaPolicyV1::local_first
                      : planner::CpuPlannerNumaPolicyV1::single_node;
  evidence.local_logical_processor_capacity =
      platform::logical_cpu_count_v1(selected.topology);
  evidence.local_physical_core_capacity =
      platform::physical_core_count_v1(selected.topology);
  evidence.selected_numa_node_count = 1;
  evidence.selected_numa_nodes[0] = placement.numa_nodes.front();
  evidence.crosses_numa_nodes = false;
  // Worker CPU affinity does not establish any NUMA page policy. The
  // benchmark deliberately makes no first-touch or memory-binding claim.
  evidence.caller_first_touch_required = false;
  if (evidence.local_logical_processor_capacity == 0 ||
      evidence.local_physical_core_capacity == 0)
    evidence.evidence_complete = false;
  return evidence;
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
                   ComputeOnlyPackedLayout compute_layout,
                   std::shared_ptr<runtime::CpuExecutionContextV1> context)
      : plan(std::move(value)),
        packing_mode(packing),
        compute_only_layout(compute_layout),
        execution_context(std::move(context)) {}
  planner::CpuGemmPlanV3 plan;
  PackingModeV1 packing_mode = PackingModeV1::include;
  ComputeOnlyPackedLayout compute_only_layout;
  std::shared_ptr<runtime::CpuExecutionContextV1> execution_context;
  mutable runtime::CpuPackedBViewV1 packed_b;
  mutable bool packed_b_ready = false;
  mutable const float *compute_only_lhs = nullptr;
  mutable const float *compute_only_rhs = nullptr;
  mutable const std::byte *compute_only_workspace = nullptr;
  mutable bool compute_only_ready = false;
};

struct ExecutionContextRecord {
  AffinityPolicyV2 affinity_policy = AffinityPolicyV2::none;
  SmtPolicyV2 smt_policy = SmtPolicyV2::physical_cores_only;
  bool affinity_user_requested = false;
  bool affinity_policy_induced = false;
  std::vector<std::uint32_t> worker_cpu_ids;
  std::shared_ptr<runtime::CpuExecutionContextV1> context;
  runtime::CpuExecutionStatusV1 creation_status =
      runtime::CpuExecutionStatusV1::invalid_configuration;
  runtime::CpuWorkerAffinityReportV1 affinity_report;
  runtime::CpuRuntimeValidationEvidenceV1 validation_evidence;
  planner::CpuPlannerPlacementEvidenceV1 placement;
  std::string diagnostic;
};

class PlannerRunner final : public GemmRunnerV1 {
 public:
  PlannerRunner()
      : capabilities_(platform::discover_cpu_capabilities_v2(
            benchmark_implementation_evidence())),
        topology_(discover_benchmark_topology(capabilities_.architecture)) {
    process_affinity_ = platform::discover_current_thread_affinity_v1();
    if (process_affinity_.discovery_complete) {
      auto restricted = platform::restrict_cpu_topology_v1(
          topology_, process_affinity_.allowed_logical_cpus);
      if (restricted) {
        topology_ = std::move(restricted.topology);
        topology_restriction_diagnostic_ =
            "topology restricted to inherited process scheduler mask " +
            format_cpu_ids(process_affinity_.allowed_logical_cpus);
      } else {
        topology_.discovery_complete = false;
        topology_restriction_diagnostic_ =
            "process scheduler mask could not be projected onto topology: " +
            restricted.reason;
      }
    } else {
      topology_.discovery_complete = false;
      topology_restriction_diagnostic_ =
          "process scheduler-affinity discovery is incomplete; parallel "
          "planning fails closed";
    }

    available_processors_ = platform::logical_cpu_count_v1(topology_);
    auto record = std::make_shared<ExecutionContextRecord>();
    record->placement = unbound_placement_evidence(topology_);
    record->diagnostic =
        "workers inherit the process mask and are not individually pinned; "
        "NUMA page placement is not requested or claimed";
    default_context_ = record;
    context_records_.push_back(record);
    caller_placement_diagnostic_ =
        "benchmark caller scheduler affinity not requested: no measured "
        "bound-worker placement has been established";
  }

  RunnerEnvironmentV1 environment() const override {
    std::string probe_context_error;
    const auto probe_context = context_for(
        AffinityPolicyV2::none, SmtPolicyV2::physical_cores_only, 1,
        probe_context_error, false);
    const auto probe_problem = problem({1, 1, 1}, alignof(float));
    const auto baseline =
        runtime::discover_cpu_gemm_implementation_resources_v1(probe_problem,
            1, runtime::CpuExternalProviderProbeV1::include);
    const auto resources = runtime::augment_cpu_gemm_implementation_resources_v2(
        probe_problem, baseline,
        probe_context != nullptr ? probe_context->context.get() : nullptr,
        probe_context != nullptr
            ? probe_context->validation_evidence
            : runtime::CpuRuntimeValidationEvidenceV1{});
    planner::CpuThreadPolicyV1 thread_policy;
    thread_policy.worker_affinity_active =
        probe_context != nullptr && probe_context->placement.affinity_applied;
    const auto probe = planner::plan_cpu_gemm_v3(
        probe_problem, capabilities_, topology_, thread_policy, resources,
        planner::CpuGemmRequestV3::automatic, 0,
        probe_context != nullptr ? probe_context->placement
                                 : planner::CpuPlannerPlacementEvidenceV1{});
    std::array<char, 8192> diagnostic{};
    planner::format_cpu_gemm_plan_v3(probe, diagnostic.data(),
                                     diagnostic.size());
    const auto provider = runtime::openblas_provider_info_v1();
    RunnerEnvironmentV1 result;
    result.capability_record = platform::format_cpu_capabilities_v2(capabilities_);
    result.capability_record += "\nplanner_probe=";
    result.capability_record += diagnostic.data();
    result.capability_runtime_validation_source =
        "benchmark-process numerical self-test: exact persistent-context "
        "reference, tiled, "
        "compiler-vectorized, single-thread OpenBLAS, packed AVX2, packed "
        "AVX-512, two-worker parallel AVX2, and two-worker parallel AVX-512 "
        "are authenticated independently on the same bound or unbound worker "
        "context later used for execution; unavailable implementations remain "
        "false; every emitted measurement is independently checked by a "
        "double-precision oracle";
    result.topology_record = platform::format_cpu_topology_v1(topology_);
    result.topology_record += "\nprocess_affinity=";
    result.topology_record += topology_restriction_diagnostic_;
    result.capability_record_version = capabilities_.version;
    result.topology_record_version = topology_.version;
    const auto topology_validation = platform::validate_cpu_topology_v1(topology_);
    result.topology_discovery_complete =
        topology_validation.valid && topology_.discovery_complete;
    result.logical_processors = platform::logical_cpu_count_v1(topology_);
    result.physical_cores = platform::physical_core_count_v1(topology_);
    result.numa_nodes = platform::numa_node_count_v1(topology_);
    {
      std::lock_guard lock(context_mutex_);
      std::uint32_t pinned_contexts = 0;
      std::uint32_t user_requested_contexts = 0;
      std::uint32_t policy_induced_contexts = 0;
      for (const auto &record : context_records_) {
        if (record == nullptr || record->context == nullptr) continue;
        const auto info = record->context->info();
        result.persistent_execution_context =
            result.persistent_execution_context || info.accepting_work;
        result.execution_context_workers = std::max(
            result.execution_context_workers, info.actual_worker_count);
        result.execution_context_workers_started += info.workers_started;
        result.execution_context_submissions += info.completed_submissions;
        if (!record->worker_cpu_ids.empty() &&
            info.completed_submissions != 0 &&
            info.affinity.complete &&
            info.affinity.status == runtime::CpuWorkerAffinityStatusV1::complete)
          ++pinned_contexts;
        if (record->affinity_user_requested &&
            info.completed_submissions != 0 && info.affinity.complete)
          ++user_requested_contexts;
        if (record->affinity_policy_induced &&
            info.completed_submissions != 0 && info.affinity.complete)
          ++policy_induced_contexts;
      }
      result.worker_affinity_applied = pinned_contexts != 0;
      result.worker_affinity_user_requested = user_requested_contexts != 0;
      result.worker_affinity_policy_induced = policy_induced_contexts != 0;
      result.worker_affinity_source =
          "aggregate over persistent benchmark contexts: " +
          std::to_string(pinned_contexts) +
          " context(s) have complete per-worker scheduler affinity; " +
          std::to_string(user_requested_contexts) +
          " user-requested and " + std::to_string(policy_induced_contexts) +
          " induced by physical-cores-only SMT policy; " +
          "NUMA page placement is never claimed; " +
          topology_restriction_diagnostic_;
      result.worker_affinity_source += "; ";
      result.worker_affinity_source += caller_placement_diagnostic_;
      result.topology_record += "\nbenchmark_caller_affinity=";
      result.topology_record += caller_placement_diagnostic_;
    }
    result.available_processors = available_processors_;
    if (probe_context == nullptr || probe_context->context == nullptr) {
      result.capability_record += "\nexecution_context_status=";
      result.capability_record += probe_context_error.empty()
                                      ? "required benchmark context unavailable"
                                      : probe_context_error;
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
    std::string context_error;
    const auto context_record = context_for(
        affinity_policy, smt_policy, requested_threads, context_error);
    const bool worker_binding_required =
        affinity_policy != AffinityPolicyV2::none ||
        smt_policy == SmtPolicyV2::physical_cores_only;
    if (worker_binding_required &&
        (context_record == nullptr || context_record->context == nullptr ||
         !context_record->placement.evidence_complete)) {
      result.reason = context_error.empty()
                          ? "required worker-affinity context is unavailable"
                          : context_error;
      result.affinity_diagnostic = result.reason;
      return result;
    }
    if (context_record != nullptr) {
      result.worker_affinity_applied =
          !context_record->worker_cpu_ids.empty() &&
          context_record->affinity_report.complete &&
          context_record->affinity_report.status ==
              runtime::CpuWorkerAffinityStatusV1::complete;
      result.worker_affinity_user_requested =
          context_record->affinity_user_requested;
      result.worker_affinity_policy_induced =
          context_record->affinity_policy_induced;
      result.affinity_diagnostic = context_record->diagnostic;
      result.affinity_diagnostic += "; ";
      result.affinity_diagnostic += caller_placement_diagnostic();
    }
    const auto gemm_problem = problem(shape, minimum_alignment);
    const auto baseline =
        runtime::discover_cpu_gemm_implementation_resources_v1(
            gemm_problem, requested_threads,
            runtime::CpuExternalProviderProbeV1::include);
    const auto resources = runtime::augment_cpu_gemm_implementation_resources_v2(
        gemm_problem, baseline,
        context_record != nullptr ? context_record->context.get() : nullptr,
        context_record != nullptr
            ? context_record->validation_evidence
            : runtime::CpuRuntimeValidationEvidenceV1{});
    planner::CpuThreadPolicyV1 thread_policy;
    thread_policy.requested_threads = requested_threads;
    thread_policy.allow_smt = smt_policy == SmtPolicyV2::allow_smt;
    thread_policy.worker_affinity_active =
        context_record != nullptr && context_record->placement.affinity_applied;
    if (context_record != nullptr && context_record->context != nullptr)
      thread_policy.maximum_threads =
          context_record->context->info().actual_worker_count;
    auto selected = planner::plan_cpu_gemm_v3(
        gemm_problem, capabilities_, topology_, thread_policy, resources,
        request, 0,
        context_record != nullptr
            ? context_record->placement
            : planner::CpuPlannerPlacementEvidenceV1{});
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
    if (result.legal && result.worker_affinity_applied &&
        selected.selected_variant ==
            planner::CpuGemmVariantV3::external_openblas &&
        selected.candidates[static_cast<std::size_t>(
            selected.selected_variant)].actual_threads != 1) {
      result.legal = false;
      result.worker_affinity_applied = false;
      result.reason =
          "multi-thread OpenBLAS cannot honor native worker affinity because "
          "provider-thread affinity is not authenticated; request one provider "
          "thread or use a native parallel variant";
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
    ComputeOnlyPackedLayout compute_layout;
    if (result.legal) {
      const std::size_t selected_index =
          static_cast<std::size_t>(selected.selected_variant);
      auto &candidate = selected.candidates[selected_index];
      result.actual_threads = candidate.actual_threads;
      result.parallel_row_tasks = candidate.parallel_row_tasks;
      result.parallel_column_tasks = candidate.parallel_column_tasks;
      result.parallel_task_count = candidate.parallel_task_count;
      result.workspace_bytes = candidate.required_workspace_bytes;
      result.shared_workspace_bytes = candidate.shared_workspace_bytes;
      result.per_worker_workspace_bytes = candidate.per_worker_workspace_bytes;
      result.workspace_alignment = candidate.required_workspace_alignment;
      result.packing_required = is_single_packed(selected.selected_variant) ||
                                is_parallel_packed(selected.selected_variant);
      result.supports_prepacked_b = is_single_packed(selected.selected_variant);
      result.persistent_execution_context =
          is_parallel_packed(selected.selected_variant) ||
          result.worker_affinity_applied;
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
      if (result.legal && result.worker_affinity_applied &&
          !is_parallel_packed(selected.selected_variant)) {
        result.timing_scope +=
            "; timed region includes submission to pinned persistent worker 0 "
            "and synchronization";
      }
      if (result.legal)
        result.state = std::make_shared<BackendPlanState>(
            selected, packing_mode, compute_layout,
            context_record != nullptr ? context_record->context : nullptr);
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
    result.diagnostic += " worker_affinity_applied=";
    result.diagnostic += result.worker_affinity_applied ? "true" : "false";
    result.diagnostic += " worker_affinity_user_requested=";
    result.diagnostic +=
        result.worker_affinity_user_requested ? "true" : "false";
    result.diagnostic += " worker_affinity_policy_induced=";
    result.diagnostic +=
        result.worker_affinity_policy_induced ? "true" : "false";
    result.diagnostic += " available_processors=";
    result.diagnostic += std::to_string(available_processors_);
    result.diagnostic += " affinity_detail=";
    result.diagnostic += result.affinity_diagnostic;
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
    const auto state =
        std::dynamic_pointer_cast<const BackendPlanState>(plan.state);
    if (!plan.worker_affinity_applied || state == nullptr ||
        is_parallel_packed(state->plan.selected_variant)) {
      return execute_direct(plan, shape, lhs, rhs, output, workspace,
                            prepacked_b_storage, packing_is_prepared, error);
    }
    if (state->execution_context == nullptr) {
      error = "affinity-aware serial execution requires a persistent context";
      return false;
    }
    if (state->plan.selected_variant ==
            planner::CpuGemmVariantV3::external_openblas &&
        plan.actual_threads != 1) {
      error =
          "multi-thread OpenBLAS provider affinity is not authenticated";
      return false;
    }

    SerialDispatchPayload payload{this,
                                  &plan,
                                  shape,
                                  lhs,
                                  rhs,
                                  output,
                                  workspace,
                                  prepacked_b_storage,
                                  packing_is_prepared,
                                  &error};
    const auto nesting =
        state->plan.selected_variant ==
                planner::CpuGemmVariantV3::external_openblas
            ? runtime::CpuProviderNestingPolicyV1::external_provider_active
            : runtime::CpuProviderNestingPolicyV1::native_only;
    const auto status = state->execution_context->run_tasks(
        1, 1, nesting, &PlannerRunner::serial_dispatch_task, &payload);
    if (status != runtime::CpuExecutionStatusV1::success) {
      if (error.empty())
        error = runtime::cpu_execution_status_message_v1(status);
      return false;
    }
    return payload.succeeded;
  }

  bool execute_direct(const RunnerPlanV1 &plan, const GemmShapeV1 &shape,
                      const float *lhs, const float *rhs, float *output,
                      std::span<std::byte> workspace,
                      std::span<const std::byte> prepacked_b_storage,
                      bool packing_is_prepared,
                      std::string &error) const {
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
        if (!state->execution_context) {
          error = "persistent CPU execution context is unavailable";
          return false;
        }
        runtime::CpuParallelGemmReportV1 execution_report;
        const auto status =
            state->plan.selected_variant ==
                    planner::CpuGemmVariantV3::native_parallel_avx512_fma
                ? runtime::cpu_execute_parallel_packed_avx512_v1(
                      *state->execution_context, state->plan.problem, lhs, rhs,
                      output, workspace.data(), workspace.size(),
                      plan.actual_threads,
                      runtime::CpuProviderNestingPolicyV1::native_only,
                      &execution_report)
                : runtime::cpu_execute_parallel_packed_avx2_v1(
                      *state->execution_context, state->plan.problem, lhs, rhs,
                      output, workspace.data(), workspace.size(),
                      plan.actual_threads,
                      runtime::CpuProviderNestingPolicyV1::native_only,
                      &execution_report);
        if (status != runtime::CpuParallelGemmStatusV1::success) {
          error = runtime::cpu_parallel_gemm_status_message_v1(status);
          return false;
        }
        if (execution_report.actual_threads != plan.actual_threads ||
            execution_report.row_task_count != plan.parallel_row_tasks ||
            execution_report.column_task_count !=
                plan.parallel_column_tasks ||
            execution_report.task_count != plan.parallel_task_count ||
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
  struct SerialDispatchPayload {
    const PlannerRunner *runner = nullptr;
    const RunnerPlanV1 *plan = nullptr;
    GemmShapeV1 shape;
    const float *lhs = nullptr;
    const float *rhs = nullptr;
    float *output = nullptr;
    std::span<std::byte> workspace;
    std::span<const std::byte> prepacked_b_storage;
    bool packing_is_prepared = false;
    std::string *error = nullptr;
    bool succeeded = false;
  };

  static runtime::CpuExecutionStatusV1 serial_dispatch_task(
      std::size_t task_index, std::size_t worker_index,
      void *user_data) noexcept {
    auto *payload = static_cast<SerialDispatchPayload *>(user_data);
    if (payload == nullptr || payload->runner == nullptr ||
        payload->plan == nullptr || payload->error == nullptr ||
        task_index != 0 || worker_index != 0)
      return runtime::CpuExecutionStatusV1::invalid_configuration;
    try {
      payload->succeeded = payload->runner->execute_direct(
          *payload->plan, payload->shape, payload->lhs, payload->rhs,
          payload->output, payload->workspace, payload->prepacked_b_storage,
          payload->packing_is_prepared, *payload->error);
      return payload->succeeded
                 ? runtime::CpuExecutionStatusV1::success
                 : runtime::CpuExecutionStatusV1::callback_failed;
    } catch (...) {
      try {
        *payload->error =
            "affinity-aware serial execution raised an internal exception";
      } catch (...) {
      }
      return runtime::CpuExecutionStatusV1::resource_exhausted;
    }
  }

  std::shared_ptr<ExecutionContextRecord> context_for(
      AffinityPolicyV2 affinity_policy, SmtPolicyV2 smt_policy,
      std::uint32_t requested_threads, std::string &error,
      bool isolate_benchmark_caller = true) const {
    const bool policy_induced_affinity =
        affinity_policy == AffinityPolicyV2::none &&
        smt_policy == SmtPolicyV2::physical_cores_only;
    if (affinity_policy == AffinityPolicyV2::none &&
        !policy_induced_affinity) {
      std::lock_guard lock(context_mutex_);
      if (default_context_ != nullptr && default_context_->context == nullptr) {
        runtime::CpuExecutionContextConfigV1 config;
        config.requested_threads = available_processors_;
        config.maximum_threads = available_processors_;
        auto context = runtime::CpuExecutionContextV1::create(
            config, &default_context_->creation_status,
            &default_context_->affinity_report);
        if (context != nullptr) {
          default_context_->context =
              std::shared_ptr<runtime::CpuExecutionContextV1>(
                  std::move(context));
          default_context_->validation_evidence =
              runtime::validate_cpu_runtime_variants_v1(
                  *default_context_->context);
        }
      }
      if (default_context_ == nullptr || default_context_->context == nullptr) {
        error = "persistent unbound execution context is unavailable: ";
        error += runtime::cpu_execution_status_message_v1(
            default_context_ != nullptr
                ? default_context_->creation_status
                : runtime::CpuExecutionStatusV1::invalid_configuration);
      }
      return default_context_;
    }

    const auto validation = platform::validate_cpu_topology_v1(topology_);
    if (!validation || !topology_.discovery_complete ||
        topology_.numa_nodes.empty()) {
      error = "explicit worker affinity requires a complete process-restricted "
              "CPU topology";
      return nullptr;
    }

    platform::CpuPlacementRequestV1 request;
    request.requested_workers =
        policy_induced_affinity
            ? std::min(requested_threads,
                       platform::physical_core_count_v1(topology_))
            : requested_threads;
    request.affinity = policy_induced_affinity
                           ? platform::CpuAffinityPolicyV1::compact
                           : platform_affinity_policy(affinity_policy);
    request.smt = platform_smt_policy(smt_policy);
    request.allow_cross_numa = false;
    if (affinity_policy == AffinityPolicyV2::local_first)
      request.preferred_numa_node = topology_.numa_nodes.front().node_id;
    std::lock_guard lock(context_mutex_);
    if (isolate_benchmark_caller && caller_affinity_failed_) {
      error = caller_placement_diagnostic_;
      return nullptr;
    }

    platform::CpuTopologyV1 worker_topology = topology_;
    const auto restrict_workers_away_from_caller = [&]() -> bool {
      if (!caller_affinity_applied_) return true;
      std::vector<std::uint32_t> allowed;
      allowed.reserve(topology_.logical_processors.size());
      for (const auto &processor : topology_.logical_processors) {
        if (processor.online &&
            std::find(caller_reserved_cpu_ids_.begin(),
                      caller_reserved_cpu_ids_.end(),
                      processor.logical_cpu) == caller_reserved_cpu_ids_.end())
          allowed.push_back(processor.logical_cpu);
      }
      const auto restricted =
          platform::restrict_cpu_topology_v1(topology_, allowed);
      if (!restricted) {
        error = "worker topology cannot exclude reserved benchmark caller "
                "CPUs: ";
        error += restricted.reason;
        return false;
      }
      worker_topology = restricted.topology;
      return true;
    };
    if (!restrict_workers_away_from_caller()) return nullptr;

    auto placement =
        platform::plan_cpu_placement_v1(worker_topology, request);
    const auto placement_is_usable = [&]() {
      if (placement.status != platform::CpuPlacementStatusV1::selected) {
        error = caller_affinity_applied_
                    ? "worker CPU placement rejected after reserving the "
                      "benchmark caller: "
                    : "worker CPU placement rejected: ";
        error += placement.reason;
        return false;
      }
      if (placement.crosses_numa_nodes || placement.numa_nodes.size() != 1) {
        error =
            "benchmark worker placement refuses cross-NUMA execution because "
            "no NUMA page-placement backend is active";
        return false;
      }
      return true;
    };
    if (!placement_is_usable()) return nullptr;

    if (isolate_benchmark_caller) {
      if (caller_affinity_applied_) {
        const auto application = platform::apply_current_thread_affinity_v1(
            caller_cpu_id_);
        if (application.status != platform::ThreadAffinityStatusV1::applied) {
          caller_affinity_failed_ = true;
          caller_placement_diagnostic_ =
              "benchmark caller scheduler-affinity reauthentication failed: "
              "caller_cpu_id=" +
              std::to_string(caller_cpu_id_) + " status=" +
              std::string(platform::to_string(application.status)) +
              " platform_error=" +
              std::to_string(application.platform_error) +
              "; bound benchmark planning fails closed";
          error = caller_placement_diagnostic_;
          return nullptr;
        }
      } else {
        const auto spare =
            select_benchmark_caller_spare_v1(topology_, placement);
        if (!spare.selected) {
          caller_placement_diagnostic_ =
              "benchmark caller isolation unavailable: no spare logical CPU "
              "remains outside worker_cpu_ids=" +
              format_cpu_ids(placement.logical_cpus) +
              "; caller_scheduler_affinity_applied=false";
        } else {
          const auto application =
              platform::apply_current_thread_affinity_v1(spare.logical_cpu);
          if (application.status != platform::ThreadAffinityStatusV1::applied) {
            caller_affinity_failed_ = true;
            caller_placement_diagnostic_ =
                "benchmark caller scheduler-affinity application failed: "
                "selected caller_cpu_id=" +
                std::to_string(spare.logical_cpu) + " status=" +
                std::string(platform::to_string(application.status)) +
                " platform_error=" +
                std::to_string(application.platform_error) +
                "; bound benchmark planning fails closed";
            error = caller_placement_diagnostic_;
            return nullptr;
          }
          caller_affinity_applied_ = true;
          caller_cpu_id_ = spare.logical_cpu;
          caller_reserved_cpu_ids_ = spare.reserved_logical_cpus;
          caller_placement_diagnostic_ =
              "benchmark caller scheduler affinity applied: caller_cpu_id=" +
              std::to_string(spare.logical_cpu) +
              " dedicated_physical_core=" +
              (spare.dedicated_physical_core ? std::string("true")
                                             : std::string("false")) +
              " reserved_cpu_ids=" +
              format_cpu_ids(spare.reserved_logical_cpus) +
              " selected_numa_node=" +
              (spare.numa_node == platform::kUnknownTopologyIdV1
                   ? std::string("unknown")
                   : std::to_string(spare.numa_node)) +
              "; caller is excluded from worker placement; "
              "numa_memory_placement=false";
          if (!restrict_workers_away_from_caller()) return nullptr;
          placement =
              platform::plan_cpu_placement_v1(worker_topology, request);
          if (!placement_is_usable()) return nullptr;
        }
      }
    }

    for (const auto &record : context_records_) {
      if (record != nullptr && record->affinity_policy == affinity_policy &&
          record->smt_policy == smt_policy &&
          record->affinity_policy_induced == policy_induced_affinity &&
          record->worker_cpu_ids == placement.logical_cpus)
        return record;
    }

    auto record = std::make_shared<ExecutionContextRecord>();
    record->affinity_policy = affinity_policy;
    record->smt_policy = smt_policy;
    record->affinity_user_requested =
        affinity_policy != AffinityPolicyV2::none;
    record->affinity_policy_induced = policy_induced_affinity;
    record->worker_cpu_ids = placement.logical_cpus;
    runtime::CpuExecutionContextConfigV1 config;
    config.requested_threads = placement.actual_workers;
    config.maximum_threads = placement.actual_workers;
    config.worker_cpu_ids = placement.logical_cpus;
    auto context = runtime::CpuExecutionContextV1::create(
        config, &record->creation_status, &record->affinity_report);
    if (context == nullptr || !record->affinity_report.complete ||
        record->affinity_report.status !=
            runtime::CpuWorkerAffinityStatusV1::complete ||
        record->affinity_report.applied_workers != placement.actual_workers ||
        record->affinity_report.numa_memory_placement_applied) {
      error = "strict worker-affinity context creation failed: status=";
      error += runtime::cpu_execution_status_message_v1(record->creation_status);
      error += " affinity=";
      error += runtime::cpu_worker_affinity_status_message_v1(
          record->affinity_report.status);
      return nullptr;
    }
    record->context =
        std::shared_ptr<runtime::CpuExecutionContextV1>(std::move(context));
    record->placement = bound_placement_evidence(topology_, placement);
    if (!record->placement.evidence_complete) {
      error = "strict worker affinity applied, but complete single-node "
              "placement evidence could not be constructed";
      return nullptr;
    }
    record->validation_evidence =
        runtime::validate_cpu_runtime_variants_v1(*record->context);
    record->diagnostic = policy_induced_affinity
                             ? "physical-cores-only SMT policy induced strict "
                               "one-logical-CPU-per-core scheduler affinity; "
                               "user_requested=false"
                             : "strict per-worker scheduler affinity complete; "
                               "user_requested=true";
    record->diagnostic +=
        " policy=" + std::string(platform::to_string(placement.affinity)) +
        " cpu_ids=" + format_cpu_ids(record->worker_cpu_ids) +
        " selected_numa_node=" +
        std::to_string(record->placement.selected_numa_nodes[0]) +
        " numa_memory_placement=false";
    context_records_.push_back(record);
    return record;
  }

  static std::uint32_t pointer_alignment(const float *lhs, const float *rhs,
                                         const float *output) noexcept {
    std::uint32_t alignment = planner::pointer_alignment_bytes(lhs);
    alignment = std::min(alignment, planner::pointer_alignment_bytes(rhs));
    alignment = std::min(alignment, planner::pointer_alignment_bytes(output));
    return alignment;
  }

  std::string caller_placement_diagnostic() const {
    std::lock_guard lock(context_mutex_);
    return caller_placement_diagnostic_;
  }

  platform::CpuCapabilitiesV2 capabilities_;
  platform::CpuTopologyV1 topology_;
  platform::ThreadAffinityInventoryV1 process_affinity_;
  std::string topology_restriction_diagnostic_;
  std::shared_ptr<ExecutionContextRecord> default_context_;
  mutable std::mutex context_mutex_;
  mutable std::vector<std::shared_ptr<ExecutionContextRecord>> context_records_;
  mutable bool caller_affinity_applied_ = false;
  mutable bool caller_affinity_failed_ = false;
  mutable std::uint32_t caller_cpu_id_ = platform::kUnknownTopologyIdV1;
  mutable std::vector<std::uint32_t> caller_reserved_cpu_ids_;
  mutable std::string caller_placement_diagnostic_;
  std::uint32_t available_processors_ = 0;
};

}  // namespace

std::unique_ptr<GemmRunnerV1> make_planner_runner_v1() {
  return std::make_unique<PlannerRunner>();
}

}  // namespace matcore::mdslc::bench
