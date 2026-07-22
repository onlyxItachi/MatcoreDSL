#include "cpu_planner_v3.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace matcore::mdslc::planner {
namespace {

inline constexpr std::uint64_t kParallelMacroTileRowsV1 = 128;

constexpr std::uint64_t feature(platform::CpuFeatureV2 value) noexcept {
  return platform::feature_bit(value);
}

void populate_requirement_metadata(CpuCandidateDecisionV3 *decision) noexcept {
  if (decision == nullptr) return;
  const std::uint64_t portable =
      feature(platform::CpuFeatureV2::portable_scalar_f32);
  const std::uint64_t avx2_fma =
      feature(platform::CpuFeatureV2::avx2) |
      feature(platform::CpuFeatureV2::fma);
  const std::uint64_t avx512_fma =
      feature(platform::CpuFeatureV2::avx512f) |
      feature(platform::CpuFeatureV2::fma);
  std::uint64_t required = portable;
  switch (decision->variant) {
    case CpuGemmVariantV3::reference:
    case CpuGemmVariantV3::tiled:
    case CpuGemmVariantV3::external_openblas:
      break;
    case CpuGemmVariantV3::compiler_vectorized:
    case CpuGemmVariantV3::native_packed_avx2_fma:
    case CpuGemmVariantV3::native_parallel_avx2_fma:
      required |= avx2_fma;
      break;
    case CpuGemmVariantV3::native_packed_avx512_fma:
    case CpuGemmVariantV3::native_parallel_avx512_fma:
      required |= avx512_fma;
      break;
  }
  decision->required_hardware_features = required;
  decision->required_os_features = required;
  decision->required_compiler_features = required;
  decision->required_implementation_features = required;
}

bool request_matches(CpuGemmRequestV3 request,
                     CpuGemmVariantV3 variant) noexcept {
  if (request == CpuGemmRequestV3::automatic) return true;
  return static_cast<unsigned>(request) ==
         static_cast<unsigned>(variant) + 1U;
}

std::string_view request_name(CpuGemmRequestV3 request) noexcept {
  switch (request) {
    case CpuGemmRequestV3::automatic:
      return "automatic";
    case CpuGemmRequestV3::force_reference:
      return "force-reference";
    case CpuGemmRequestV3::force_tiled:
      return "force-tiled";
    case CpuGemmRequestV3::force_compiler_vectorized:
      return "force-compiler-vectorized";
    case CpuGemmRequestV3::force_external_openblas:
      return "force-external-openblas";
    case CpuGemmRequestV3::force_native_packed_avx2_fma:
      return "force-native-packed-avx2-fma";
    case CpuGemmRequestV3::force_native_packed_avx512_fma:
      return "force-native-packed-avx512-fma";
    case CpuGemmRequestV3::force_native_parallel_avx2_fma:
      return "force-native-parallel-avx2-fma";
    case CpuGemmRequestV3::force_native_parallel_avx512_fma:
      return "force-native-parallel-avx512-fma";
  }
  return "invalid";
}

std::string_view status_name(CpuPlanStatusV1 status) noexcept {
  switch (status) {
    case CpuPlanStatusV1::selected:
      return "selected";
    case CpuPlanStatusV1::no_legal_variant:
      return "no-legal-variant";
    case CpuPlanStatusV1::forced_variant_illegal:
      return "forced-variant-illegal";
    case CpuPlanStatusV1::invalid_problem:
      return "invalid-problem";
    case CpuPlanStatusV1::invalid_capabilities:
      return "invalid-capabilities";
  }
  return "invalid";
}

CpuGemmRequestV2 v2_request(CpuGemmVariantV3 variant) noexcept {
  switch (variant) {
    case CpuGemmVariantV3::reference:
      return CpuGemmRequestV2::force_reference;
    case CpuGemmVariantV3::tiled:
      return CpuGemmRequestV2::force_tiled;
    case CpuGemmVariantV3::compiler_vectorized:
      return CpuGemmRequestV2::force_compiler_vectorized;
    case CpuGemmVariantV3::external_openblas:
      return CpuGemmRequestV2::force_external_openblas;
    case CpuGemmVariantV3::native_packed_avx2_fma:
    case CpuGemmVariantV3::native_packed_avx512_fma:
    case CpuGemmVariantV3::native_parallel_avx2_fma:
    case CpuGemmVariantV3::native_parallel_avx512_fma:
      return CpuGemmRequestV2::force_native_packed_avx2_fma;
  }
  return CpuGemmRequestV2::force_reference;
}

std::size_t v2_index(CpuGemmVariantV3 variant) noexcept {
  switch (variant) {
    case CpuGemmVariantV3::reference:
      return 0;
    case CpuGemmVariantV3::tiled:
      return 1;
    case CpuGemmVariantV3::compiler_vectorized:
      return 2;
    case CpuGemmVariantV3::external_openblas:
      return 3;
    case CpuGemmVariantV3::native_packed_avx2_fma:
    case CpuGemmVariantV3::native_packed_avx512_fma:
    case CpuGemmVariantV3::native_parallel_avx2_fma:
    case CpuGemmVariantV3::native_parallel_avx512_fma:
      return 4;
  }
  return 0;
}

std::string_view topology_reason(
    const CpuPlannerTopologyViewV1 &topology) noexcept {
  if (topology.version != kCpuPlannerTopologyViewVersionV1)
    return "CPU planner topology view version is unsupported";
  if (!topology.discovery_complete)
    return "complete CPU topology discovery is required for parallel planning";
  if (topology.logical_processors == 0 || topology.physical_cores == 0 ||
      topology.available_processors == 0 || topology.numa_nodes == 0)
    return "complete CPU topology counts must be positive";
  if (!topology.numa_node_ids_complete ||
      topology.numa_nodes > kCpuPlannerReportedNumaNodeLimitV1)
    return "complete CPU topology NUMA-node IDs are required";
  for (std::uint32_t index = 0; index < topology.numa_nodes; ++index) {
    if (topology.numa_node_ids[index] == platform::kUnknownTopologyIdV1 ||
        (index != 0 && topology.numa_node_ids[index] <=
                           topology.numa_node_ids[index - 1]))
      return "CPU topology NUMA-node IDs must be known, unique, and sorted";
  }
  if (topology.physical_cores > topology.logical_processors)
    return "physical core count exceeds logical processor count";
  if (topology.available_processors > topology.logical_processors)
    return "available processor count exceeds logical processor count";
  return {};
}

std::string_view thread_policy_reason(
    const CpuThreadPolicyV1 &policy) noexcept {
  if (policy.version != kCpuThreadPolicyVersionV1)
    return "CPU thread policy version is unsupported";
  if (policy.requested_threads == 0)
    return "requested thread count must be positive";
  return {};
}

bool known_affinity(platform::CpuAffinityPolicyV1 affinity) noexcept {
  switch (affinity) {
    case platform::CpuAffinityPolicyV1::compact:
    case platform::CpuAffinityPolicyV1::scatter:
    case platform::CpuAffinityPolicyV1::local_first:
      return true;
  }
  return false;
}

bool known_numa_policy(CpuPlannerNumaPolicyV1 policy) noexcept {
  switch (policy) {
    case CpuPlannerNumaPolicyV1::single_node:
    case CpuPlannerNumaPolicyV1::local_first:
      return true;
  }
  return false;
}

std::string_view numa_policy_name(CpuPlannerNumaPolicyV1 policy) noexcept {
  switch (policy) {
    case CpuPlannerNumaPolicyV1::single_node:
      return "single-node";
    case CpuPlannerNumaPolicyV1::local_first:
      return "local-first";
  }
  return "invalid";
}

std::string_view placement_reason(
    const CpuPlannerTopologyViewV1 &topology,
    const CpuPlannerPlacementEvidenceV1 &placement) noexcept {
  if (placement.version != kCpuPlannerPlacementEvidenceVersionV1)
    return "CPU placement evidence version is unsupported";
  if (!placement.evidence_complete)
    return "complete CPU placement evidence is required for parallel planning";
  if (!known_affinity(placement.affinity) ||
      !known_numa_policy(placement.numa))
    return "CPU placement policy is invalid";
  if (placement.local_logical_processor_capacity == 0 ||
      placement.local_physical_core_capacity == 0 ||
      placement.local_physical_core_capacity >
          placement.local_logical_processor_capacity ||
      placement.local_logical_processor_capacity >
          topology.logical_processors ||
      placement.local_physical_core_capacity > topology.physical_cores)
    return "CPU placement local capacity is invalid";
  if (placement.selected_numa_node_count == 0 ||
      placement.selected_numa_node_count > topology.numa_nodes ||
      placement.selected_numa_node_count >
          kCpuPlannerReportedNumaNodeLimitV1)
    return "CPU placement selected NUMA-node count is invalid";
  for (std::uint32_t index = 0;
       index < placement.selected_numa_node_count; ++index) {
    if (placement.selected_numa_nodes[index] ==
            platform::kUnknownTopologyIdV1 ||
        (index != 0 && placement.selected_numa_nodes[index] <=
                           placement.selected_numa_nodes[index - 1]))
      return "CPU placement NUMA-node IDs must be known, unique, and sorted";
    if (!std::binary_search(topology.numa_node_ids.begin(),
                            topology.numa_node_ids.begin() +
                                topology.numa_nodes,
                            placement.selected_numa_nodes[index]))
      return "CPU placement references a NUMA node absent from topology";
  }
  if (placement.crosses_numa_nodes !=
      (placement.selected_numa_node_count > 1))
    return "CPU placement cross-NUMA evidence is inconsistent";
  if (placement.affinity_applied && !placement.affinity_requested)
    return "CPU affinity cannot be applied when it was not requested";
  if (placement.affinity_requested && !placement.affinity_applied)
    return "requested CPU affinity was not completely applied";
  if (placement.numa == CpuPlannerNumaPolicyV1::single_node &&
      placement.crosses_numa_nodes)
    return "single-node NUMA policy cannot cross NUMA nodes";
  if (placement.numa == CpuPlannerNumaPolicyV1::single_node &&
      topology.numa_nodes > 1 && !placement.affinity_applied)
    return "multi-node single-node policy requires applied affinity";
  if (placement.crosses_numa_nodes &&
      !placement.caller_first_touch_required)
    return "cross-NUMA placement must expose caller-owned first touch";
  return {};
}

std::uint32_t local_thread_capacity(
    const CpuThreadPolicyV1 &policy,
    const CpuPlannerPlacementEvidenceV1 &placement) noexcept {
  return policy.allow_smt ? placement.local_logical_processor_capacity
                          : placement.local_physical_core_capacity;
}

std::uint32_t policy_thread_ceiling(
    const CpuPlannerTopologyViewV1 &topology,
    const CpuThreadPolicyV1 &policy,
    const CpuGemmImplementationResourcesV2 &resources,
    const CpuPlannerPlacementEvidenceV1 &placement) noexcept {
  std::uint32_t result = policy.requested_threads;
  if (policy.maximum_threads != 0) result = std::min(result, policy.maximum_threads);
  result = std::min(result, resources.execution_context_worker_capacity);
  result = std::min(result, topology.available_processors);
  result = std::min(result, policy.allow_smt ? topology.logical_processors
                                             : topology.physical_cores);
  if (placement.evidence_complete &&
      placement.numa == CpuPlannerNumaPolicyV1::single_node) {
    result = std::min(result, local_thread_capacity(policy, placement));
  }
  return result;
}

std::uint64_t divide_round_up(std::uint64_t numerator,
                              std::uint64_t denominator) noexcept {
  if (denominator == 0) return std::numeric_limits<std::uint64_t>::max();
  return numerator / denominator + (numerator % denominator != 0 ? 1U : 0U);
}

bool total_parallel_workspace(std::uint64_t shared_bytes,
                        std::uint64_t per_worker_bytes,
                        std::uint32_t threads,
                        std::uint64_t *total) noexcept {
  if (total == nullptr ||
      (per_worker_bytes != 0 &&
       threads > std::numeric_limits<std::uint64_t>::max() /
                     per_worker_bytes))
    return false;
  const std::uint64_t worker_bytes = per_worker_bytes * threads;
  if (worker_bytes > std::numeric_limits<std::uint64_t>::max() - shared_bytes)
    return false;
  *total = shared_bytes + worker_bytes;
  return true;
}

std::uint64_t parallel_cost(const CpuGemmProblemV1 &problem,
                            std::uint32_t threads, bool avx512,
                            bool crosses_numa_nodes) noexcept {
  const std::uint64_t work = detail::operation_count(problem);
  const std::uint64_t compute_factor = avx512 ? 1 : 2;
  const std::uint64_t compute = divide_round_up(
      detail::saturating_multiply(work, compute_factor), threads);
  const std::uint64_t a_pack = detail::saturating_multiply(
      detail::saturating_multiply(static_cast<std::uint64_t>(problem.m),
                                  static_cast<std::uint64_t>(problem.k)),
      6);
  // B is packed once into a caller-owned shared immutable image before the
  // row-band workers run. Only transient A storage is per-worker.
  const std::uint64_t b_pack = detail::saturating_multiply(
      detail::saturating_multiply(static_cast<std::uint64_t>(problem.k),
                                  static_cast<std::uint64_t>(problem.n)),
      6);
  std::uint64_t result = detail::saturating_add(
      detail::saturating_add(compute, detail::saturating_add(a_pack, b_pack)),
      detail::saturating_add(UINT64_C(200000),
                             detail::saturating_multiply(threads, 20000)));
  if (crosses_numa_nodes) {
    // Cross-node execution is permitted only through explicit evidence.  The
    // planner applies a conservative static penalty for remote output traffic,
    // shared packed-B reads, and coordination.  This is deterministic policy,
    // not an assertion about a particular NUMA fabric's measured bandwidth.
    const std::uint64_t remote_compute = divide_round_up(work, 4);
    const std::uint64_t remote_packing = detail::saturating_multiply(b_pack, 2);
    result = detail::saturating_add(
        result, detail::saturating_add(
                    detail::saturating_add(remote_compute, remote_packing),
                    UINT64_C(250000)));
  }
  return result;
}

std::uint64_t avx512_single_cost(const CpuGemmProblemV1 &problem) noexcept {
  const std::uint64_t work = detail::operation_count(problem);
  const std::uint64_t packing = detail::saturating_add(
      detail::saturating_multiply(static_cast<std::uint64_t>(problem.m),
                                  static_cast<std::uint64_t>(problem.k)),
      detail::saturating_multiply(static_cast<std::uint64_t>(problem.k),
                                  static_cast<std::uint64_t>(problem.n)));
  return detail::saturating_add(
      detail::saturating_add(work, detail::saturating_multiply(packing, 5)),
      48000);
}

std::uint64_t external_cost(const CpuGemmProblemV1 &problem,
                            std::uint32_t threads) noexcept {
  if (problem.m == 1) return std::numeric_limits<std::uint64_t>::max();
  return detail::saturating_add(
      divide_round_up(detail::operation_count(problem), threads), 2000);
}

void populate_resource_metadata(
    CpuCandidateDecisionV3 *decision,
    const CpuGemmImplementationResourcesV2 &resources) noexcept {
  if (decision == nullptr) return;
  switch (decision->variant) {
    case CpuGemmVariantV3::reference:
    case CpuGemmVariantV3::tiled:
      decision->runtime_validated = true;
      break;
    case CpuGemmVariantV3::compiler_vectorized:
      decision->runtime_validated =
          resources.native_packed_avx2_fma_runtime_validated;
      break;
    case CpuGemmVariantV3::external_openblas:
      decision->runtime_validated = resources.baseline.openblas_linked;
      break;
    case CpuGemmVariantV3::native_packed_avx2_fma:
      decision->runtime_validated =
          resources.native_packed_avx2_fma_runtime_validated;
      if (resources.baseline.native_packed_workspace_size_valid) {
        decision->required_workspace_bytes =
            resources.baseline.native_packed_workspace_bytes;
        decision->shared_workspace_bytes =
            resources.baseline.native_packed_workspace_bytes;
        decision->required_workspace_alignment =
            resources.baseline.native_packed_workspace_alignment;
      }
      break;
    case CpuGemmVariantV3::native_packed_avx512_fma:
      decision->runtime_validated =
          resources.native_packed_avx512_fma_runtime_validated;
      if (resources.native_packed_avx512_workspace_size_valid) {
        decision->required_workspace_bytes =
            resources.native_packed_avx512_workspace_bytes;
        decision->shared_workspace_bytes =
            resources.native_packed_avx512_workspace_bytes;
        decision->required_workspace_alignment =
            resources.native_packed_avx512_workspace_alignment;
      }
      break;
    case CpuGemmVariantV3::native_parallel_avx2_fma:
      decision->runtime_validated =
          resources.native_packed_avx2_fma_runtime_validated;
      if (resources.native_parallel_avx2_workspace_size_valid) {
        decision->shared_workspace_bytes =
            resources.native_parallel_avx2_shared_workspace_bytes;
        decision->per_worker_workspace_bytes =
            resources.native_parallel_avx2_per_worker_workspace_bytes;
        decision->required_workspace_alignment =
            resources.native_parallel_avx2_workspace_alignment;
      }
      break;
    case CpuGemmVariantV3::native_parallel_avx512_fma:
      decision->runtime_validated =
          resources.native_packed_avx512_fma_runtime_validated;
      if (resources.native_parallel_avx512_workspace_size_valid) {
        decision->shared_workspace_bytes =
            resources.native_parallel_avx512_shared_workspace_bytes;
        decision->per_worker_workspace_bytes =
            resources.native_parallel_avx512_per_worker_workspace_bytes;
        decision->required_workspace_alignment =
            resources.native_parallel_avx512_workspace_alignment;
      }
      break;
  }
}

}  // namespace

const std::array<CpuGemmVariantRecordV3, kCpuGemmCandidateCountV3> &
cpu_gemm_variant_registry_v3() noexcept {
  return kCpuGemmVariantRegistryV3;
}

CpuPlannerCapabilityProjectionV1 project_cpu_capabilities_v2_for_planner_v1(
    const platform::CpuCapabilitiesV2 &capabilities) noexcept {
  CpuPlannerCapabilityProjectionV1 result;
  const platform::CpuCapabilitiesValidationV2 validation =
      platform::validate_cpu_capabilities_v2(capabilities);
  result.valid = validation.valid;
  result.reason = validation.reason;
  if (!validation.valid) return result;

  result.baseline.version = kCpuCapabilitiesVersionV1;
  switch (capabilities.architecture) {
    case platform::ArchitectureKindV1::x86_64:
      result.baseline.architecture = CpuArchitectureV1::x86_64;
      break;
    case platform::ArchitectureKindV1::aarch64:
      result.baseline.architecture = CpuArchitectureV1::aarch64;
      break;
    case platform::ArchitectureKindV1::unknown:
      result.baseline.architecture = CpuArchitectureV1::unknown;
      break;
  }
  result.baseline.detection_complete =
      capabilities.architecture != platform::ArchitectureKindV1::unknown &&
      platform::domain_complete_v2(capabilities.hardware,
                                   capabilities.architecture) &&
      platform::domain_complete_v2(capabilities.os_enabled,
                                   capabilities.architecture) &&
      platform::domain_complete_v2(capabilities.compiler,
                                   capabilities.architecture);
  result.baseline.features = 0;
  if (platform::has_usable_feature_v2(
          capabilities, platform::CpuFeatureV2::portable_scalar_f32)) {
    result.baseline.features |= feature_bit(CpuFeatureV1::portable_scalar_f32);
  }
  if (platform::has_usable_feature_v2(capabilities,
                                      platform::CpuFeatureV2::avx2)) {
    result.baseline.features |= feature_bit(CpuFeatureV1::avx2);
  }
  if (platform::has_usable_feature_v2(capabilities,
                                      platform::CpuFeatureV2::fma)) {
    result.baseline.features |= feature_bit(CpuFeatureV1::fma);
  }
  result.baseline.usable_vector_bits = capabilities.usable_vector_bits;

  result.avx512f_hardware = platform::feature_available(
      capabilities.hardware, platform::CpuFeatureV2::avx512f);
  result.avx512f_os_enabled = platform::feature_available(
      capabilities.os_enabled, platform::CpuFeatureV2::avx512f);
  result.avx512f_compiler_supported = platform::feature_available(
      capabilities.compiler, platform::CpuFeatureV2::avx512f);
  result.avx512f_implementation_available = platform::feature_available(
      capabilities.implementation, platform::CpuFeatureV2::avx512f);
  result.avx512f_runtime_validated =
      platform::has_runtime_validated_feature_v2(
          capabilities, platform::CpuFeatureV2::avx512f);
  result.avx2_implementation_available = platform::feature_available(
      capabilities.implementation, platform::CpuFeatureV2::avx2);
  result.avx2_runtime_validated = platform::has_runtime_validated_feature_v2(
      capabilities, platform::CpuFeatureV2::avx2);
  result.fma_implementation_available = platform::feature_available(
      capabilities.implementation, platform::CpuFeatureV2::fma);
  result.fma_runtime_validated = platform::has_runtime_validated_feature_v2(
      capabilities, platform::CpuFeatureV2::fma);
  return result;
}

CpuPlannerTopologyViewV1 project_cpu_topology_v1_for_planner_v1(
    const platform::CpuTopologyV1 &topology,
    std::uint32_t available_processors_override) noexcept {
  CpuPlannerTopologyViewV1 result;
  const platform::CpuTopologyValidationV1 validation =
      platform::validate_cpu_topology_v1(topology);
  result.discovery_complete = validation.valid && topology.discovery_complete;
  result.logical_processors = platform::logical_cpu_count_v1(topology);
  result.physical_cores = platform::physical_core_count_v1(topology);
  result.available_processors =
      available_processors_override == 0 ? result.logical_processors
                                         : available_processors_override;
  result.numa_nodes = platform::numa_node_count_v1(topology);
  result.numa_node_ids_complete =
      topology.numa_nodes.size() <= kCpuPlannerReportedNumaNodeLimitV1;
  if (result.numa_node_ids_complete) {
    for (std::size_t index = 0; index < topology.numa_nodes.size(); ++index) {
      result.numa_node_ids[index] = topology.numa_nodes[index].node_id;
    }
  }
  if (result.available_processors > result.logical_processors)
    result.discovery_complete = false;
  if (available_processors_override != 0 &&
      available_processors_override != result.logical_processors)
    result.discovery_complete = false;
  if (!result.numa_node_ids_complete) result.discovery_complete = false;
  return result;
}

CpuGemmPlanV3 plan_cpu_gemm_v3(
    const CpuGemmProblemV1 &problem,
    const CpuCapabilitiesV1 &baseline_capabilities,
    const CpuPlannerTopologyViewV1 &topology,
    const CpuThreadPolicyV1 &thread_policy,
    const CpuGemmImplementationResourcesV2 &resources,
    CpuGemmRequestV3 request,
    const CpuPlannerPlacementEvidenceV1 &placement) noexcept {
  CpuGemmPlanV3 plan;
  plan.problem = problem;
  plan.baseline_capabilities = baseline_capabilities;
  plan.topology = topology;
  plan.thread_policy = thread_policy;
  plan.placement = placement;
  plan.resources = resources;
  plan.request = request;

  const std::string_view policy_error = thread_policy_reason(thread_policy);
  const CpuGemmPlanV1 problem_check = plan_cpu_gemm_v1(
      problem, baseline_capabilities, CpuGemmRequestV1::force_reference);
  if (problem_check.status == CpuPlanStatusV1::invalid_problem ||
      problem_check.status == CpuPlanStatusV1::invalid_capabilities) {
    plan.status = problem_check.status;
    plan.selection_reason = problem_check.selection_reason;
  } else if (!policy_error.empty()) {
    plan.status = CpuPlanStatusV1::invalid_capabilities;
    plan.selection_reason = policy_error;
  }

  const std::string_view topology_error = topology_reason(topology);
  const std::string_view placement_error =
      topology_error.empty() ? placement_reason(topology, placement)
                             : topology_error;
  const std::uint64_t macro_tiles =
      problem.m > 0
          ? (static_cast<std::uint64_t>(problem.m) +
             kParallelMacroTileRowsV1 - 1U) /
                kParallelMacroTileRowsV1
          : 0;
  const std::uint32_t thread_ceiling =
      topology_error.empty()
          ? policy_thread_ceiling(topology, thread_policy, resources,
                                  placement)
          : 0;

  bool found = false;
  std::uint64_t best_cost = std::numeric_limits<std::uint64_t>::max();
  std::uint16_t best_priority = std::numeric_limits<std::uint16_t>::max();
  std::size_t best_index = kCpuGemmCandidateCountV3;

  for (std::size_t index = 0; index < kCpuGemmCandidateCountV3; ++index) {
    const CpuGemmVariantRecordV3 &record = kCpuGemmVariantRegistryV3[index];
    CpuCandidateDecisionV3 &decision = plan.candidates[index];
    decision.variant = record.variant;
    decision.stable_id = record.stable_id;
    decision.deterministic_priority = record.deterministic_priority;
    populate_requirement_metadata(&decision);
    populate_resource_metadata(&decision, resources);
    if (plan.status == CpuPlanStatusV1::invalid_problem ||
        plan.status == CpuPlanStatusV1::invalid_capabilities) {
      decision.reason = plan.selection_reason;
      continue;
    }

    if (record.variant <= CpuGemmVariantV3::native_packed_avx2_fma) {
      CpuGemmImplementationResourcesV1 v2_resources = resources.baseline;
      v2_resources.requested_threads = 1;
      if (record.variant == CpuGemmVariantV3::external_openblas) {
        const std::uint32_t topology_limit = topology_error.empty()
                                                 ? std::min(
                                                       topology.available_processors,
                                                       thread_policy.allow_smt
                                                           ? topology.logical_processors
                                                           : topology.physical_cores)
                                                 : 1;
        std::uint32_t provider_threads =
            std::min(thread_policy.requested_threads, topology_limit);
        if (thread_policy.maximum_threads != 0)
          provider_threads =
              std::min(provider_threads, thread_policy.maximum_threads);
        provider_threads =
            std::min(provider_threads, v2_resources.openblas_maximum_threads);
        v2_resources.requested_threads = provider_threads;
      }
      const CpuGemmPlanV2 base = plan_cpu_gemm_v2(
          problem, baseline_capabilities, v2_resources,
          v2_request(record.variant));
      const CpuCandidateDecisionV2 &base_decision =
          base.candidates[v2_index(record.variant)];
      decision.legal = base_decision.legal;
      decision.reason = base_decision.reason;
      decision.required_workspace_bytes =
          base_decision.required_workspace_bytes;
      decision.required_workspace_alignment =
          base_decision.required_workspace_alignment;
      decision.actual_threads = base_decision.actual_threads;
      decision.estimated_cost = base_decision.estimated_cost;
      decision.shared_workspace_bytes =
          base_decision.required_workspace_bytes;
      if (record.variant == CpuGemmVariantV3::reference ||
          record.variant == CpuGemmVariantV3::tiled) {
        decision.runtime_validated = true;
      } else if (record.variant == CpuGemmVariantV3::external_openblas) {
        decision.runtime_validated = resources.baseline.openblas_linked;
      } else if (record.variant == CpuGemmVariantV3::compiler_vectorized) {
        decision.runtime_validated =
            resources.native_packed_avx2_fma_runtime_validated;
        if (decision.legal && !decision.runtime_validated) {
          decision.legal = false;
          decision.reason =
              "compiler-vectorized AVX2/FMA implementation is not runtime-validated";
          decision.estimated_cost = std::numeric_limits<std::uint64_t>::max();
        }
      } else {
        decision.runtime_validated =
            resources.native_packed_avx2_fma_runtime_validated;
        if (decision.legal && !decision.runtime_validated) {
          decision.legal = false;
          decision.reason =
              "native packed AVX2/FMA implementation is not runtime-validated";
          decision.estimated_cost = std::numeric_limits<std::uint64_t>::max();
        }
      }
      if (record.variant == CpuGemmVariantV3::external_openblas &&
          decision.legal) {
        decision.estimated_cost = external_cost(problem, decision.actual_threads);
      }
    } else if (record.variant ==
               CpuGemmVariantV3::native_packed_avx512_fma) {
      if (!resources.avx512f_hardware)
        decision.reason = "AVX-512F hardware support is unavailable";
      else if (!baseline_capabilities.detection_complete ||
               baseline_capabilities.architecture != CpuArchitectureV1::x86_64)
        decision.reason = "AVX-512 F32 candidate requires complete x86_64 discovery";
      else if (!has_feature(baseline_capabilities, CpuFeatureV1::fma))
        decision.reason = "AVX-512 F32 candidate requires FMA";
      else if (!resources.avx512f_os_enabled)
        decision.reason = "AVX-512F architectural state is not OS-enabled";
      else if (!resources.avx512f_compiler_supported)
        decision.reason = "AVX-512F compiler support is unavailable";
      else if (!resources.native_packed_avx512_fma_compiled)
        decision.reason = "native packed AVX-512 implementation is not compiled";
      else if (!resources.native_packed_avx512_fma_runtime_validated)
        decision.reason = "native packed AVX-512 implementation is not runtime-validated";
      else if (!resources.native_packed_avx512_workspace_size_valid)
        decision.reason = "native packed AVX-512 workspace requirement overflowed";
      else if (resources.native_packed_avx512_workspace_alignment < 64 ||
               (resources.native_packed_avx512_workspace_alignment &
                (resources.native_packed_avx512_workspace_alignment - 1U)) != 0)
        decision.reason = "native packed AVX-512 workspace alignment is invalid";
      else {
        decision.legal = true;
        decision.reason = "legal";
        decision.actual_threads = 1;
        decision.required_workspace_bytes =
            resources.native_packed_avx512_workspace_bytes;
        decision.required_workspace_alignment =
            resources.native_packed_avx512_workspace_alignment;
        decision.shared_workspace_bytes =
            resources.native_packed_avx512_workspace_bytes;
        decision.runtime_validated = true;
        decision.estimated_cost = avx512_single_cost(problem);
      }
    } else {
      const bool avx512 = record.variant ==
                          CpuGemmVariantV3::native_parallel_avx512_fma;
      const std::uint32_t actual_threads = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(thread_ceiling, macro_tiles));
      decision.actual_threads = actual_threads;
      std::uint64_t total_workspace = 0;
      if (decision.shared_workspace_bytes != 0 &&
          decision.per_worker_workspace_bytes != 0 && actual_threads != 0 &&
          total_parallel_workspace(decision.shared_workspace_bytes,
                                   decision.per_worker_workspace_bytes,
                                   actual_threads, &total_workspace)) {
        decision.required_workspace_bytes = total_workspace;
      }
      if (placement.evidence_complete && placement.crosses_numa_nodes) {
        decision.crosses_numa_nodes =
            placement.affinity != platform::CpuAffinityPolicyV1::local_first ||
            actual_threads > local_thread_capacity(thread_policy, placement);
      }

      if (!placement_error.empty())
        decision.reason = placement_error;
      else if (!resources.execution_context_available ||
               resources.execution_context_worker_capacity == 0)
        decision.reason = "persistent CPU execution context is unavailable";
      else if (thread_policy.external_provider_parallelism_active)
        decision.reason = "native/provider nested parallelism is prohibited";
      else if ((!avx512 && !resources.native_parallel_avx2_fma_compiled) ||
               (avx512 && !resources.native_parallel_avx512_fma_compiled))
        decision.reason = avx512
                              ? "parallel AVX-512 implementation is not compiled"
                              : "parallel AVX2/FMA implementation is not compiled";
      else if (!avx512 &&
               (!resources.baseline.native_packed_avx2_fma_compiled ||
                !resources.native_parallel_avx2_workspace_size_valid))
        decision.reason = "parallel AVX2/FMA workspace implementation is unavailable";
      else if (avx512 &&
               (!resources.native_packed_avx512_fma_compiled ||
                !resources.native_parallel_avx512_workspace_size_valid))
        decision.reason = "parallel AVX-512 workspace implementation is unavailable";
      else if (avx512 && !resources.avx512f_hardware)
        decision.reason = "AVX-512F hardware support is unavailable";
      else if (avx512 &&
               (!baseline_capabilities.detection_complete ||
                baseline_capabilities.architecture != CpuArchitectureV1::x86_64))
        decision.reason = "parallel AVX-512 requires complete x86_64 discovery";
      else if (avx512 && !resources.avx512f_os_enabled)
        decision.reason = "AVX-512F architectural state is not OS-enabled";
      else if (avx512 && !resources.avx512f_compiler_supported)
        decision.reason = "AVX-512F compiler support is unavailable";
      else if (avx512 &&
               !resources.native_packed_avx512_fma_runtime_validated)
        decision.reason = "native packed AVX-512 implementation is not runtime-validated";
      else if (!avx512 &&
               (!baseline_capabilities.detection_complete ||
                baseline_capabilities.architecture != CpuArchitectureV1::x86_64 ||
                !has_feature(baseline_capabilities, CpuFeatureV1::avx2) ||
                !has_feature(baseline_capabilities, CpuFeatureV1::fma) ||
                baseline_capabilities.usable_vector_bits < 256))
        decision.reason = "parallel AVX2/FMA requires usable 256-bit state";
      else if (!avx512 &&
               !resources.native_packed_avx2_fma_runtime_validated)
        decision.reason =
            "native packed AVX2/FMA implementation is not runtime-validated";
      else if (avx512 &&
               !has_feature(baseline_capabilities, CpuFeatureV1::fma))
        decision.reason = "parallel AVX-512 F32 requires FMA";
      else {
        if (actual_threads < 2)
          decision.reason = "parallel candidate requires at least two output macro-tiles and workers";
        else if (divide_round_up(detail::operation_count(problem),
                                 actual_threads) <
                 kCpuParallelMinimumWorkPerThreadV1)
          decision.reason = "parallel work per thread is below the deterministic threshold";
        else {
          if (decision.shared_workspace_bytes == 0 ||
              decision.per_worker_workspace_bytes == 0)
            decision.reason = "parallel workspace requirements must be nonzero";
          else if (decision.required_workspace_alignment < 32 ||
                   (decision.required_workspace_alignment &
                    (decision.required_workspace_alignment - 1U)) != 0)
            decision.reason = "parallel per-worker workspace alignment is invalid";
          else if (decision.required_workspace_bytes == 0)
            decision.reason = "parallel workspace requirement overflowed";
          else {
            decision.legal = true;
            decision.reason = "legal";
            decision.estimated_cost =
                parallel_cost(problem, actual_threads, avx512,
                              decision.crosses_numa_nodes);
          }
        }
      }
    }

    if (!decision.legal || !request_matches(request, record.variant)) continue;
    if (!found || decision.estimated_cost < best_cost ||
        (decision.estimated_cost == best_cost &&
         (decision.deterministic_priority < best_priority ||
          (decision.deterministic_priority == best_priority &&
           index < best_index)))) {
      found = true;
      best_cost = decision.estimated_cost;
      best_priority = decision.deterministic_priority;
      best_index = index;
      plan.selected_variant = record.variant;
      plan.selected_id = record.stable_id;
    }
  }

  if (plan.status == CpuPlanStatusV1::invalid_problem ||
      plan.status == CpuPlanStatusV1::invalid_capabilities)
    return plan;
  if (!found) {
    plan.status = request == CpuGemmRequestV3::automatic
                      ? CpuPlanStatusV1::no_legal_variant
                      : CpuPlanStatusV1::forced_variant_illegal;
    plan.selection_reason = request == CpuGemmRequestV3::automatic
                                ? "no legal CPU GEMM variant"
                                : "requested CPU GEMM variant is illegal";
    return plan;
  }
  plan.status = CpuPlanStatusV1::selected;
  plan.selection_reason =
      request == CpuGemmRequestV3::automatic
          ? "lowest deterministic static cost; ties use priority then registry order"
          : "explicit legal variant request";
  return plan;
}

CpuGemmPlanV3 plan_cpu_gemm_v3(
    const CpuGemmProblemV1 &problem,
    const platform::CpuCapabilitiesV2 &capabilities,
    const platform::CpuTopologyV1 &topology,
    const CpuThreadPolicyV1 &thread_policy,
    const CpuGemmImplementationResourcesV2 &resources,
    CpuGemmRequestV3 request,
    std::uint32_t available_processors_override,
    const CpuPlannerPlacementEvidenceV1 &placement) noexcept {
  const CpuPlannerCapabilityProjectionV1 capability =
      project_cpu_capabilities_v2_for_planner_v1(capabilities);
  const CpuPlannerTopologyViewV1 topology_view =
      project_cpu_topology_v1_for_planner_v1(
          topology, available_processors_override);
  CpuGemmImplementationResourcesV2 normalized = resources;
  normalized.avx512f_hardware = capability.avx512f_hardware;
  normalized.avx512f_os_enabled = capability.avx512f_os_enabled;
  normalized.avx512f_compiler_supported =
      capability.avx512f_compiler_supported;
  normalized.baseline.native_packed_avx2_fma_compiled =
      normalized.baseline.native_packed_avx2_fma_compiled &&
      capability.avx2_implementation_available &&
      capability.fma_implementation_available;
  normalized.native_parallel_avx2_fma_compiled =
      normalized.native_parallel_avx2_fma_compiled &&
      capability.avx2_implementation_available &&
      capability.fma_implementation_available;
  normalized.native_packed_avx2_fma_runtime_validated =
      normalized.native_packed_avx2_fma_runtime_validated &&
      capability.avx2_runtime_validated && capability.fma_runtime_validated;
  normalized.native_packed_avx512_fma_compiled =
      normalized.native_packed_avx512_fma_compiled &&
      capability.avx512f_implementation_available;
  normalized.native_parallel_avx512_fma_compiled =
      normalized.native_parallel_avx512_fma_compiled &&
      capability.avx512f_implementation_available;
  normalized.native_packed_avx512_fma_runtime_validated =
      normalized.native_packed_avx512_fma_runtime_validated &&
      capability.avx512f_runtime_validated;

  if (!capability.valid) {
    CpuGemmPlanV3 invalid;
    invalid.problem = problem;
    invalid.baseline_capabilities = capability.baseline;
    invalid.topology = topology_view;
    invalid.thread_policy = thread_policy;
    invalid.placement = placement;
    invalid.resources = normalized;
    invalid.request = request;
    invalid.status = CpuPlanStatusV1::invalid_capabilities;
    invalid.selection_reason = capability.reason;
    for (std::size_t index = 0; index < invalid.candidates.size(); ++index) {
      invalid.candidates[index].variant = kCpuGemmVariantRegistryV3[index].variant;
      invalid.candidates[index].stable_id =
          kCpuGemmVariantRegistryV3[index].stable_id;
      invalid.candidates[index].deterministic_priority =
          kCpuGemmVariantRegistryV3[index].deterministic_priority;
      populate_requirement_metadata(&invalid.candidates[index]);
      populate_resource_metadata(&invalid.candidates[index], normalized);
      invalid.candidates[index].reason = capability.reason;
    }
    return invalid;
  }
  return plan_cpu_gemm_v3(problem, capability.baseline, topology_view,
                          thread_policy, normalized, request, placement);
}

std::size_t format_cpu_gemm_plan_v3(const CpuGemmPlanV3 &plan,
                                    char *output,
                                    std::size_t capacity) noexcept {
  if (output == nullptr) capacity = 0;
  detail::DiagnosticWriter writer{output, capacity};
  writer.text("cpu-planner-v3 request-id=");
  writer.number(static_cast<std::uint64_t>(plan.request));
  writer.text(" request=");
  writer.text(request_name(plan.request));
  writer.text(" requested-threads=");
  writer.number(plan.thread_policy.requested_threads);
  writer.text(" max-threads=");
  writer.number(plan.thread_policy.maximum_threads);
  writer.text(" allow-smt=");
  writer.text(plan.thread_policy.allow_smt ? "true" : "false");
  writer.text(" external-provider-parallelism=");
  writer.text(plan.thread_policy.external_provider_parallelism_active
                  ? "true"
                  : "false");
  writer.text(" physical-cores=");
  writer.number(plan.topology.physical_cores);
  writer.text(" logical-processors=");
  writer.number(plan.topology.logical_processors);
  writer.text(" available-processors=");
  writer.number(plan.topology.available_processors);
  writer.text(" numa-nodes=");
  writer.number(plan.topology.numa_nodes);
  writer.text(" placement-complete=");
  writer.text(plan.placement.evidence_complete ? "true" : "false");
  writer.text(" affinity-requested=");
  writer.text(plan.placement.affinity_requested ? "true" : "false");
  writer.text(" affinity-applied=");
  writer.text(plan.placement.affinity_applied ? "true" : "false");
  writer.text(" affinity=");
  writer.text(platform::to_string(plan.placement.affinity));
  writer.text(" numa-policy=");
  writer.text(numa_policy_name(plan.placement.numa));
  writer.text(" local-logical-capacity=");
  writer.number(plan.placement.local_logical_processor_capacity);
  writer.text(" local-physical-capacity=");
  writer.number(plan.placement.local_physical_core_capacity);
  writer.text(" selected-numa-nodes=[");
  for (std::uint32_t index = 0;
       index < plan.placement.selected_numa_node_count &&
       index < kCpuPlannerReportedNumaNodeLimitV1;
       ++index) {
    if (index != 0) writer.character(',');
    writer.number(plan.placement.selected_numa_nodes[index]);
  }
  writer.character(']');
  writer.text(" cross-numa=");
  writer.text(plan.placement.crosses_numa_nodes ? "true" : "false");
  writer.text(" caller-first-touch=");
  writer.text(plan.placement.caller_first_touch_required ? "true" : "false");
  writer.text(" status=");
  writer.text(status_name(plan.status));
  writer.text(" selected=");
  writer.text(plan.selected_id.empty() ? "none" : plan.selected_id);
  writer.text(" reason=");
  writer.text(plan.selection_reason);
  writer.text(" candidates=[");
  for (std::size_t index = 0; index < plan.candidates.size(); ++index) {
    if (index != 0) writer.character(',');
    const CpuCandidateDecisionV3 &candidate = plan.candidates[index];
    writer.text(candidate.stable_id);
    writer.character(':');
    writer.text(candidate.legal ? "legal" : "rejected");
    writer.text(":reason=");
    writer.text(candidate.reason);
    writer.text(":cost=");
    writer.number(candidate.estimated_cost);
    writer.text(":priority=");
    writer.number(candidate.deterministic_priority);
    writer.text(":workspace=");
    writer.number(candidate.required_workspace_bytes);
    writer.text(":shared-workspace=");
    writer.number(candidate.shared_workspace_bytes);
    writer.text(":per-worker-workspace=");
    writer.number(candidate.per_worker_workspace_bytes);
    writer.text(":alignment=");
    writer.number(candidate.required_workspace_alignment);
    writer.text(":threads=");
    writer.number(candidate.actual_threads);
    writer.text(":runtime-validated=");
    writer.text(candidate.runtime_validated ? "true" : "false");
    writer.text(":required-hardware=");
    writer.number(candidate.required_hardware_features);
    writer.text(":required-os=");
    writer.number(candidate.required_os_features);
    writer.text(":required-compiler=");
    writer.number(candidate.required_compiler_features);
    writer.text(":required-implementation=");
    writer.number(candidate.required_implementation_features);
    writer.text(":cross-numa=");
    writer.text(candidate.crosses_numa_nodes ? "true" : "false");
  }
  writer.character(']');
  return writer.finish();
}

}  // namespace matcore::mdslc::planner
