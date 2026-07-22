#include "cpu_planner_v3.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

namespace {

namespace planner = matcore::mdslc::planner;
namespace platform = matcore::mdslc::platform;

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

planner::CpuGemmProblemV1 problem(std::int64_t m, std::int64_t n,
                                  std::int64_t k) {
  return {m, n, k, planner::CpuScalarTypeV1::f32,
          planner::CpuScalarTypeV1::f32,
          planner::CpuLayoutV1::row_major_contiguous, 64};
}

planner::CpuCapabilitiesV1 avx2_capabilities() {
  return {planner::kCpuCapabilitiesVersionV1,
          planner::CpuArchitectureV1::x86_64,
          true,
          planner::feature_bit(planner::CpuFeatureV1::portable_scalar_f32) |
              planner::feature_bit(planner::CpuFeatureV1::avx2) |
              planner::feature_bit(planner::CpuFeatureV1::fma),
          256};
}

planner::CpuPlannerTopologyViewV1 topology() {
  planner::CpuPlannerTopologyViewV1 result;
  result.discovery_complete = true;
  result.logical_processors = 24;
  result.physical_cores = 12;
  result.available_processors = 24;
  result.numa_nodes = 1;
  result.numa_node_ids_complete = true;
  result.numa_node_ids[0] = 0;
  return result;
}

planner::CpuThreadPolicyV1 policy(std::uint32_t requested,
                                  std::uint32_t maximum = 0,
                                  bool allow_smt = false,
                                  bool worker_affinity_active = false) {
  return {planner::kCpuThreadPolicyVersionV1, requested, maximum, allow_smt,
          false, worker_affinity_active};
}

planner::CpuPlannerPlacementEvidenceV1 single_node_placement(
    std::uint32_t logical_capacity = 24,
    std::uint32_t physical_capacity = 12) {
  planner::CpuPlannerPlacementEvidenceV1 result;
  result.evidence_complete = true;
  result.affinity_requested = true;
  result.affinity_applied = true;
  result.affinity = platform::CpuAffinityPolicyV1::compact;
  result.numa = planner::CpuPlannerNumaPolicyV1::single_node;
  result.local_logical_processor_capacity = logical_capacity;
  result.local_physical_core_capacity = physical_capacity;
  result.selected_numa_node_count = 1;
  result.selected_numa_nodes[0] = 0;
  result.caller_first_touch_required = true;
  return result;
}

planner::CpuGemmImplementationResourcesV2 resources() {
  planner::CpuGemmImplementationResourcesV2 result;
  result.baseline.openblas_linked = true;
  result.baseline.openblas_local_thread_control = true;
  result.baseline.openblas_maximum_threads = 24;
  result.baseline.native_packed_avx2_fma_compiled = true;
  result.baseline.native_packed_workspace_size_valid = true;
  result.baseline.native_packed_workspace_bytes = 393216;
  result.baseline.native_packed_workspace_alignment = 64;
  result.baseline.requested_threads = 1;
  result.reference_f32_runtime_validated = true;
  result.tiled_f32_runtime_validated = true;
  result.compiler_vectorized_f32_runtime_validated = true;
  result.external_openblas_f32_runtime_validated = true;
  result.native_packed_avx2_fma_runtime_validated = true;
  result.execution_context_available = true;
  result.execution_context_worker_capacity = 24;
  result.native_parallel_avx2_fma_compiled = true;
  result.native_parallel_avx2_fma_runtime_validated = true;
  result.native_parallel_avx2_workspace_size_valid = true;
  result.native_parallel_avx2_shared_workspace_bytes = 262144;
  result.native_parallel_avx2_per_worker_workspace_bytes = 131072;
  result.native_parallel_avx2_workspace_alignment = 64;
  return result;
}

platform::CpuCapabilitiesV2 synthetic_capabilities_v2() {
  const std::uint64_t vector =
      platform::feature_bit(platform::CpuFeatureV2::portable_scalar_f32) |
      platform::feature_bit(platform::CpuFeatureV2::avx2) |
      platform::feature_bit(platform::CpuFeatureV2::fma) |
      platform::feature_bit(platform::CpuFeatureV2::avx512f) |
      platform::feature_bit(platform::CpuFeatureV2::avx512dq) |
      platform::feature_bit(platform::CpuFeatureV2::avx512bw) |
      platform::feature_bit(platform::CpuFeatureV2::avx512vl);
  platform::CpuCapabilitiesV2 result;
  result.architecture = platform::ArchitectureKindV1::x86_64;
  result.hardware = {platform::kKnownCpuFeatureBitsV2, vector};
  result.os_enabled = {platform::kKnownCpuFeatureBitsV2, vector};
  result.compiler = {platform::kKnownCpuFeatureBitsV2, vector};
  result.implementation = {platform::kKnownCpuFeatureBitsV2, vector};
  result.runtime_validation = {platform::kKnownCpuFeatureBitsV2, vector};
  result.os_xstate_mask_known = true;
  result.os_xstate_mask = (UINT64_C(1) << 1) | (UINT64_C(1) << 2) |
                          (UINT64_C(1) << 5) | (UINT64_C(1) << 6) |
                          (UINT64_C(1) << 7);
  result.usable_vector_bits = 512;
  return result;
}

platform::CpuTopologyV1 synthetic_topology_v1() {
  platform::CpuTopologyV1 result;
  result.architecture = platform::ArchitectureKindV1::x86_64;
  result.discovery_complete = true;
  result.logical_processors = {
      {0, 0, 0, 0, 0, true}, {1, 1, 0, 0, 0, true},
      {2, 2, 0, 0, 0, true}, {3, 3, 0, 0, 0, true},
      {4, 0, 0, 0, 1, true}, {5, 1, 0, 0, 1, true},
      {6, 2, 0, 0, 1, true}, {7, 3, 0, 0, 1, true},
  };
  result.numa_nodes = {{0, {0, 1, 2, 3, 4, 5, 6, 7}}};
  result.cache_groups = {
      {3, platform::CpuCacheTypeV1::unified, 8388608, 64,
       {0, 1, 2, 3, 4, 5, 6, 7}}};
  return result;
}

void registry_and_parallel_policy() {
  expect(planner::cpu_gemm_variant_registry_v3().size() == 8 &&
             planner::cpu_gemm_variant_registry_v3()[6].stable_id ==
                 "cpu.native-parallel.avx2-fma.f32.v1" &&
             planner::cpu_gemm_variant_registry_v3()[7].stable_id ==
                 "cpu.native-parallel.avx512-fma.f32.v1",
         "planner v3 registry has stable single and parallel IDs");

  const auto large = problem(1024, 1024, 1024);
  const auto forced = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  const auto &parallel = forced.candidates[6];
  expect(forced.status == planner::CpuPlanStatusV1::selected &&
             forced.selected_id ==
                 "cpu.native-parallel.avx2-fma.f32.v1" &&
             parallel.legal && parallel.actual_threads == 8 &&
             parallel.required_workspace_bytes ==
                 UINT64_C(262144) + UINT64_C(8) * UINT64_C(131072) &&
             parallel.shared_workspace_bytes == UINT64_C(262144) &&
             parallel.per_worker_workspace_bytes == UINT64_C(131072) &&
             parallel.runtime_validated &&
             parallel.required_workspace_alignment == 64,
         "forced parallel AVX2 plan reports exact shared and worker workspace");

  const auto capped = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(16, 4), resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  expect(capped.status == planner::CpuPlanStatusV1::selected &&
             capped.candidates[6].actual_threads == 4,
         "explicit maximum thread policy caps native workers");

  const auto many_tiles = problem(4096, 256, 256);
  const auto physical = planner::plan_cpu_gemm_v3(
      many_tiles, avx2_capabilities(), topology(), policy(24), resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  const auto smt = planner::plan_cpu_gemm_v3(
      many_tiles, avx2_capabilities(), topology(), policy(24, 0, true),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  expect(physical.candidates[6].actual_threads == 12 &&
             smt.candidates[6].actual_threads == 24,
         "physical-core default and explicit SMT policy are deterministic");

  auto provider_resources = resources();
  provider_resources.baseline.openblas_maximum_threads = 3;
  const auto provider = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(12), provider_resources,
      planner::CpuGemmRequestV3::force_external_openblas);
  expect(provider.status == planner::CpuPlanStatusV1::selected &&
             provider.candidates[3].actual_threads == 3,
         "external provider never exceeds its reported thread ceiling");

  const auto bound_provider = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(4, 0, false, true),
      provider_resources,
      planner::CpuGemmRequestV3::force_external_openblas);
  expect(bound_provider.status ==
                 planner::CpuPlanStatusV1::forced_variant_illegal &&
             !bound_provider.candidates[3].legal &&
             bound_provider.candidates[3].reason ==
                 "multi-thread OpenBLAS is unavailable under bound native workers",
         "bound native placement rejects provider-owned multithreading in the planner");

  const auto bound_provider_single = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(1, 0, false, true),
      provider_resources,
      planner::CpuGemmRequestV3::force_external_openblas);
  expect(bound_provider_single.status == planner::CpuPlanStatusV1::selected &&
             bound_provider_single.candidates[3].actual_threads == 1,
         "one provider thread remains executable on a bound native worker");
}

void parallel_rejection_boundaries() {
  const auto missing_placement = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), topology(), policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
  expect(missing_placement.status ==
                 planner::CpuPlanStatusV1::forced_variant_illegal &&
             missing_placement.candidates[6].reason ==
                 "complete CPU placement evidence is required for parallel planning",
         "parallel planning fails closed without injected placement evidence");

  auto unapplied = single_node_placement();
  unapplied.affinity_applied = false;
  const auto affinity_failure = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), topology(), policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma, unapplied);
  expect(affinity_failure.candidates[6].reason ==
             "requested CPU affinity was not completely applied",
         "requested affinity must be completely applied before planning");

  const auto tiny = planner::plan_cpu_gemm_v3(
      problem(32, 32, 32), avx2_capabilities(), topology(), policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  expect(tiny.status == planner::CpuPlanStatusV1::forced_variant_illegal &&
             tiny.candidates[6].runtime_validated &&
             tiny.candidates[6].reason ==
                 "parallel candidate requires at least two output macro-tiles and workers",
         "tiny problem cannot select parallel execution without erasing exact "
         "runtime evidence");

  const auto low_work = planner::plan_cpu_gemm_v3(
      problem(256, 32, 32), avx2_capabilities(), topology(), policy(2),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  expect(low_work.status == planner::CpuPlanStatusV1::forced_variant_illegal &&
             low_work.candidates[6].reason ==
                 "parallel work per thread is below the deterministic threshold",
         "parallel coordination is rejected below static work threshold");

  auto incomplete = topology();
  incomplete.discovery_complete = false;
  const auto no_topology = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), incomplete, policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  expect(no_topology.candidates[6].reason ==
             "complete CPU topology discovery is required for parallel planning",
         "parallel planning fails closed on incomplete topology");

  auto nested_policy = policy(8);
  nested_policy.external_provider_parallelism_active = true;
  const auto nested = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), topology(),
      nested_policy, resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  expect(nested.candidates[6].reason ==
             "native/provider nested parallelism is prohibited",
         "planner prevents native/OpenBLAS oversubscription");

  auto no_context = resources();
  no_context.execution_context_available = false;
  const auto unavailable = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), topology(), policy(8),
      no_context,
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node_placement());
  expect(unavailable.candidates[6].reason ==
             "persistent CPU execution context is unavailable",
         "parallel implementation requires persistent context");

  auto bad_policy = policy(0);
  const auto invalid = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), topology(), bad_policy,
      resources(), planner::CpuGemmRequestV3::automatic);
  expect(invalid.status == planner::CpuPlanStatusV1::invalid_capabilities &&
             invalid.selection_reason ==
                 "requested thread count must be positive",
         "invalid thread policy fails before candidate selection");
}

void bound_context_crossover_calibration() {
  const auto bound_policy = policy(4, 0, false, true);
  const auto placement = single_node_placement(4, 4);

  const auto below_band = planner::plan_cpu_gemm_v3(
      problem(16, 16, 8), avx2_capabilities(), topology(), bound_policy,
      resources(), planner::CpuGemmRequestV3::automatic, placement);
  const auto lower_boundary = planner::plan_cpu_gemm_v3(
      problem(16, 16, 16), avx2_capabilities(), topology(), bound_policy,
      resources(), planner::CpuGemmRequestV3::automatic, placement);
  const auto tail_shape = planner::plan_cpu_gemm_v3(
      problem(64, 7, 19), avx2_capabilities(), topology(), bound_policy,
      resources(), planner::CpuGemmRequestV3::automatic, placement);
  expect(below_band.status == planner::CpuPlanStatusV1::selected &&
             below_band.selected_id == "cpu.tiled.f32.v1" &&
             lower_boundary.selected_id ==
                 "cpu.native-packed.avx2-fma.f32.v1" &&
             tail_shape.selected_id ==
                 "cpu.native-packed.avx2-fma.f32.v1",
         "bound-context calibration is limited to the measured small-work band");

  const auto unbound = planner::plan_cpu_gemm_v3(
      problem(16, 16, 16), avx2_capabilities(), topology(), policy(4),
      resources(), planner::CpuGemmRequestV3::automatic,
      single_node_placement());
  const auto bound_single = planner::plan_cpu_gemm_v3(
      problem(16, 16, 16), avx2_capabilities(), topology(),
      policy(1, 0, false, true), resources(),
      planner::CpuGemmRequestV3::automatic,
      single_node_placement(1, 1));
  expect(unbound.selected_id == "cpu.external.openblas.f32.v1" &&
             bound_single.selected_id == "cpu.external.openblas.f32.v1",
         "bound-context calibration does not override provider-permitted policies");
}

void numa_evidence_and_cost() {
  planner::CpuPlannerTopologyViewV1 two_node = topology();
  two_node.logical_processors = 16;
  two_node.physical_cores = 8;
  two_node.available_processors = 16;
  two_node.numa_nodes = 2;
  two_node.numa_node_ids[1] = 1;

  auto local = single_node_placement(8, 8);
  local.numa = planner::CpuPlannerNumaPolicyV1::local_first;
  local.affinity = platform::CpuAffinityPolicyV1::local_first;

  auto crossing = local;
  crossing.local_logical_processor_capacity = 4;
  crossing.local_physical_core_capacity = 4;
  crossing.selected_numa_node_count = 2;
  crossing.selected_numa_nodes[1] = 1;
  crossing.crosses_numa_nodes = true;

  const auto local_plan = planner::plan_cpu_gemm_v3(
      problem(2048, 512, 512), avx2_capabilities(), two_node, policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma, local);
  const auto cross_plan = planner::plan_cpu_gemm_v3(
      problem(2048, 512, 512), avx2_capabilities(), two_node, policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma, crossing);
  expect(local_plan.status == planner::CpuPlanStatusV1::selected &&
             cross_plan.status == planner::CpuPlanStatusV1::selected &&
             !local_plan.candidates[6].crosses_numa_nodes &&
             cross_plan.candidates[6].crosses_numa_nodes &&
             cross_plan.candidates[6].estimated_cost >
                 local_plan.candidates[6].estimated_cost,
         "cross-NUMA execution receives explicit conservative cost overhead");

  auto inconsistent = crossing;
  inconsistent.caller_first_touch_required = false;
  const auto rejected = planner::plan_cpu_gemm_v3(
      problem(2048, 512, 512), avx2_capabilities(), two_node, policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      inconsistent);
  expect(rejected.candidates[6].reason ==
             "cross-NUMA placement must expose caller-owned first touch",
         "cross-NUMA placement fails closed without first-touch ownership evidence");

  auto absent_node = local;
  absent_node.selected_numa_nodes[0] = 99;
  const auto absent_node_plan = planner::plan_cpu_gemm_v3(
      problem(2048, 512, 512), avx2_capabilities(), two_node, policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      absent_node);
  expect(absent_node_plan.candidates[6].reason ==
             "CPU placement references a NUMA node absent from topology",
         "placement evidence cannot name a NUMA node absent from topology");

  auto single_node = single_node_placement(8, 4);
  const auto bounded = planner::plan_cpu_gemm_v3(
      problem(2048, 512, 512), avx2_capabilities(), two_node, policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma,
      single_node);
  expect(bounded.status == planner::CpuPlanStatusV1::selected &&
             bounded.candidates[6].actual_threads == 4,
         "single-node policy clamps workers to local physical capacity");
}

void avx512_fail_closed_and_determinism() {
  const auto large = problem(1024, 1024, 1024);
  const auto unavailable = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), resources(),
      planner::CpuGemmRequestV3::force_native_packed_avx512_fma);
  expect(unavailable.status ==
             planner::CpuPlanStatusV1::forced_variant_illegal &&
             unavailable.candidates[5].reason ==
                 "AVX-512F hardware support is unavailable",
         "AVX-512 candidate fails closed without normalized capability facts");

  auto avx512 = resources();
  avx512.avx512f_hardware = true;
  avx512.avx512f_os_enabled = true;
  avx512.avx512f_compiler_supported = true;
  avx512.native_packed_avx512_fma_compiled = true;
  avx512.native_packed_avx512_fma_runtime_validated = true;
  avx512.native_packed_avx512_workspace_size_valid = true;
  avx512.native_packed_avx512_workspace_bytes = 393216;
  avx512.native_packed_avx512_workspace_alignment = 64;
  avx512.native_parallel_avx512_fma_compiled = true;
  avx512.native_parallel_avx512_fma_runtime_validated = true;
  avx512.native_parallel_avx512_workspace_size_valid = true;
  avx512.native_parallel_avx512_shared_workspace_bytes = 262144;
  avx512.native_parallel_avx512_per_worker_workspace_bytes = 131072;
  avx512.native_parallel_avx512_workspace_alignment = 64;
  const auto packed = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), avx512,
      planner::CpuGemmRequestV3::force_native_packed_avx512_fma);
  const auto parallel = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), avx512,
      planner::CpuGemmRequestV3::force_native_parallel_avx512_fma,
      single_node_placement());
  expect(packed.status == planner::CpuPlanStatusV1::selected &&
             packed.candidates[5].actual_threads == 1 &&
             parallel.status == planner::CpuPlanStatusV1::selected &&
             parallel.candidates[7].actual_threads == 8,
         "synthetic fully validated AVX-512 facts make exact variants legal");

  auto invalid_workspace = avx512;
  invalid_workspace.native_packed_avx512_workspace_size_valid = false;
  const auto validated_but_illegal = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), invalid_workspace,
      planner::CpuGemmRequestV3::force_native_packed_avx512_fma);
  expect(validated_but_illegal.status ==
                 planner::CpuPlanStatusV1::forced_variant_illegal &&
             validated_but_illegal.candidates[5].runtime_validated &&
             validated_but_illegal.candidates[5].reason ==
                 "native packed AVX-512 workspace requirement overflowed",
         "contextual AVX-512 illegality preserves exact runtime evidence");

  const auto first = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), resources(),
      planner::CpuGemmRequestV3::automatic, single_node_placement());
  const auto second = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), resources(),
      planner::CpuGemmRequestV3::automatic, single_node_placement());
  const std::size_t required =
      planner::format_cpu_gemm_plan_v3(first, nullptr, 0);
  std::string first_text(required + 1, '\0');
  std::string second_text(required + 1, '\0');
  planner::format_cpu_gemm_plan_v3(first, first_text.data(), first_text.size());
  planner::format_cpu_gemm_plan_v3(second, second_text.data(),
                                   second_text.size());
  first_text.resize(required);
  second_text.resize(required);
  expect(first.status == planner::CpuPlanStatusV1::selected &&
             first.selected_id == second.selected_id &&
             first_text == second_text &&
             first_text.find("cpu.native-parallel.avx2-fma.f32.v1") !=
                 std::string::npos &&
             first_text.find("threads=8") != std::string::npos &&
             first_text.find("priority=") != std::string::npos &&
             first_text.find("required-hardware=") != std::string::npos &&
             first_text.find("required-os=") != std::string::npos &&
             first_text.find("required-compiler=") != std::string::npos &&
             first_text.find("required-implementation=") !=
                 std::string::npos &&
             first_text.find("placement-complete=true") != std::string::npos,
         "automatic planner and full candidate diagnostic are deterministic");
}

void versioned_capability_and_topology_projection() {
  const auto capabilities = synthetic_capabilities_v2();
  const auto machine_topology = synthetic_topology_v1();
  const auto capability =
      planner::project_cpu_capabilities_v2_for_planner_v1(capabilities);
  const auto restricted =
      platform::restrict_cpu_topology_v1(machine_topology, {0, 1, 2});
  const auto topology_view = planner::project_cpu_topology_v1_for_planner_v1(
      restricted.topology);
  expect(capability.valid && capability.avx512f_hardware &&
             capability.avx512f_os_enabled &&
             capability.avx512f_compiler_supported &&
             capability.avx512f_implementation_available &&
             capability.avx512f_runtime_validated &&
             capability.avx2_implementation_available &&
             capability.avx2_runtime_validated &&
             capability.fma_implementation_available &&
             capability.fma_runtime_validated &&
             capability.baseline.usable_vector_bits == 512,
         "capability v2 domains project without losing AVX-512 legality facts");
  expect(restricted && topology_view.discovery_complete &&
             topology_view.logical_processors == 3 &&
             topology_view.physical_cores == 3 &&
             topology_view.available_processors == 3 &&
             topology_view.numa_nodes == 1,
         "topology v1 projects deterministic counts and affinity ceiling");
  const auto count_only = planner::project_cpu_topology_v1_for_planner_v1(
      machine_topology, 3);
  expect(!count_only.discovery_complete,
         "count-only affinity override cannot authorize parallel placement");

  auto available = resources();
  available.native_packed_avx512_fma_compiled = true;
  available.native_packed_avx512_fma_runtime_validated = true;
  available.native_packed_avx512_workspace_size_valid = true;
  available.native_packed_avx512_workspace_bytes = 393216;
  available.native_packed_avx512_workspace_alignment = 64;
  available.native_parallel_avx512_fma_compiled = true;
  available.native_parallel_avx512_fma_runtime_validated = true;
  available.native_parallel_avx512_workspace_size_valid = true;
  available.native_parallel_avx512_shared_workspace_bytes = 262144;
  available.native_parallel_avx512_per_worker_workspace_bytes = 131072;
  available.native_parallel_avx512_workspace_alignment = 64;
  const auto projected = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), capabilities, restricted.topology,
      policy(8, 0, true), available,
      planner::CpuGemmRequestV3::force_native_parallel_avx512_fma, 0,
      single_node_placement(3, 3));
  expect(projected.status == planner::CpuPlanStatusV1::selected &&
             projected.candidates[7].actual_threads == 3,
         "planner consumes versioned capability/topology records and affinity limit");

  auto unvalidated = capabilities;
  unvalidated.runtime_validation.available &=
      ~(platform::feature_bit(platform::CpuFeatureV2::avx512f) |
        platform::feature_bit(platform::CpuFeatureV2::avx512dq) |
        platform::feature_bit(platform::CpuFeatureV2::avx512bw) |
        platform::feature_bit(platform::CpuFeatureV2::avx512vl));
  unvalidated.usable_vector_bits = 512;
  const auto rejected = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), unvalidated, machine_topology,
      policy(4), available,
      planner::CpuGemmRequestV3::force_native_packed_avx512_fma);
  expect(rejected.status == planner::CpuPlanStatusV1::forced_variant_illegal &&
             rejected.candidates[5].reason ==
                 "native packed AVX-512 implementation is not runtime-validated",
         "capability projection cannot be bypassed by optimistic resources");
}

}  // namespace

int main() {
  registry_and_parallel_policy();
  parallel_rejection_boundaries();
  bound_context_crossover_calibration();
  numa_evidence_and_cost();
  avx512_fail_closed_and_determinism();
  versioned_capability_and_topology_projection();
  if (failures != 0) {
    std::cerr << failures << " planner v3 checks failed\n";
    return 1;
  }
  std::cout << "CPU planner v3 PASS\n";
  return 0;
}
