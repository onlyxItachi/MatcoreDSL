#ifndef MATCORE_MDSLC_PLANNER_CPU_PLANNER_V3_H
#define MATCORE_MDSLC_PLANNER_CPU_PLANNER_V3_H

#include "cpu_planner_v2.h"
#include "cpu_capability_v2.h"
#include "cpu_topology_v1.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace matcore::mdslc::planner {

inline constexpr std::uint32_t kCpuPlannerVersionV3 = 3;
inline constexpr std::uint32_t kCpuPlannerTopologyViewVersionV1 = 1;
inline constexpr std::uint32_t kCpuThreadPolicyVersionV1 = 1;
inline constexpr std::uint32_t kCpuPlannerPlacementEvidenceVersionV1 = 1;
inline constexpr std::size_t kCpuGemmCandidateCountV3 = 8;
inline constexpr std::size_t kCpuPlannerReportedNumaNodeLimitV1 = 16;
inline constexpr std::uint64_t kCpuParallelMinimumWorkPerThreadV1 =
    UINT64_C(1) << 20;
inline constexpr std::uint64_t
    kCpuParallelColumnOnlyMinimumWorkPerThreadV1 = UINT64_C(1) << 23;
inline constexpr std::uint64_t
    kCpuParallelTwoDimensionalMinimumWorkPerThreadV1 = UINT64_C(1) << 25;
inline constexpr std::uint64_t kCpuParallelMacroTileRowsV1 = 128;
inline constexpr std::uint64_t kCpuParallelRegisterTileRowsV1 = 4;
inline constexpr std::uint64_t kCpuParallelCacheLineFloatCountV1 = 16;
inline constexpr std::uint64_t kCpuParallelColumnTileColumnsV1 = 256;

enum class CpuGemmVariantV3 : std::uint8_t {
  reference = 0,
  tiled = 1,
  compiler_vectorized = 2,
  external_openblas = 3,
  native_packed_avx2_fma = 4,
  native_packed_avx512_fma = 5,
  native_parallel_avx2_fma = 6,
  native_parallel_avx512_fma = 7,
};

enum class CpuGemmRequestV3 : std::uint8_t {
  automatic = 0,
  force_reference = 1,
  force_tiled = 2,
  force_compiler_vectorized = 3,
  force_external_openblas = 4,
  force_native_packed_avx2_fma = 5,
  force_native_packed_avx512_fma = 6,
  force_native_parallel_avx2_fma = 7,
  force_native_parallel_avx512_fma = 8,
};

// A lossless planner-facing projection of the versioned topology record. The
// platform layer owns discovery; the planner consumes only injected facts and
// performs no sysfs or operating-system calls.
struct CpuPlannerTopologyViewV1 {
  std::uint32_t version = kCpuPlannerTopologyViewVersionV1;
  bool discovery_complete = false;
  std::uint32_t logical_processors = 0;
  std::uint32_t physical_cores = 0;
  std::uint32_t available_processors = 0;
  std::uint32_t numa_nodes = 0;
  bool numa_node_ids_complete = false;
  std::array<std::uint32_t, kCpuPlannerReportedNumaNodeLimitV1>
      numa_node_ids{};
};

struct CpuThreadPolicyV1 {
  std::uint32_t version = kCpuThreadPolicyVersionV1;
  std::uint32_t requested_threads = 1;
  std::uint32_t maximum_threads = 0;
  bool allow_smt = false;
  bool external_provider_parallelism_active = false;
  // Provider-owned multithreading is not executable while work is constrained
  // to runtime-owned bound workers. Single-thread provider execution remains
  // legal because it can run on one bound worker.
  bool worker_affinity_active = false;
};

/*
 * Pure planner/runtime contract for deterministic disjoint output tasking.
 * The runtime never splits K, so no reduction or output synchronization is
 * required. Row boundaries are aligned to both the packed microkernel's MR
 * and a complete 64-byte output-cacheline cycle. Cacheline-safe problems may
 * additionally expose NC-sized column panels when measured work floors make
 * that decomposition worthwhile.
 */
struct CpuParallelTaskPlanV1 {
  std::uint64_t macro_tile_count = 0;
  std::uint64_t row_quantum = 0;
  std::uint64_t row_group_count = 0;
  std::uint64_t row_task_count = 0;
  std::uint64_t column_panel_count = 0;
  std::uint64_t column_task_count = 0;
  std::uint64_t task_count = 0;
  std::uint32_t actual_threads = 0;
};

enum class CpuPlannerNumaPolicyV1 : std::uint8_t {
  single_node = 0,
  local_first = 1,
};

/*
 * Placement is injected evidence, never inferred by the planner.  A complete
 * record describes the worker context that will execute the plan.  The two
 * local capacities are measured after any scheduler/cpuset restriction and
 * distinguish physical-core and SMT-enabled ceilings.  Selected NUMA IDs are
 * fixed-size to keep planning allocation-free and noexcept.
 */
struct CpuPlannerPlacementEvidenceV1 {
  std::uint32_t version = kCpuPlannerPlacementEvidenceVersionV1;
  bool evidence_complete = false;
  bool affinity_requested = false;
  bool affinity_applied = false;
  platform::CpuAffinityPolicyV1 affinity =
      platform::CpuAffinityPolicyV1::compact;
  CpuPlannerNumaPolicyV1 numa = CpuPlannerNumaPolicyV1::single_node;
  std::uint32_t local_logical_processor_capacity = 0;
  std::uint32_t local_physical_core_capacity = 0;
  std::uint32_t selected_numa_node_count = 0;
  std::array<std::uint32_t, kCpuPlannerReportedNumaNodeLimitV1>
      selected_numa_nodes{};
  bool crosses_numa_nodes = false;
  bool caller_first_touch_required = false;
};

struct CpuGemmImplementationResourcesV2 {
  CpuGemmImplementationResourcesV1 baseline;
  bool reference_f32_runtime_validated = false;
  bool tiled_f32_runtime_validated = false;
  bool compiler_vectorized_f32_runtime_validated = false;
  bool external_openblas_f32_runtime_validated = false;
  bool native_packed_avx2_fma_runtime_validated = false;
  bool execution_context_available = false;
  std::uint32_t execution_context_worker_capacity = 0;
  bool native_parallel_avx2_fma_compiled = false;
  bool native_parallel_avx2_fma_runtime_validated = false;
  bool native_parallel_avx2_workspace_size_valid = false;
  std::uint64_t native_parallel_avx2_shared_workspace_bytes = 0;
  std::uint64_t native_parallel_avx2_per_worker_workspace_bytes = 0;
  std::uint32_t native_parallel_avx2_workspace_alignment = 0;

  bool native_packed_avx512_fma_compiled = false;
  bool native_packed_avx512_fma_runtime_validated = false;
  bool native_packed_avx512_workspace_size_valid = false;
  std::uint64_t native_packed_avx512_workspace_bytes = 0;
  std::uint32_t native_packed_avx512_workspace_alignment = 0;
  bool native_parallel_avx512_fma_compiled = false;
  bool native_parallel_avx512_fma_runtime_validated = false;
  bool native_parallel_avx512_workspace_size_valid = false;
  std::uint64_t native_parallel_avx512_shared_workspace_bytes = 0;
  std::uint64_t native_parallel_avx512_per_worker_workspace_bytes = 0;
  std::uint32_t native_parallel_avx512_workspace_alignment = 0;

  // These are normalized from capability v2's hardware, OS-state, compiler,
  // implementation, and runtime-validation domains. All must be true before
  // an AVX-512 body can be legal.
  bool avx512f_hardware = false;
  bool avx512f_os_enabled = false;
  bool avx512f_compiler_supported = false;
};

struct CpuGemmVariantRecordV3 {
  CpuGemmVariantV3 variant;
  std::string_view stable_id;
  std::uint16_t deterministic_priority;
};

inline constexpr std::array<CpuGemmVariantRecordV3,
                            kCpuGemmCandidateCountV3>
    kCpuGemmVariantRegistryV3{{
        {CpuGemmVariantV3::reference, "cpu.reference.f32.v1", 80},
        {CpuGemmVariantV3::tiled, "cpu.tiled.f32.v1", 70},
        {CpuGemmVariantV3::compiler_vectorized,
         "cpu.compiler-vectorized.avx2-fma.f32.v1", 60},
        {CpuGemmVariantV3::external_openblas,
         "cpu.external.openblas.f32.v1", 10},
        {CpuGemmVariantV3::native_packed_avx2_fma,
         "cpu.native-packed.avx2-fma.f32.v1", 50},
        {CpuGemmVariantV3::native_packed_avx512_fma,
         "cpu.native-packed.avx512-fma.f32.v1", 40},
        {CpuGemmVariantV3::native_parallel_avx2_fma,
         "cpu.native-parallel.avx2-fma.f32.v1", 30},
        {CpuGemmVariantV3::native_parallel_avx512_fma,
         "cpu.native-parallel.avx512-fma.f32.v1", 20},
    }};

struct CpuCandidateDecisionV3 {
  CpuGemmVariantV3 variant = CpuGemmVariantV3::reference;
  std::string_view stable_id;
  bool legal = false;
  std::string_view reason;
  std::uint64_t estimated_cost = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t required_workspace_bytes = 0;
  std::uint32_t required_workspace_alignment = 1;
  std::uint32_t actual_threads = 0;
  std::uint16_t deterministic_priority = 0;
  std::uint64_t shared_workspace_bytes = 0;
  std::uint64_t per_worker_workspace_bytes = 0;
  std::uint64_t required_hardware_features = 0;
  std::uint64_t required_os_features = 0;
  std::uint64_t required_compiler_features = 0;
  std::uint64_t required_implementation_features = 0;
  bool runtime_validated = false;
  bool crosses_numa_nodes = false;
};

struct CpuGemmPlanV3 {
  std::uint32_t planner_version = kCpuPlannerVersionV3;
  CpuPlanStatusV1 status = CpuPlanStatusV1::no_legal_variant;
  CpuGemmProblemV1 problem;
  CpuCapabilitiesV1 baseline_capabilities;
  CpuPlannerTopologyViewV1 topology;
  CpuThreadPolicyV1 thread_policy;
  CpuPlannerPlacementEvidenceV1 placement;
  CpuGemmImplementationResourcesV2 resources;
  CpuGemmRequestV3 request = CpuGemmRequestV3::automatic;
  std::array<CpuCandidateDecisionV3, kCpuGemmCandidateCountV3> candidates{};
  CpuGemmVariantV3 selected_variant = CpuGemmVariantV3::reference;
  std::string_view selected_id;
  std::string_view selection_reason;
};

struct CpuPlannerCapabilityProjectionV1 {
  bool valid = false;
  std::string_view reason;
  CpuCapabilitiesV1 baseline;
  bool avx512f_hardware = false;
  bool avx512f_os_enabled = false;
  bool avx512f_compiler_supported = false;
  bool avx512f_implementation_available = false;
  bool avx512f_runtime_validated = false;
  bool avx2_implementation_available = false;
  bool avx2_runtime_validated = false;
  bool fma_implementation_available = false;
  bool fma_runtime_validated = false;
};

CpuPlannerCapabilityProjectionV1 project_cpu_capabilities_v2_for_planner_v1(
    const platform::CpuCapabilitiesV2 &capabilities) noexcept;

CpuPlannerTopologyViewV1 project_cpu_topology_v1_for_planner_v1(
    const platform::CpuTopologyV1 &topology,
    // A count-only restriction cannot establish physical-core or NUMA
    // membership and therefore makes parallel topology evidence incomplete.
    // Call restrict_cpu_topology_v1 first for parallel planning.
    std::uint32_t available_processors_override = 0) noexcept;

CpuParallelTaskPlanV1 plan_cpu_parallel_tasks_v1(
    const CpuGemmProblemV1 &problem,
    std::uint32_t requested_threads) noexcept;

const std::array<CpuGemmVariantRecordV3, kCpuGemmCandidateCountV3> &
cpu_gemm_variant_registry_v3() noexcept;

CpuGemmPlanV3 plan_cpu_gemm_v3(
    const CpuGemmProblemV1 &problem,
    const CpuCapabilitiesV1 &baseline_capabilities,
    const CpuPlannerTopologyViewV1 &topology,
    const CpuThreadPolicyV1 &thread_policy,
    const CpuGemmImplementationResourcesV2 &resources,
    CpuGemmRequestV3 request = CpuGemmRequestV3::automatic,
    const CpuPlannerPlacementEvidenceV1 &placement = {}) noexcept;

CpuGemmPlanV3 plan_cpu_gemm_v3(
    const CpuGemmProblemV1 &problem,
    const platform::CpuCapabilitiesV2 &capabilities,
    const platform::CpuTopologyV1 &topology,
    const CpuThreadPolicyV1 &thread_policy,
    const CpuGemmImplementationResourcesV2 &resources,
    CpuGemmRequestV3 request = CpuGemmRequestV3::automatic,
    std::uint32_t available_processors_override = 0,
    const CpuPlannerPlacementEvidenceV1 &placement = {}) noexcept;

std::size_t format_cpu_gemm_plan_v3(const CpuGemmPlanV3 &plan,
                                    char *output,
                                    std::size_t capacity) noexcept;

}  // namespace matcore::mdslc::planner

#endif
