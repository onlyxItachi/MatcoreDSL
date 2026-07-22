#ifndef MATCORE_MDSLC_PLANNER_CPU_PLANNER_V3_H
#define MATCORE_MDSLC_PLANNER_CPU_PLANNER_V3_H

#include "cpu_planner_v2.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace matcore::mdslc::planner {

inline constexpr std::uint32_t kCpuPlannerVersionV3 = 3;
inline constexpr std::uint32_t kCpuPlannerTopologyViewVersionV1 = 1;
inline constexpr std::uint32_t kCpuThreadPolicyVersionV1 = 1;
inline constexpr std::size_t kCpuGemmCandidateCountV3 = 8;
inline constexpr std::uint64_t kCpuParallelMinimumWorkPerThreadV1 =
    UINT64_C(1) << 20;

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
};

struct CpuThreadPolicyV1 {
  std::uint32_t version = kCpuThreadPolicyVersionV1;
  std::uint32_t requested_threads = 1;
  std::uint32_t maximum_threads = 0;
  bool allow_smt = false;
  bool external_provider_parallelism_active = false;
};

struct CpuGemmImplementationResourcesV2 {
  CpuGemmImplementationResourcesV1 baseline;
  bool execution_context_available = false;
  std::uint32_t execution_context_worker_capacity = 0;
  bool native_parallel_avx2_fma_compiled = false;
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
};

struct CpuGemmPlanV3 {
  std::uint32_t planner_version = kCpuPlannerVersionV3;
  CpuPlanStatusV1 status = CpuPlanStatusV1::no_legal_variant;
  CpuGemmProblemV1 problem;
  CpuCapabilitiesV1 baseline_capabilities;
  CpuPlannerTopologyViewV1 topology;
  CpuThreadPolicyV1 thread_policy;
  CpuGemmImplementationResourcesV2 resources;
  CpuGemmRequestV3 request = CpuGemmRequestV3::automatic;
  std::array<CpuCandidateDecisionV3, kCpuGemmCandidateCountV3> candidates{};
  CpuGemmVariantV3 selected_variant = CpuGemmVariantV3::reference;
  std::string_view selected_id;
  std::string_view selection_reason;
};

const std::array<CpuGemmVariantRecordV3, kCpuGemmCandidateCountV3> &
cpu_gemm_variant_registry_v3() noexcept;

CpuGemmPlanV3 plan_cpu_gemm_v3(
    const CpuGemmProblemV1 &problem,
    const CpuCapabilitiesV1 &baseline_capabilities,
    const CpuPlannerTopologyViewV1 &topology,
    const CpuThreadPolicyV1 &thread_policy,
    const CpuGemmImplementationResourcesV2 &resources,
    CpuGemmRequestV3 request = CpuGemmRequestV3::automatic) noexcept;

std::size_t format_cpu_gemm_plan_v3(const CpuGemmPlanV3 &plan,
                                    char *output,
                                    std::size_t capacity) noexcept;

}  // namespace matcore::mdslc::planner

#endif
