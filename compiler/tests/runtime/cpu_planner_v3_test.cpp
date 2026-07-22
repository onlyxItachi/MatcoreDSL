#include "cpu_planner_v3.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

namespace {

namespace planner = matcore::mdslc::planner;

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
  return {planner::kCpuPlannerTopologyViewVersionV1, true, 24, 12, 24, 1};
}

planner::CpuThreadPolicyV1 policy(std::uint32_t requested,
                                  std::uint32_t maximum = 0,
                                  bool allow_smt = false) {
  return {planner::kCpuThreadPolicyVersionV1, requested, maximum, allow_smt,
          false};
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
  result.execution_context_available = true;
  result.execution_context_worker_capacity = 24;
  result.native_parallel_avx2_fma_compiled = true;
  result.native_parallel_avx2_workspace_size_valid = true;
  result.native_parallel_avx2_shared_workspace_bytes = 262144;
  result.native_parallel_avx2_per_worker_workspace_bytes = 131072;
  result.native_parallel_avx2_workspace_alignment = 64;
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
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
  const auto &parallel = forced.candidates[6];
  expect(forced.status == planner::CpuPlanStatusV1::selected &&
             forced.selected_id ==
                 "cpu.native-parallel.avx2-fma.f32.v1" &&
             parallel.legal && parallel.actual_threads == 8 &&
             parallel.required_workspace_bytes ==
                 UINT64_C(262144) + UINT64_C(8) * UINT64_C(131072) &&
             parallel.required_workspace_alignment == 64,
         "forced parallel AVX2 plan reports exact shared and worker workspace");

  const auto capped = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(16, 4), resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
  expect(capped.status == planner::CpuPlanStatusV1::selected &&
             capped.candidates[6].actual_threads == 4,
         "explicit maximum thread policy caps native workers");

  const auto many_tiles = problem(4096, 256, 256);
  const auto physical = planner::plan_cpu_gemm_v3(
      many_tiles, avx2_capabilities(), topology(), policy(24), resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
  const auto smt = planner::plan_cpu_gemm_v3(
      many_tiles, avx2_capabilities(), topology(), policy(24, 0, true),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
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
}

void parallel_rejection_boundaries() {
  const auto tiny = planner::plan_cpu_gemm_v3(
      problem(32, 32, 32), avx2_capabilities(), topology(), policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
  expect(tiny.status == planner::CpuPlanStatusV1::forced_variant_illegal &&
             tiny.candidates[6].reason ==
                 "parallel candidate requires at least two output macro-tiles and workers",
         "tiny problem cannot select parallel execution");

  const auto low_work = planner::plan_cpu_gemm_v3(
      problem(256, 32, 32), avx2_capabilities(), topology(), policy(2),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
  expect(low_work.status == planner::CpuPlanStatusV1::forced_variant_illegal &&
             low_work.candidates[6].reason ==
                 "parallel work per thread is below the deterministic threshold",
         "parallel coordination is rejected below static work threshold");

  auto incomplete = topology();
  incomplete.discovery_complete = false;
  const auto no_topology = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), incomplete, policy(8),
      resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
  expect(no_topology.candidates[6].reason ==
             "complete CPU topology discovery is required for parallel planning",
         "parallel planning fails closed on incomplete topology");

  auto nested_policy = policy(8);
  nested_policy.external_provider_parallelism_active = true;
  const auto nested = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), topology(),
      nested_policy, resources(),
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
  expect(nested.candidates[6].reason ==
             "native/provider nested parallelism is prohibited",
         "planner prevents native/OpenBLAS oversubscription");

  auto no_context = resources();
  no_context.execution_context_available = false;
  const auto unavailable = planner::plan_cpu_gemm_v3(
      problem(1024, 1024, 1024), avx2_capabilities(), topology(), policy(8),
      no_context,
      planner::CpuGemmRequestV3::force_native_parallel_avx2_fma);
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
  avx512.native_parallel_avx512_workspace_size_valid = true;
  avx512.native_parallel_avx512_shared_workspace_bytes = 262144;
  avx512.native_parallel_avx512_per_worker_workspace_bytes = 131072;
  avx512.native_parallel_avx512_workspace_alignment = 64;
  const auto packed = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), avx512,
      planner::CpuGemmRequestV3::force_native_packed_avx512_fma);
  const auto parallel = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), avx512,
      planner::CpuGemmRequestV3::force_native_parallel_avx512_fma);
  expect(packed.status == planner::CpuPlanStatusV1::selected &&
             packed.candidates[5].actual_threads == 1 &&
             parallel.status == planner::CpuPlanStatusV1::selected &&
             parallel.candidates[7].actual_threads == 8,
         "synthetic fully validated AVX-512 facts make exact variants legal");

  const auto first = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), resources(),
      planner::CpuGemmRequestV3::automatic);
  const auto second = planner::plan_cpu_gemm_v3(
      large, avx2_capabilities(), topology(), policy(8), resources(),
      planner::CpuGemmRequestV3::automatic);
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
             first_text.find("threads=8") != std::string::npos,
         "automatic planner and full candidate diagnostic are deterministic");
}

}  // namespace

int main() {
  registry_and_parallel_policy();
  parallel_rejection_boundaries();
  avx512_fail_closed_and_determinism();
  if (failures != 0) {
    std::cerr << failures << " planner v3 checks failed\n";
    return 1;
  }
  std::cout << "CPU planner v3 PASS\n";
  return 0;
}
