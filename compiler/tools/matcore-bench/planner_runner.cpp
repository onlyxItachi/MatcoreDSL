#include "benchmark.h"

#include "cpu_backend_registry.h"
#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"
#include "cpu_planner_v2.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>

namespace matcore::mdslc::bench {
namespace {

namespace planner = matcore::mdslc::planner;
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
                 planner::CpuGemmRequestV2 &request) noexcept {
  if (stable_id == "auto") {
    request = planner::CpuGemmRequestV2::automatic;
  } else if (stable_id == "cpu.reference.f32.v1") {
    request = planner::CpuGemmRequestV2::force_reference;
  } else if (stable_id == "cpu.tiled.f32.v1") {
    request = planner::CpuGemmRequestV2::force_tiled;
  } else if (stable_id == "cpu.compiler-vectorized.avx2-fma.f32.v1") {
    request = planner::CpuGemmRequestV2::force_compiler_vectorized;
  } else if (stable_id == "cpu.external.openblas.f32.v1") {
    request = planner::CpuGemmRequestV2::force_external_openblas;
  } else if (stable_id == "cpu.native-packed.avx2-fma.f32.v1") {
    request = planner::CpuGemmRequestV2::force_native_packed_avx2_fma;
  } else {
    return false;
  }
  return true;
}

planner::CpuGemmRequestV1 legacy_request(
    planner::CpuGemmVariantV2 variant) noexcept {
  switch (variant) {
    case planner::CpuGemmVariantV2::reference:
      return planner::CpuGemmRequestV1::force_reference;
    case planner::CpuGemmVariantV2::tiled:
      return planner::CpuGemmRequestV1::force_tiled;
    case planner::CpuGemmVariantV2::compiler_vectorized:
      return planner::CpuGemmRequestV1::force_compiler_vectorized;
    case planner::CpuGemmVariantV2::external_openblas:
    case planner::CpuGemmVariantV2::native_packed_avx2_fma:
      return planner::CpuGemmRequestV1::force_reference;
  }
  return planner::CpuGemmRequestV1::force_reference;
}

struct BackendPlanState final : RunnerPlanStateV1 {
  BackendPlanState(planner::CpuGemmPlanV2 value, PackingModeV1 packing)
      : plan(std::move(value)), packing_mode(packing) {}
  planner::CpuGemmPlanV2 plan;
  PackingModeV1 packing_mode = PackingModeV1::include;
  mutable runtime::CpuPackedBViewV1 packed_b;
  mutable bool packed_b_ready = false;
};

class PlannerRunner final : public GemmRunnerV1 {
 public:
  PlannerRunner() : capabilities_(planner::discover_cpu_capabilities_v1()) {}

  RunnerEnvironmentV1 environment() const override {
    const auto probe_problem = problem({1, 1, 1}, alignof(float));
    const auto resources =
        runtime::discover_cpu_gemm_implementation_resources_v1(probe_problem,
                                                                1);
    const auto probe =
        planner::plan_cpu_gemm_v2(probe_problem, capabilities_, resources);
    std::array<char, 4096> diagnostic{};
    planner::format_cpu_gemm_plan_v2(probe, diagnostic.data(),
                                     diagnostic.size());
    const auto provider = runtime::openblas_provider_info_v1();
    return {diagnostic.data(), provider.linked ? "OpenBLAS" : "unavailable",
            provider.package_version, provider.runtime_config};
  }

  std::vector<std::string> variant_ids() const override {
    std::vector<std::string> result;
    for (const auto &record : planner::kCpuGemmVariantRegistryV2)
      result.emplace_back(record.stable_id);
    return result;
  }

  RunnerPlanV1 plan(const GemmShapeV1 &shape,
                    std::uint32_t minimum_alignment,
                    std::uint32_t requested_threads,
                    std::string_view requested_variant,
                    PackingModeV1 packing_mode) const override {
    RunnerPlanV1 result;
    planner::CpuGemmRequestV2 request{};
    if (!request_for(requested_variant, request)) {
      result.reason = "requested stable variant ID is not registered";
      return result;
    }
    const auto gemm_problem = problem(shape, minimum_alignment);
    const auto resources =
        runtime::discover_cpu_gemm_implementation_resources_v1(
            gemm_problem, requested_threads);
    auto selected = planner::plan_cpu_gemm_v2(
        gemm_problem, capabilities_, resources, request);
    result.legal = selected.status == planner::CpuPlanStatusV1::selected;
    result.selected_variant = std::string(selected.selected_id);
    result.reason = std::string(selected.selection_reason);
    if (result.legal &&
        selected.selected_variant ==
            planner::CpuGemmVariantV2::native_packed_avx2_fma &&
        packing_mode == PackingModeV1::exclude) {
      result.legal = false;
      result.reason =
          "native packed compute-only timing is unavailable because A packing remains required";
    }

    if (result.legal) {
      const std::size_t selected_index =
          static_cast<std::size_t>(selected.selected_variant);
      const auto &candidate = selected.candidates[selected_index];
      result.actual_threads = candidate.actual_threads;
      result.workspace_bytes = candidate.required_workspace_bytes;
      result.workspace_alignment = candidate.required_workspace_alignment;
      result.packing_required =
          selected.selected_variant ==
          planner::CpuGemmVariantV2::native_packed_avx2_fma;
      result.supports_prepacked_b = result.packing_required;
      if (result.packing_required && packing_mode == PackingModeV1::prepack_b) {
        runtime::CpuPackedGemmWorkspaceRequirementsV1 packed_requirements;
        const auto packed_status =
            runtime::cpu_packed_avx2_prepacked_b_requirements_v1(
                gemm_problem, &packed_requirements);
        runtime::CpuPackedGemmWorkspaceRequirementsV1 execute_requirements;
        const auto execute_status =
            runtime::cpu_packed_avx2_workspace_requirements_v1(
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
          result.workspace_alignment = static_cast<std::uint32_t>(
              execute_requirements.alignment_bytes);
          selected.candidates[selected_index].required_workspace_bytes =
              execute_requirements.total_bytes;
          selected.candidates[selected_index].required_workspace_alignment =
              static_cast<std::uint32_t>(execute_requirements.alignment_bytes);
        }
      }
      if (result.legal)
        result.state =
            std::make_shared<BackendPlanState>(selected, packing_mode);
    }
    std::array<char, 8192> diagnostic{};
    planner::format_cpu_gemm_plan_v2(selected, diagnostic.data(),
                                     diagnostic.size());
    result.diagnostic = diagnostic.data();
    if (!result.legal && result.reason.empty())
      result.reason = "planner found no legal variant";
    return result;
  }

  bool prepare(const RunnerPlanV1 &plan, const GemmShapeV1 &shape,
               const float *, const float *rhs,
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
    if (!prepack_b) return true;
    if (state->plan.selected_variant !=
        planner::CpuGemmVariantV2::native_packed_avx2_fma) {
      error = "selected implementation does not support prepacked-B";
      return false;
    }
    const auto packed_status = runtime::cpu_prepare_packed_b_avx2_v1(
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
      case planner::CpuGemmVariantV2::reference:
      case planner::CpuGemmVariantV2::tiled:
      case planner::CpuGemmVariantV2::compiler_vectorized: {
        const auto legacy = planner::plan_cpu_gemm_v1(
            state->plan.problem, capabilities_,
            legacy_request(state->plan.selected_variant));
        if (!planner::execute_cpu_gemm_plan_v1(legacy, lhs, rhs, output)) {
          error = "selected legacy implementation failed";
          return false;
        }
        return true;
      }
      case planner::CpuGemmVariantV2::external_openblas: {
        std::uint32_t actual_threads = 0;
        const auto provider_status = runtime::execute_openblas_gemm_f32_v1(
            state->plan.problem, lhs, rhs, output,
            state->plan.resources.requested_threads, &actual_threads);
        if (provider_status != runtime::OpenBlasExecutionStatusV1::success ||
            actual_threads != plan.actual_threads) {
          error = "OpenBLAS failed or did not honor the planned thread count";
          return false;
        }
        return true;
      }
      case planner::CpuGemmVariantV2::native_packed_avx2_fma: {
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

  planner::CpuCapabilitiesV1 capabilities_;
};

}  // namespace

std::unique_ptr<GemmRunnerV1> make_planner_runner_v1() {
  return std::make_unique<PlannerRunner>();
}

}  // namespace matcore::mdslc::bench
