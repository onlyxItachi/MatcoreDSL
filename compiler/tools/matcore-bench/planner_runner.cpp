#include "benchmark.h"

#include "cpu_planner.h"

#include <array>
#include <string>
#include <utility>

namespace matcore::mdslc::bench {
namespace {

namespace planner = matcore::mdslc::planner;

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
                 planner::CpuGemmRequestV1 &request) noexcept {
  if (stable_id == "auto") {
    request = planner::CpuGemmRequestV1::automatic;
  } else if (stable_id == "cpu.reference.f32.v1") {
    request = planner::CpuGemmRequestV1::force_reference;
  } else if (stable_id == "cpu.tiled.f32.v1") {
    request = planner::CpuGemmRequestV1::force_tiled;
  } else if (stable_id == "cpu.compiler-vectorized.avx2-fma.f32.v1") {
    request = planner::CpuGemmRequestV1::force_compiler_vectorized;
  } else {
    return false;
  }
  return true;
}

struct PlannerPlanState final : RunnerPlanStateV1 {
  explicit PlannerPlanState(planner::CpuGemmPlanV1 value)
      : plan(std::move(value)) {}
  planner::CpuGemmPlanV1 plan;
};

class PlannerRunner final : public GemmRunnerV1 {
 public:
  PlannerRunner() : capabilities_(planner::discover_cpu_capabilities_v1()) {}

  RunnerEnvironmentV1 environment() const override {
    const auto probe = planner::plan_cpu_gemm_v1(
        problem({1, 1, 1}, alignof(float)), capabilities_);
    std::array<char, 2048> diagnostic{};
    planner::format_cpu_gemm_plan_v1(probe, diagnostic.data(), diagnostic.size());
    return {diagnostic.data(), "mdslc-native", "planner-v1",
            "reference,tiled,compiler-vectorized"};
  }

  std::vector<std::string> variant_ids() const override {
    std::vector<std::string> result;
    for (const auto &record : planner::cpu_gemm_variant_registry_v1())
      result.emplace_back(record.stable_id);
    return result;
  }

  RunnerPlanV1 plan(const GemmShapeV1 &shape,
                    std::uint32_t minimum_alignment,
                    std::uint32_t requested_threads,
                    std::string_view requested_variant) const override {
    RunnerPlanV1 result;
    if (requested_threads != 1) {
      result.reason = "planner-v1 runner supports exactly one thread";
      return result;
    }
    planner::CpuGemmRequestV1 request{};
    if (!request_for(requested_variant, request)) {
      result.reason = "requested stable variant ID is not registered";
      return result;
    }
    const auto selected = planner::plan_cpu_gemm_v1(
        problem(shape, minimum_alignment), capabilities_, request);
    result.legal = selected.status == planner::CpuPlanStatusV1::selected;
    result.selected_variant = std::string(selected.selected_id);
    result.reason = std::string(selected.selection_reason);
    std::array<char, 4096> diagnostic{};
    planner::format_cpu_gemm_plan_v1(selected, diagnostic.data(),
                                     diagnostic.size());
    result.diagnostic = diagnostic.data();
    result.actual_threads = 1;
    result.workspace_alignment = 1;
    if (result.legal)
      result.state = std::make_shared<PlannerPlanState>(selected);
    if (!result.legal && result.reason.empty())
      result.reason = "planner found no legal variant";
    return result;
  }

  bool prepare(const RunnerPlanV1 &plan, const GemmShapeV1 &,
               const float *, const float *, std::span<std::byte> workspace,
               std::span<std::byte> prepacked_b_storage,
               bool prepack_b, std::string &error) const override {
    if (prepack_b) {
      error = "planner-v1 variants do not support prepacked-B execution";
      return false;
    }
    if (workspace.size() < plan.workspace_bytes) {
      error = "runner workspace is smaller than its declared requirement";
      return false;
    }
    if (prepacked_b_storage.size() < plan.prepacked_b_bytes) {
      error = "prepacked-B storage is smaller than its declared requirement";
      return false;
    }
    return true;
  }

  bool execute(const RunnerPlanV1 &plan, const GemmShapeV1 &shape,
               const float *lhs, const float *rhs, float *output,
               std::span<std::byte> workspace,
               std::span<const std::byte> prepacked_b_storage, bool,
               std::string &error) const override {
    if (workspace.size() < plan.workspace_bytes) {
      error = "runner workspace is smaller than its declared requirement";
      return false;
    }
    if (prepacked_b_storage.size() < plan.prepacked_b_bytes) {
      error = "prepacked-B storage is smaller than its declared requirement";
      return false;
    }
    const auto state = std::dynamic_pointer_cast<const PlannerPlanState>(plan.state);
    if (!state) {
      error = "selected planner state is absent or incompatible";
      return false;
    }
    if (state->plan.problem.m != shape.m || state->plan.problem.n != shape.n ||
        state->plan.problem.k != shape.k ||
        state->plan.selected_id != plan.selected_variant ||
        pointer_alignment(lhs, rhs, output) <
            state->plan.problem.minimum_alignment_bytes) {
      error = "prepared planner state does not match execution buffers";
      return false;
    }
    if (!planner::execute_cpu_gemm_plan_v1(state->plan, lhs, rhs, output)) {
      error = "selected planner-v1 implementation failed";
      return false;
    }
    return true;
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
