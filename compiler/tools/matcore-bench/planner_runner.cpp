#include "benchmark.h"

#include "cpu_backend_registry.h"
#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"
#include "cpu_planner_v2.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
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

struct BackendPlanState final : RunnerPlanStateV1 {
  BackendPlanState(planner::CpuGemmPlanV2 value, PackingModeV1 packing,
                   ComputeOnlyPackedLayout compute_layout)
      : plan(std::move(value)),
        packing_mode(packing),
        compute_only_layout(compute_layout) {}
  planner::CpuGemmPlanV2 plan;
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
    if (!result.legal && request != planner::CpuGemmRequestV2::automatic) {
      const std::size_t candidate_index =
          static_cast<std::size_t>(request) - 1U;
      if (candidate_index < selected.candidates.size() &&
          !selected.candidates[candidate_index].reason.empty())
        result.reason = std::string(selected.candidates[candidate_index].reason);
    }
    if (result.legal && packing_mode == PackingModeV1::exclude &&
        selected.selected_variant !=
            planner::CpuGemmVariantV2::native_packed_avx2_fma) {
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
      result.workspace_bytes = candidate.required_workspace_bytes;
      result.workspace_alignment = candidate.required_workspace_alignment;
      result.packing_required =
          selected.selected_variant ==
          planner::CpuGemmVariantV2::native_packed_avx2_fma;
      result.supports_prepacked_b = result.packing_required;
      if (result.packing_required && packing_mode == PackingModeV1::exclude) {
        if (!compute_only_layout(shape, compute_layout)) {
          result.legal = false;
          result.reason =
              "native packed compute-only workspace requirement overflows";
        } else {
          result.workspace_bytes = compute_layout.total_bytes;
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
      } else if (result.packing_required &&
                 packing_mode == PackingModeV1::prepack_b) {
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
          result.timing_scope =
              "prepacked-B execution: B packing prepared before timing; timed "
              "region includes transient A packing, AVX2/FMA compute, tail "
              "handling, and output stores";
        }
      } else if (result.packing_required) {
        result.timing_scope =
            "packed execution: timed region includes transient A and B "
            "packing, AVX2/FMA compute, tail handling, and output stores";
      } else if (selected.selected_variant ==
                 planner::CpuGemmVariantV2::external_openblas) {
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
    planner::format_cpu_gemm_plan_v2(selected, diagnostic.data(),
                                     diagnostic.size());
    result.diagnostic = diagnostic.data();
    if (result.legal && !result.timing_scope.empty()) {
      result.diagnostic += "\nbenchmark_timing_scope=";
      result.diagnostic += result.timing_scope;
    }
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
            planner::CpuGemmVariantV2::native_packed_avx2_fma) {
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
    if (state->plan.selected_variant !=
        planner::CpuGemmVariantV2::native_packed_avx2_fma) {
      error = "selected implementation does not support prepacked-B";
      return false;
    }
    state->packed_b_ready = false;
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
