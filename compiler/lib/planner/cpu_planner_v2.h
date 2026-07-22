#ifndef MATCORE_MDSLC_PLANNER_CPU_PLANNER_V2_H
#define MATCORE_MDSLC_PLANNER_CPU_PLANNER_V2_H

#include "cpu_planner.h"

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace matcore::mdslc::planner {

inline constexpr std::uint32_t kCpuPlannerVersionV2 = 2;
inline constexpr std::size_t kCpuGemmCandidateCountV2 = 5;

enum class CpuGemmVariantV2 : std::uint8_t {
  reference = 0,
  tiled = 1,
  compiler_vectorized = 2,
  external_openblas = 3,
  native_packed_avx2_fma = 4,
};

enum class CpuGemmRequestV2 : std::uint8_t {
  automatic = 0,
  force_reference = 1,
  force_tiled = 2,
  force_compiler_vectorized = 3,
  force_external_openblas = 4,
  force_native_packed_avx2_fma = 5,
};

struct CpuGemmImplementationResourcesV1 {
  bool openblas_linked = false;
  bool openblas_local_thread_control = false;
  bool native_packed_avx2_fma_compiled = false;
  bool native_packed_workspace_size_valid = false;
  std::uint64_t native_packed_workspace_bytes = 0;
  std::uint32_t native_packed_workspace_alignment = 0;
  std::uint32_t requested_threads = 1;
};

struct CpuGemmVariantRecordV2 {
  CpuGemmVariantV2 variant;
  std::string_view stable_id;
  std::uint64_t required_features;
  std::uint16_t deterministic_priority;
};

inline constexpr std::array<CpuGemmVariantRecordV2,
                            kCpuGemmCandidateCountV2>
    kCpuGemmVariantRegistryV2{{
        {CpuGemmVariantV2::reference, "cpu.reference.f32.v1",
         feature_bit(CpuFeatureV1::portable_scalar_f32), 50},
        {CpuGemmVariantV2::tiled, "cpu.tiled.f32.v1",
         feature_bit(CpuFeatureV1::portable_scalar_f32), 40},
        {CpuGemmVariantV2::compiler_vectorized,
         "cpu.compiler-vectorized.avx2-fma.f32.v1",
         feature_bit(CpuFeatureV1::portable_scalar_f32) |
             feature_bit(CpuFeatureV1::avx2) |
             feature_bit(CpuFeatureV1::fma),
         30},
        {CpuGemmVariantV2::external_openblas,
         "cpu.external.openblas.f32.v1",
         feature_bit(CpuFeatureV1::portable_scalar_f32), 10},
        {CpuGemmVariantV2::native_packed_avx2_fma,
         "cpu.native-packed.avx2-fma.f32.v1",
         feature_bit(CpuFeatureV1::portable_scalar_f32) |
             feature_bit(CpuFeatureV1::avx2) |
             feature_bit(CpuFeatureV1::fma),
         20},
    }};

struct CpuCandidateDecisionV2 {
  CpuGemmVariantV2 variant = CpuGemmVariantV2::reference;
  std::string_view stable_id;
  bool legal = false;
  std::string_view reason;
  std::uint64_t estimated_cost = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t required_workspace_bytes = 0;
  std::uint32_t required_workspace_alignment = 1;
  std::uint32_t actual_threads = 1;
  std::uint16_t deterministic_priority = 0;
};

struct CpuGemmPlanV2 {
  std::uint32_t planner_version = kCpuPlannerVersionV2;
  CpuPlanStatusV1 status = CpuPlanStatusV1::no_legal_variant;
  CpuGemmProblemV1 problem;
  CpuCapabilitiesV1 capabilities;
  CpuGemmImplementationResourcesV1 resources;
  CpuGemmRequestV2 request = CpuGemmRequestV2::automatic;
  std::array<CpuCandidateDecisionV2, kCpuGemmCandidateCountV2> candidates{};
  CpuGemmVariantV2 selected_variant = CpuGemmVariantV2::reference;
  std::string_view selected_id;
  std::string_view selection_reason;
};

namespace detail_v2 {

constexpr CpuGemmRequestV1 legacy_request(CpuGemmVariantV2 variant) noexcept {
  switch (variant) {
    case CpuGemmVariantV2::reference:
      return CpuGemmRequestV1::force_reference;
    case CpuGemmVariantV2::tiled:
      return CpuGemmRequestV1::force_tiled;
    case CpuGemmVariantV2::compiler_vectorized:
      return CpuGemmRequestV1::force_compiler_vectorized;
    case CpuGemmVariantV2::external_openblas:
    case CpuGemmVariantV2::native_packed_avx2_fma:
      return CpuGemmRequestV1::force_reference;
  }
  return CpuGemmRequestV1::force_reference;
}

constexpr bool request_matches(CpuGemmRequestV2 request,
                               CpuGemmVariantV2 variant) noexcept {
  switch (request) {
    case CpuGemmRequestV2::automatic:
      return true;
    case CpuGemmRequestV2::force_reference:
      return variant == CpuGemmVariantV2::reference;
    case CpuGemmRequestV2::force_tiled:
      return variant == CpuGemmVariantV2::tiled;
    case CpuGemmRequestV2::force_compiler_vectorized:
      return variant == CpuGemmVariantV2::compiler_vectorized;
    case CpuGemmRequestV2::force_external_openblas:
      return variant == CpuGemmVariantV2::external_openblas;
    case CpuGemmRequestV2::force_native_packed_avx2_fma:
      return variant == CpuGemmVariantV2::native_packed_avx2_fma;
  }
  return false;
}

constexpr std::string_view extra_legality_reason(
    CpuGemmVariantV2 variant, const CpuGemmProblemV1 &problem,
    const CpuCapabilitiesV1 &capabilities,
    const CpuGemmImplementationResourcesV1 &resources) noexcept {
  if (resources.requested_threads == 0)
    return "requested thread count must be positive";
  switch (variant) {
    case CpuGemmVariantV2::reference:
    case CpuGemmVariantV2::tiled:
    case CpuGemmVariantV2::compiler_vectorized:
      return {};
    case CpuGemmVariantV2::external_openblas:
      if (!resources.openblas_linked)
        return "OpenBLAS CBLAS adapter is not linked";
      if (!resources.openblas_local_thread_control)
        return "OpenBLAS local thread control is unavailable";
      if (problem.m > INT_MAX || problem.n > INT_MAX || problem.k > INT_MAX)
        return "OpenBLAS LP64 dimensions exceed INT_MAX";
      return {};
    case CpuGemmVariantV2::native_packed_avx2_fma:
      if (!resources.native_packed_avx2_fma_compiled)
        return "native packed AVX2/FMA implementation is not compiled";
      if (!resources.native_packed_workspace_size_valid)
        return "native packed workspace requirement overflowed";
      if (resources.native_packed_workspace_alignment < 32 ||
          (resources.native_packed_workspace_alignment &
           (resources.native_packed_workspace_alignment - 1U)) != 0)
        return "native packed workspace alignment is invalid";
      if (!capabilities.detection_complete ||
          capabilities.architecture != CpuArchitectureV1::x86_64 ||
          !has_feature(capabilities, CpuFeatureV1::avx2) ||
          !has_feature(capabilities, CpuFeatureV1::fma) ||
          capabilities.usable_vector_bits < 256)
        return "native packed candidate requires usable x86_64 AVX2/FMA";
      if (resources.requested_threads != 1)
        return "native packed v1 is single-threaded";
      return {};
  }
  return "unknown CPU GEMM variant";
}

constexpr std::uint64_t estimated_cost(
    CpuGemmVariantV2 variant, const CpuGemmProblemV1 &problem) noexcept {
  const std::uint64_t work = detail::operation_count(problem);
  const std::uint64_t packing = detail::saturating_add(
      detail::saturating_multiply(static_cast<std::uint64_t>(problem.m),
                                  static_cast<std::uint64_t>(problem.k)),
      detail::saturating_multiply(static_cast<std::uint64_t>(problem.k),
                                  static_cast<std::uint64_t>(problem.n)));
  switch (variant) {
    case CpuGemmVariantV2::reference:
      return detail::saturating_multiply(work, 16);
    case CpuGemmVariantV2::tiled:
      return detail::saturating_add(detail::saturating_multiply(work, 7),
                                    4096);
    case CpuGemmVariantV2::compiler_vectorized:
      return detail::saturating_add(detail::saturating_multiply(work, 4),
                                    16384);
    case CpuGemmVariantV2::external_openblas:
      return detail::saturating_add(work, 180000);
    case CpuGemmVariantV2::native_packed_avx2_fma:
      return detail::saturating_add(
          detail::saturating_add(detail::saturating_multiply(work, 2),
                                 detail::saturating_multiply(packing, 6)),
          48000);
  }
  return std::numeric_limits<std::uint64_t>::max();
}

}  // namespace detail_v2

inline CpuGemmPlanV2 plan_cpu_gemm_v2(
    const CpuGemmProblemV1 &problem, const CpuCapabilitiesV1 &capabilities,
    const CpuGemmImplementationResourcesV1 &resources,
    CpuGemmRequestV2 request = CpuGemmRequestV2::automatic) noexcept {
  CpuGemmPlanV2 plan;
  plan.problem = problem;
  plan.capabilities = capabilities;
  plan.resources = resources;
  plan.request = request;

  const CpuGemmPlanV1 base = plan_cpu_gemm_v1(
      problem, capabilities, CpuGemmRequestV1::force_reference);
  if (base.status == CpuPlanStatusV1::invalid_problem ||
      base.status == CpuPlanStatusV1::invalid_capabilities) {
    plan.status = base.status;
    plan.selection_reason = base.selection_reason;
  }

  bool found = false;
  std::uint64_t best_cost = std::numeric_limits<std::uint64_t>::max();
  std::uint16_t best_priority = std::numeric_limits<std::uint16_t>::max();
  std::size_t best_index = kCpuGemmCandidateCountV2;
  for (std::size_t index = 0; index < kCpuGemmCandidateCountV2; ++index) {
    const CpuGemmVariantRecordV2 &record = kCpuGemmVariantRegistryV2[index];
    CpuCandidateDecisionV2 &decision = plan.candidates[index];
    decision.variant = record.variant;
    decision.stable_id = record.stable_id;
    decision.deterministic_priority = record.deterministic_priority;
    decision.actual_threads =
        record.variant == CpuGemmVariantV2::external_openblas
            ? resources.requested_threads
            : 1;
    if (record.variant == CpuGemmVariantV2::native_packed_avx2_fma) {
      decision.required_workspace_bytes =
          resources.native_packed_workspace_bytes;
      decision.required_workspace_alignment =
          resources.native_packed_workspace_alignment;
    }

    if (base.status == CpuPlanStatusV1::invalid_problem ||
        base.status == CpuPlanStatusV1::invalid_capabilities) {
      decision.reason = base.selection_reason;
      continue;
    }

    if (record.variant == CpuGemmVariantV2::reference ||
        record.variant == CpuGemmVariantV2::tiled ||
        record.variant == CpuGemmVariantV2::compiler_vectorized) {
      const CpuGemmPlanV1 legacy = plan_cpu_gemm_v1(
          problem, capabilities, detail_v2::legacy_request(record.variant));
      if (legacy.status != CpuPlanStatusV1::selected) {
        const std::size_t legacy_index = static_cast<std::size_t>(
            record.variant == CpuGemmVariantV2::reference
                ? CpuGemmVariantV1::reference
                : record.variant == CpuGemmVariantV2::tiled
                      ? CpuGemmVariantV1::tiled
                      : CpuGemmVariantV1::compiler_vectorized);
        decision.reason = legacy.candidates[legacy_index].reason;
        continue;
      }
    }

    const std::string_view extra_reason = detail_v2::extra_legality_reason(
        record.variant, problem, capabilities, resources);
    if (!extra_reason.empty()) {
      decision.reason = extra_reason;
      continue;
    }
    decision.legal = true;
    decision.reason = "legal";
    decision.estimated_cost = detail_v2::estimated_cost(record.variant, problem);
    if (!detail_v2::request_matches(request, record.variant)) continue;
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
    plan.status = request == CpuGemmRequestV2::automatic
                      ? CpuPlanStatusV1::no_legal_variant
                      : CpuPlanStatusV1::forced_variant_illegal;
    plan.selection_reason = request == CpuGemmRequestV2::automatic
                                ? "no legal CPU GEMM variant"
                                : "requested CPU GEMM variant is illegal";
    return plan;
  }
  plan.status = CpuPlanStatusV1::selected;
  plan.selection_reason =
      request == CpuGemmRequestV2::automatic
          ? "lowest deterministic calibrated cost; ties use priority then registry order"
          : "explicit legal variant request";
  return plan;
}

inline std::size_t format_cpu_gemm_plan_v2(const CpuGemmPlanV2 &plan,
                                           char *output,
                                           std::size_t capacity) noexcept {
  if (output == nullptr) capacity = 0;
  detail::DiagnosticWriter writer{output, capacity};
  writer.text("cpu-planner-v2 request=");
  writer.number(static_cast<std::uint8_t>(plan.request));
  writer.text(" threads=");
  writer.number(plan.resources.requested_threads);
  writer.text(" status=");
  writer.number(static_cast<std::uint8_t>(plan.status));
  writer.text(" m=");
  writer.number(plan.problem.m > 0 ? static_cast<std::uint64_t>(plan.problem.m)
                                   : 0);
  writer.text(" n=");
  writer.number(plan.problem.n > 0 ? static_cast<std::uint64_t>(plan.problem.n)
                                   : 0);
  writer.text(" k=");
  writer.number(plan.problem.k > 0 ? static_cast<std::uint64_t>(plan.problem.k)
                                   : 0);
  writer.text(" selected=");
  writer.text(plan.selected_id.empty() ? "none" : plan.selected_id);
  writer.text(" reason=");
  writer.text(plan.selection_reason);
  writer.text(" candidates=[");
  for (std::size_t index = 0; index < plan.candidates.size(); ++index) {
    if (index != 0) writer.character(',');
    const CpuCandidateDecisionV2 &candidate = plan.candidates[index];
    writer.text(candidate.stable_id);
    writer.character(':');
    if (candidate.legal) {
      writer.text("legal:cost=");
      writer.number(candidate.estimated_cost);
      writer.text(":workspace=");
      writer.number(candidate.required_workspace_bytes);
      writer.text(":alignment=");
      writer.number(candidate.required_workspace_alignment);
      writer.text(":threads=");
      writer.number(candidate.actual_threads);
    } else {
      writer.text("rejected:");
      writer.text(candidate.reason);
    }
  }
  writer.character(']');
  return writer.finish();
}

}  // namespace matcore::mdslc::planner

#endif
