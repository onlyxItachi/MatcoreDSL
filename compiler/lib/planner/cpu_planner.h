#ifndef MATCORE_MDSLC_PLANNER_CPU_PLANNER_H
#define MATCORE_MDSLC_PLANNER_CPU_PLANNER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace matcore::mdslc::planner {

inline constexpr std::uint32_t kCpuCapabilitiesVersionV1 = 1;
inline constexpr std::uint32_t kCpuPlannerVersionV1 = 1;
inline constexpr std::size_t kCpuGemmCandidateCountV1 = 3;

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__clang__) || defined(__GNUC__))
#define MATCORE_MDSLC_CPU_AVX2_FMA_TARGET \
  __attribute__((target("avx2,fma"), noinline))
#else
#define MATCORE_MDSLC_CPU_AVX2_FMA_TARGET
#endif

#if defined(__clang__) || defined(__GNUC__)
#define MATCORE_MDSLC_CPU_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define MATCORE_MDSLC_CPU_RESTRICT __restrict
#else
#define MATCORE_MDSLC_CPU_RESTRICT
#endif

enum class CpuArchitectureV1 : std::uint8_t {
  unknown = 0,
  x86_64 = 1,
  aarch64 = 2,
};

enum class CpuFeatureV1 : std::uint64_t {
  portable_scalar_f32 = UINT64_C(1) << 0,
  avx2 = UINT64_C(1) << 1,
  fma = UINT64_C(1) << 2,
};

constexpr std::uint64_t feature_bit(CpuFeatureV1 feature) noexcept {
  return static_cast<std::uint64_t>(feature);
}

struct CpuCapabilitiesV1 {
  std::uint32_t version = kCpuCapabilitiesVersionV1;
  CpuArchitectureV1 architecture = CpuArchitectureV1::unknown;
  bool detection_complete = false;
  std::uint64_t features = feature_bit(CpuFeatureV1::portable_scalar_f32);
  std::uint16_t usable_vector_bits = 0;
};

constexpr bool has_feature(const CpuCapabilitiesV1 &capabilities,
                           CpuFeatureV1 feature) noexcept {
  return (capabilities.features & feature_bit(feature)) != 0;
}

inline CpuCapabilitiesV1 discover_cpu_capabilities_v1() noexcept {
  CpuCapabilitiesV1 result;
#if defined(__x86_64__) || defined(_M_X64)
  result.architecture = CpuArchitectureV1::x86_64;
#if (defined(__clang__) || defined(__GNUC__)) && !defined(_MSC_VER)
  __builtin_cpu_init();
  result.detection_complete = true;
  if (__builtin_cpu_supports("avx2")) {
    result.features |= feature_bit(CpuFeatureV1::avx2);
    result.usable_vector_bits = 256;
  }
  if (__builtin_cpu_supports("fma")) {
    result.features |= feature_bit(CpuFeatureV1::fma);
  }
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  result.architecture = CpuArchitectureV1::aarch64;
  result.detection_complete = true;
#endif
  return result;
}

enum class CpuScalarTypeV1 : std::uint8_t {
  invalid = 0,
  f32 = 1,
};

enum class CpuLayoutV1 : std::uint8_t {
  invalid = 0,
  row_major_contiguous = 1,
};

struct CpuGemmProblemV1 {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  CpuScalarTypeV1 element_type = CpuScalarTypeV1::f32;
  CpuScalarTypeV1 accumulation_type = CpuScalarTypeV1::f32;
  CpuLayoutV1 layout = CpuLayoutV1::row_major_contiguous;
  std::uint32_t minimum_alignment_bytes = alignof(float);
};

enum class CpuGemmVariantV1 : std::uint8_t {
  reference = 0,
  tiled = 1,
  compiler_vectorized = 2,
};

enum class CpuGemmRequestV1 : std::uint8_t {
  automatic = 0,
  force_reference = 1,
  force_tiled = 2,
  force_compiler_vectorized = 3,
};

struct CpuGemmVariantRecordV1 {
  CpuGemmVariantV1 variant;
  std::string_view stable_id;
  std::uint64_t required_features;
  std::uint16_t deterministic_priority;
};

inline constexpr std::array<CpuGemmVariantRecordV1,
                            kCpuGemmCandidateCountV1>
    kCpuGemmVariantRegistryV1{{
        {CpuGemmVariantV1::reference, "cpu.reference.f32.v1",
         feature_bit(CpuFeatureV1::portable_scalar_f32), 30},
        {CpuGemmVariantV1::tiled, "cpu.tiled.f32.v1",
         feature_bit(CpuFeatureV1::portable_scalar_f32), 20},
        {CpuGemmVariantV1::compiler_vectorized,
         "cpu.compiler-vectorized.avx2-fma.f32.v1",
         feature_bit(CpuFeatureV1::portable_scalar_f32) |
             feature_bit(CpuFeatureV1::avx2) |
             feature_bit(CpuFeatureV1::fma),
         10},
    }};

constexpr const auto &cpu_gemm_variant_registry_v1() noexcept {
  return kCpuGemmVariantRegistryV1;
}

enum class CpuPlanStatusV1 : std::uint8_t {
  selected = 0,
  no_legal_variant = 1,
  forced_variant_illegal = 2,
  invalid_problem = 3,
  invalid_capabilities = 4,
};

struct CpuCandidateDecisionV1 {
  CpuGemmVariantV1 variant = CpuGemmVariantV1::reference;
  std::string_view stable_id;
  bool legal = false;
  std::string_view reason;
  std::uint64_t estimated_cost = std::numeric_limits<std::uint64_t>::max();
  std::uint16_t deterministic_priority = 0;
};

struct CpuGemmPlanV1 {
  std::uint32_t planner_version = kCpuPlannerVersionV1;
  CpuPlanStatusV1 status = CpuPlanStatusV1::no_legal_variant;
  CpuGemmProblemV1 problem;
  CpuCapabilitiesV1 capabilities;
  CpuGemmRequestV1 request = CpuGemmRequestV1::automatic;
  std::array<CpuCandidateDecisionV1, 3> candidates{};
  CpuGemmVariantV1 selected_variant = CpuGemmVariantV1::reference;
  std::string_view selected_id;
  std::string_view selection_reason;
};

namespace detail {

constexpr std::uint64_t saturating_add(std::uint64_t lhs,
                                       std::uint64_t rhs) noexcept {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return rhs > maximum - lhs ? maximum : lhs + rhs;
}

constexpr std::uint64_t saturating_multiply(std::uint64_t lhs,
                                            std::uint64_t rhs) noexcept {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  return lhs != 0 && rhs > maximum / lhs ? maximum : lhs * rhs;
}

constexpr std::uint64_t operation_count(const CpuGemmProblemV1 &problem) noexcept {
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0) return 0;
  return saturating_multiply(
      saturating_multiply(static_cast<std::uint64_t>(problem.m),
                          static_cast<std::uint64_t>(problem.n)),
      static_cast<std::uint64_t>(problem.k));
}

constexpr std::string_view common_legality_reason(
    const CpuGemmProblemV1 &problem) noexcept {
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0)
    return "M, N, and K must be positive";
  if (problem.element_type != CpuScalarTypeV1::f32)
    return "variant supports only f32 elements";
  if (problem.accumulation_type != CpuScalarTypeV1::f32)
    return "variant supports only f32 accumulation";
  if (problem.layout != CpuLayoutV1::row_major_contiguous)
    return "variant requires row-major contiguous matrices";
  if (problem.minimum_alignment_bytes < alignof(float))
    return "variant requires float-aligned storage";
  if ((problem.minimum_alignment_bytes &
       (problem.minimum_alignment_bytes - 1U)) != 0)
    return "minimum alignment must be a power of two";
  return {};
}

constexpr std::string_view capability_legality_reason(
    const CpuGemmVariantRecordV1 &record,
    const CpuCapabilitiesV1 &capabilities) noexcept {
  if (capabilities.version != kCpuCapabilitiesVersionV1)
    return "CPU capability record version is unsupported";
  if (!has_feature(capabilities, CpuFeatureV1::portable_scalar_f32))
    return "portable scalar f32 capability is required";
  if (record.variant == CpuGemmVariantV1::compiler_vectorized) {
    if (!capabilities.detection_complete)
      return "AVX2/FMA discovery is incomplete";
    if ((capabilities.features & record.required_features) !=
        record.required_features)
      return "AVX2 and FMA are required";
    if (capabilities.architecture != CpuArchitectureV1::x86_64)
      return "compiler-vectorized candidate requires x86_64";
    if (capabilities.usable_vector_bits < 256)
      return "compiler-vectorized candidate requires 256-bit vectors";
  }
  return {};
}

constexpr std::uint64_t estimated_cost(
    CpuGemmVariantV1 variant, const CpuGemmProblemV1 &problem) noexcept {
  const std::uint64_t work = operation_count(problem);
  switch (variant) {
    case CpuGemmVariantV1::reference:
      return saturating_multiply(work, 8);
    case CpuGemmVariantV1::tiled: {
      const auto tail = static_cast<std::uint64_t>(problem.m % 32) +
                        static_cast<std::uint64_t>(problem.n % 32) +
                        static_cast<std::uint64_t>(problem.k % 64);
      return saturating_add(
          saturating_add(saturating_multiply(work, 4), 4096),
          saturating_multiply(tail, 64));
    }
    case CpuGemmVariantV1::compiler_vectorized: {
      std::uint64_t cost =
          saturating_add(saturating_multiply(work, 2), 16384);
      if (problem.minimum_alignment_bytes < 32)
        cost = saturating_add(cost, 24576);
      return saturating_add(
          cost, saturating_multiply(static_cast<std::uint64_t>(problem.n % 8),
                                    128));
    }
  }
  return std::numeric_limits<std::uint64_t>::max();
}

constexpr bool request_matches(CpuGemmRequestV1 request,
                               CpuGemmVariantV1 variant) noexcept {
  switch (request) {
    case CpuGemmRequestV1::automatic:
      return true;
    case CpuGemmRequestV1::force_reference:
      return variant == CpuGemmVariantV1::reference;
    case CpuGemmRequestV1::force_tiled:
      return variant == CpuGemmVariantV1::tiled;
    case CpuGemmRequestV1::force_compiler_vectorized:
      return variant == CpuGemmVariantV1::compiler_vectorized;
  }
  return false;
}

inline void reference_gemm(const CpuGemmProblemV1 &problem, const float *lhs,
                           const float *rhs, float *out) noexcept {
  for (std::int64_t i = 0; i < problem.m; ++i) {
    for (std::int64_t j = 0; j < problem.n; ++j) {
      float sum = 0.0F;
      for (std::int64_t p = 0; p < problem.k; ++p)
        sum += lhs[i * problem.k + p] * rhs[p * problem.n + j];
      out[i * problem.n + j] = sum;
    }
  }
}

inline void tiled_gemm(const CpuGemmProblemV1 &problem, const float *lhs,
                       const float *rhs, float *out) noexcept {
  constexpr std::int64_t tile_m = 32;
  constexpr std::int64_t tile_n = 32;
  constexpr std::int64_t tile_k = 64;
  for (std::int64_t i = 0; i < problem.m; ++i)
    for (std::int64_t j = 0; j < problem.n; ++j)
      out[i * problem.n + j] = 0.0F;
  for (std::int64_t ii = 0; ii < problem.m; ii += tile_m) {
    const std::int64_t i_end =
        ii + tile_m < problem.m ? ii + tile_m : problem.m;
    for (std::int64_t pp = 0; pp < problem.k; pp += tile_k) {
      const std::int64_t p_end =
          pp + tile_k < problem.k ? pp + tile_k : problem.k;
      for (std::int64_t jj = 0; jj < problem.n; jj += tile_n) {
        const std::int64_t j_end =
            jj + tile_n < problem.n ? jj + tile_n : problem.n;
        for (std::int64_t i = ii; i < i_end; ++i) {
          for (std::int64_t p = pp; p < p_end; ++p) {
            const float a = lhs[i * problem.k + p];
            for (std::int64_t j = jj; j < j_end; ++j)
              out[i * problem.n + j] += a * rhs[p * problem.n + j];
          }
        }
      }
    }
  }
}

MATCORE_MDSLC_CPU_AVX2_FMA_TARGET
inline void compiler_vectorized_gemm(
    const CpuGemmProblemV1 &problem,
    const float *MATCORE_MDSLC_CPU_RESTRICT lhs,
    const float *MATCORE_MDSLC_CPU_RESTRICT rhs,
    float *MATCORE_MDSLC_CPU_RESTRICT out) noexcept {
  for (std::int64_t i = 0; i < problem.m; ++i)
    for (std::int64_t j = 0; j < problem.n; ++j)
      out[i * problem.n + j] = 0.0F;
  for (std::int64_t i = 0; i < problem.m; ++i) {
    for (std::int64_t p = 0; p < problem.k; ++p) {
      const float a = lhs[i * problem.k + p];
#if defined(__clang__)
#pragma clang loop vectorize(enable) vectorize_width(8) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
      for (std::int64_t j = 0; j < problem.n; ++j)
        out[i * problem.n + j] += a * rhs[p * problem.n + j];
    }
  }
}

struct DiagnosticWriter {
  char *output;
  std::size_t capacity;
  std::size_t required = 0;

  void character(char value) noexcept {
    if (output != nullptr && capacity != 0 && required + 1 < capacity)
      output[required] = value;
    ++required;
  }
  void text(std::string_view value) noexcept {
    for (char character_value : value) character(character_value);
  }
  void number(std::uint64_t value) noexcept {
    char reversed[20];
    std::size_t count = 0;
    do {
      reversed[count++] = static_cast<char>('0' + value % 10);
      value /= 10;
    } while (value != 0);
    while (count != 0) character(reversed[--count]);
  }
  std::size_t finish() noexcept {
    if (output != nullptr && capacity != 0) {
      const std::size_t terminator = required < capacity ? required : capacity - 1;
      output[terminator] = '\0';
    }
    return required;
  }
};

}  // namespace detail

inline CpuGemmPlanV1 plan_cpu_gemm_v1(
    const CpuGemmProblemV1 &problem, const CpuCapabilitiesV1 &capabilities,
    CpuGemmRequestV1 request = CpuGemmRequestV1::automatic) noexcept {
  CpuGemmPlanV1 plan;
  plan.problem = problem;
  plan.capabilities = capabilities;
  plan.request = request;

  const std::string_view problem_reason = detail::common_legality_reason(problem);
  if (!problem_reason.empty()) {
    plan.status = CpuPlanStatusV1::invalid_problem;
    plan.selection_reason = problem_reason;
  } else if (capabilities.version != kCpuCapabilitiesVersionV1) {
    plan.status = CpuPlanStatusV1::invalid_capabilities;
    plan.selection_reason = "CPU capability record version is unsupported";
  }

  bool found = false;
  std::uint64_t best_cost = std::numeric_limits<std::uint64_t>::max();
  std::uint16_t best_priority = std::numeric_limits<std::uint16_t>::max();
  std::size_t best_registry_index = kCpuGemmVariantRegistryV1.size();
  for (std::size_t index = 0; index < kCpuGemmVariantRegistryV1.size(); ++index) {
    const auto &record = kCpuGemmVariantRegistryV1[index];
    auto &decision = plan.candidates[index];
    decision.variant = record.variant;
    decision.stable_id = record.stable_id;
    decision.deterministic_priority = record.deterministic_priority;
    if (!problem_reason.empty()) {
      decision.reason = problem_reason;
      continue;
    }
    const auto capability_reason =
        detail::capability_legality_reason(record, capabilities);
    if (!capability_reason.empty()) {
      decision.reason = capability_reason;
      continue;
    }
    decision.legal = true;
    decision.reason = "legal";
    decision.estimated_cost = detail::estimated_cost(record.variant, problem);
    if (!detail::request_matches(request, record.variant)) continue;
    if (!found || decision.estimated_cost < best_cost ||
        (decision.estimated_cost == best_cost &&
         (decision.deterministic_priority < best_priority ||
          (decision.deterministic_priority == best_priority &&
           index < best_registry_index)))) {
      found = true;
      best_cost = decision.estimated_cost;
      best_priority = decision.deterministic_priority;
      best_registry_index = index;
      plan.selected_variant = record.variant;
      plan.selected_id = record.stable_id;
    }
  }

  if (!problem_reason.empty() ||
      capabilities.version != kCpuCapabilitiesVersionV1)
    return plan;
  if (!found) {
    plan.status = request == CpuGemmRequestV1::automatic
                      ? CpuPlanStatusV1::no_legal_variant
                      : CpuPlanStatusV1::forced_variant_illegal;
    plan.selection_reason = request == CpuGemmRequestV1::automatic
                                ? "no legal CPU GEMM variant"
                                : "requested CPU GEMM variant is illegal";
    return plan;
  }
  plan.status = CpuPlanStatusV1::selected;
  plan.selection_reason =
      request == CpuGemmRequestV1::automatic
          ? "lowest deterministic cost; ties use priority then registry order"
          : "explicit legal variant request";
  return plan;
}

inline bool execute_cpu_gemm_plan_v1(const CpuGemmPlanV1 &plan,
                                     const float *lhs, const float *rhs,
                                     float *out) noexcept {
  if (plan.status != CpuPlanStatusV1::selected || lhs == nullptr ||
      rhs == nullptr || out == nullptr)
    return false;
  const auto selected_index = static_cast<std::size_t>(plan.selected_variant);
  if (selected_index >= plan.candidates.size() ||
      !plan.candidates[selected_index].legal ||
      plan.candidates[selected_index].variant != plan.selected_variant)
    return false;
  switch (plan.selected_variant) {
    case CpuGemmVariantV1::reference:
      detail::reference_gemm(plan.problem, lhs, rhs, out);
      return true;
    case CpuGemmVariantV1::tiled:
      detail::tiled_gemm(plan.problem, lhs, rhs, out);
      return true;
    case CpuGemmVariantV1::compiler_vectorized:
      detail::compiler_vectorized_gemm(plan.problem, lhs, rhs, out);
      return true;
  }
  return false;
}

inline std::uint32_t pointer_alignment_bytes(const void *pointer) noexcept {
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address == 0) return 0;
  std::uint32_t alignment = 1;
  while (alignment < 64 && (address % (alignment * 2U)) == 0)
    alignment *= 2U;
  return alignment;
}

inline std::size_t format_cpu_gemm_plan_v1(const CpuGemmPlanV1 &plan,
                                           char *output,
                                           std::size_t capacity) noexcept {
  if (output == nullptr) capacity = 0;
  detail::DiagnosticWriter writer{output, capacity};
  writer.text("cpu-planner-v1 arch=");
  switch (plan.capabilities.architecture) {
    case CpuArchitectureV1::unknown:
      writer.text("unknown");
      break;
    case CpuArchitectureV1::x86_64:
      writer.text("x86_64");
      break;
    case CpuArchitectureV1::aarch64:
      writer.text("aarch64");
      break;
  }
  writer.text(" detection_complete=");
  writer.text(plan.capabilities.detection_complete ? "true" : "false");
  writer.text(" features=[");
  bool wrote_feature = false;
  const auto write_feature = [&](CpuFeatureV1 feature,
                                 std::string_view name) noexcept {
    if (!has_feature(plan.capabilities, feature)) return;
    if (wrote_feature) writer.character(',');
    writer.text(name);
    wrote_feature = true;
  };
  write_feature(CpuFeatureV1::portable_scalar_f32, "portable-scalar-f32");
  write_feature(CpuFeatureV1::avx2, "avx2");
  write_feature(CpuFeatureV1::fma, "fma");
  writer.text("] vector_bits=");
  writer.number(plan.capabilities.usable_vector_bits);
  writer.text(" request=");
  switch (plan.request) {
    case CpuGemmRequestV1::automatic:
      writer.text("automatic");
      break;
    case CpuGemmRequestV1::force_reference:
      writer.text("force-reference");
      break;
    case CpuGemmRequestV1::force_tiled:
      writer.text("force-tiled");
      break;
    case CpuGemmRequestV1::force_compiler_vectorized:
      writer.text("force-compiler-vectorized");
      break;
  }
  writer.text(" status=");
  switch (plan.status) {
    case CpuPlanStatusV1::selected:
      writer.text("selected");
      break;
    case CpuPlanStatusV1::no_legal_variant:
      writer.text("no-legal-variant");
      break;
    case CpuPlanStatusV1::forced_variant_illegal:
      writer.text("forced-variant-illegal");
      break;
    case CpuPlanStatusV1::invalid_problem:
      writer.text("invalid-problem");
      break;
    case CpuPlanStatusV1::invalid_capabilities:
      writer.text("invalid-capabilities");
      break;
  }
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
    const auto &candidate = plan.candidates[index];
    writer.text(candidate.stable_id);
    writer.character(':');
    if (candidate.legal) {
      writer.text("legal:cost=");
      writer.number(candidate.estimated_cost);
    } else {
      writer.text("rejected:");
      writer.text(candidate.reason);
    }
  }
  writer.character(']');
  return writer.finish();
}

}  // namespace matcore::mdslc::planner

#undef MATCORE_MDSLC_CPU_RESTRICT
#undef MATCORE_MDSLC_CPU_AVX2_FMA_TARGET

#endif
