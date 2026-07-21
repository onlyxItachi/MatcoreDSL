#include "matcore/runtime_c.h"

#include "cpu_planner.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

std::string_view text_or_empty(const char *text) {
  return text == nullptr ? std::string_view{} : std::string_view(text);
}

matcore_tensor_desc_v0 desc(void *data, std::int64_t rows,
                            std::int64_t cols,
                            matcore_mutability_v0 mutability =
                                MATCORE_MUTABILITY_READ_ONLY_V0) {
  matcore_tensor_desc_v0 value{};
  value.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  value.struct_size = sizeof(value);
  value.data = data;
  value.dtype = MATCORE_DTYPE_F32_V0;
  value.rank = 2;
  value.dims[0] = rows;
  value.dims[1] = cols;
  value.strides[0] = cols;
  value.strides[1] = 1;
  value.memory_space = MATCORE_MEMORY_SPACE_HOST_V0;
  value.mutability = mutability;
  return value;
}

matcore_policy_v0 policy() {
  matcore_policy_v0 value{};
  value.abi_version = MATCORE_RUNTIME_ABI_VERSION_V0;
  value.struct_size = sizeof(value);
  value.target = MATCORE_TARGET_CPU_V0;
  value.fallback = MATCORE_FALLBACK_ERROR_V0;
  return value;
}

void run_shape(std::int64_t m, std::int64_t k, std::int64_t n) {
  std::vector<float> a(static_cast<std::size_t>(m * k));
  std::vector<float> b(static_cast<std::size_t>(k * n));
  std::vector<float> c(static_cast<std::size_t>(m * n), -99.0F);
  std::vector<double> oracle(static_cast<std::size_t>(m * n), 0.0);
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
  for (std::size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<float>(static_cast<int>(i % 5) - 2) / 2.0F;
  for (std::int64_t p = 0; p < k; ++p)
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t i = 0; i < m; ++i)
        oracle[static_cast<std::size_t>(i * n + j)] +=
            static_cast<double>(a[static_cast<std::size_t>(i * k + p)]) *
            static_cast<double>(b[static_cast<std::size_t>(p * n + j)]);

  auto out = desc(c.data(), m, n, MATCORE_MUTABILITY_READ_WRITE_V0);
  auto lhs = desc(a.data(), m, k);
  auto rhs = desc(b.data(), k, n);
  const auto p = policy();
  const auto result = matcore_runtime_gemm_f32_v0(&out, &lhs, &rhs, &p);
  expect(result.code == MATCORE_STATUS_OK_V0, "valid GEMM returns success");
  expect(text_or_empty(result.message) == "ok",
         "legacy GEMM success diagnostic remains stable");
  for (std::size_t i = 0; i < c.size(); ++i)
    expect(std::fabs(static_cast<double>(c[i]) - oracle[i]) < 1.0e-5,
           "GEMM matches independent oracle");
}

void run_aligned_shape() {
  constexpr std::int64_t m = 24;
  constexpr std::int64_t k = 24;
  constexpr std::int64_t n = 24;
  alignas(64) std::array<float, static_cast<std::size_t>(m * k)> a{};
  alignas(64) std::array<float, static_cast<std::size_t>(k * n)> b{};
  alignas(64) std::array<float, static_cast<std::size_t>(m * n)> c{};
  std::vector<double> oracle(static_cast<std::size_t>(m * n), 0.0);
  for (std::size_t index = 0; index < a.size(); ++index)
    a[index] = static_cast<float>(static_cast<int>(index % 9) - 4) / 3.0F;
  for (std::size_t index = 0; index < b.size(); ++index)
    b[index] = static_cast<float>(static_cast<int>(index % 7) - 3) / 5.0F;
  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t p = 0; p < k; ++p)
        oracle[static_cast<std::size_t>(i * n + j)] +=
            static_cast<double>(a[static_cast<std::size_t>(i * k + p)]) *
            static_cast<double>(b[static_cast<std::size_t>(p * n + j)]);

  auto out = desc(c.data(), m, n, MATCORE_MUTABILITY_READ_WRITE_V0);
  auto lhs = desc(a.data(), m, k);
  auto rhs = desc(b.data(), k, n);
  const auto p = policy();
  const auto result = matcore_runtime_gemm_f32_v0(&out, &lhs, &rhs, &p);
  expect(result.code == MATCORE_STATUS_OK_V0,
         "64-byte-aligned GEMM returns success");
  for (std::size_t index = 0; index < c.size(); ++index)
    expect(std::fabs(static_cast<double>(c[index]) - oracle[index]) < 1.0e-5,
           "64-byte-aligned GEMM matches independent oracle");

  const auto output_before = c;
  matcore_cpu_gemm_plan_report_v1 report{};
  report.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V1;
  report.struct_size = sizeof(report);
  expect(matcore_runtime_plan_gemm_f32_v1(&out, &lhs, &rhs, &p, &report)
             .code == MATCORE_STATUS_OK_V0 &&
             report.minimum_alignment_bytes >= 64 && c == output_before,
         "aligned runtime query preserves output and alignment metadata");
}

using matcore::mdslc::planner::CpuArchitectureV1;
using matcore::mdslc::planner::CpuCapabilitiesV1;
using matcore::mdslc::planner::CpuFeatureV1;
using matcore::mdslc::planner::CpuGemmPlanV1;
using matcore::mdslc::planner::CpuGemmProblemV1;
using matcore::mdslc::planner::CpuGemmRequestV1;
using matcore::mdslc::planner::CpuGemmVariantV1;
using matcore::mdslc::planner::CpuPlanStatusV1;
using matcore::mdslc::planner::feature_bit;

CpuCapabilitiesV1 scalar_capabilities() {
  return {matcore::mdslc::planner::kCpuCapabilitiesVersionV1,
          CpuArchitectureV1::x86_64,
          true,
          feature_bit(CpuFeatureV1::portable_scalar_f32),
          0};
}

CpuCapabilitiesV1 vector_capabilities() {
  auto value = scalar_capabilities();
  value.features |= feature_bit(CpuFeatureV1::avx2) |
                    feature_bit(CpuFeatureV1::fma);
  value.usable_vector_bits = 256;
  return value;
}

CpuGemmProblemV1 problem(std::int64_t m, std::int64_t k, std::int64_t n,
                         std::uint32_t alignment = 64) {
  return {m,
          n,
          k,
          matcore::mdslc::planner::CpuScalarTypeV1::f32,
          matcore::mdslc::planner::CpuScalarTypeV1::f32,
          matcore::mdslc::planner::CpuLayoutV1::row_major_contiguous,
          alignment};
}

void expect_selected(const CpuGemmPlanV1 &plan, CpuGemmVariantV1 variant,
                     std::string_view name) {
  expect(plan.status == CpuPlanStatusV1::selected,
         std::string(name) + " produces a plan");
  expect(plan.selected_variant == variant,
         std::string(name) + " selects the expected variant");
  expect(!plan.selected_id.empty(),
         std::string(name) + " has a stable selected ID");
  expect(!plan.selection_reason.empty(),
         std::string(name) + " explains its selection");
}

void planner_contract() {
  const auto &registry =
      matcore::mdslc::planner::cpu_gemm_variant_registry_v1();
  expect(registry.size() == 3, "CPU GEMM registry contains three variants");
  expect(registry[0].stable_id == "cpu.reference.f32.v1",
         "reference has a stable registry ID");
  expect(registry[1].stable_id == "cpu.tiled.f32.v1",
         "tiled has a stable registry ID");
  expect(registry[2].stable_id ==
             "cpu.compiler-vectorized.avx2-fma.f32.v1",
         "compiler-vectorized has a stable registry ID");
  expect(registry[0].stable_id != registry[1].stable_id &&
             registry[0].stable_id != registry[2].stable_id &&
             registry[1].stable_id != registry[2].stable_id,
         "registry stable IDs are unique");
  expect(registry[0].required_features ==
             feature_bit(CpuFeatureV1::portable_scalar_f32) &&
             registry[1].required_features ==
                 feature_bit(CpuFeatureV1::portable_scalar_f32) &&
             registry[2].required_features ==
                 (feature_bit(CpuFeatureV1::portable_scalar_f32) |
                  feature_bit(CpuFeatureV1::avx2) |
                  feature_bit(CpuFeatureV1::fma)),
         "registry feature requirements are fixed and exact");
  expect(registry[0].deterministic_priority == 30 &&
             registry[1].deterministic_priority == 20 &&
             registry[2].deterministic_priority == 10,
         "registry priorities define a stable final tie-break");

  const auto detected =
      matcore::mdslc::planner::discover_cpu_capabilities_v1();
  expect(detected.version ==
             matcore::mdslc::planner::kCpuCapabilitiesVersionV1,
         "capability discovery is versioned");
  expect(matcore::mdslc::planner::has_feature(
             detected, CpuFeatureV1::portable_scalar_f32),
         "capability discovery always records portable scalar f32");
#if defined(__x86_64__)
  expect(detected.architecture == CpuArchitectureV1::x86_64,
         "capability discovery records x86_64");
  expect(detected.detection_complete,
         "x86 builtin capability discovery completed");
#endif

  const auto vector = vector_capabilities();
  expect_selected(matcore::mdslc::planner::plan_cpu_gemm_v1(
                      problem(2, 3, 2), vector),
                  CpuGemmVariantV1::reference, "small GEMM");
  expect_selected(matcore::mdslc::planner::plan_cpu_gemm_v1(
                      problem(16, 16, 16), vector),
                  CpuGemmVariantV1::tiled, "medium GEMM");
  expect_selected(matcore::mdslc::planner::plan_cpu_gemm_v1(
                      problem(32, 32, 32), vector),
                  CpuGemmVariantV1::compiler_vectorized, "large square GEMM");
  expect_selected(matcore::mdslc::planner::plan_cpu_gemm_v1(
                      problem(33, 35, 37), vector),
                  CpuGemmVariantV1::compiler_vectorized, "tail GEMM");
  expect_selected(matcore::mdslc::planner::plan_cpu_gemm_v1(
                      problem(24, 24, 24, 64), vector),
                  CpuGemmVariantV1::compiler_vectorized,
                  "aligned GEMM");
  expect_selected(matcore::mdslc::planner::plan_cpu_gemm_v1(
                      problem(24, 24, 24, alignof(float)), vector),
                  CpuGemmVariantV1::tiled, "minimally aligned GEMM");
  const auto scalar_large = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(32, 32, 32), scalar_capabilities());
  expect_selected(scalar_large, CpuGemmVariantV1::tiled,
                  "scalar-only large GEMM");
  expect(!scalar_large.candidates[2].legal &&
             scalar_large.candidates[2].reason == "AVX2 and FMA are required",
         "vector candidate reports missing capabilities");

  auto incomplete = scalar_capabilities();
  incomplete.detection_complete = false;
  const auto incomplete_plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(32, 32, 32), incomplete);
  expect(incomplete_plan.candidates[2].reason ==
             "AVX2/FMA discovery is incomplete",
         "incomplete feature discovery fails closed with a reason");
  auto incomplete_claims = vector_capabilities();
  incomplete_claims.detection_complete = false;
  const auto incomplete_claims_plan =
      matcore::mdslc::planner::plan_cpu_gemm_v1(
          problem(32, 32, 32), incomplete_claims);
  expect(!incomplete_claims_plan.candidates[2].legal &&
             incomplete_claims_plan.candidates[2].reason ==
                 "AVX2/FMA discovery is incomplete",
         "incomplete discovery rejects claimed vector features");

  auto non_x86 = vector_capabilities();
  non_x86.architecture = CpuArchitectureV1::aarch64;
  const auto non_x86_plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(32, 32, 32), non_x86);
  expect_selected(non_x86_plan, CpuGemmVariantV1::tiled,
                  "non-x86 synthetic capabilities");
  expect(!non_x86_plan.candidates[2].legal &&
             non_x86_plan.candidates[2].reason ==
                 "compiler-vectorized candidate requires x86_64",
         "non-x86 vector request has an explicit reason");

  auto narrow_vector = vector_capabilities();
  narrow_vector.usable_vector_bits = 128;
  const auto narrow_vector_plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(32, 32, 32), narrow_vector);
  expect(!narrow_vector_plan.candidates[2].legal &&
             narrow_vector_plan.candidates[2].reason ==
                 "compiler-vectorized candidate requires 256-bit vectors",
         "narrow vectors reject the AVX2 candidate with a reason");

  auto no_scalar = scalar_capabilities();
  no_scalar.features = 0;
  const auto no_plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(4, 4, 4), no_scalar);
  expect(no_plan.status == CpuPlanStatusV1::no_legal_variant,
         "missing portable scalar capability yields no plan");
  expect(no_plan.candidates[0].reason ==
             "portable scalar f32 capability is required" &&
             no_plan.candidates[1].reason ==
                 "portable scalar f32 capability is required",
         "scalar candidates explain their missing capability");
  const auto forced_illegal = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(4, 4, 4), scalar_capabilities(),
      CpuGemmRequestV1::force_compiler_vectorized);
  expect(forced_illegal.status == CpuPlanStatusV1::forced_variant_illegal,
         "forced illegal variant fails without fallback");
  expect(forced_illegal.selected_id.empty(),
         "forced illegal variant emits no selected ID");
  std::array<float, 4> forced_lhs{1.0F, 2.0F, 3.0F, 4.0F};
  std::array<float, 4> forced_rhs{5.0F, 6.0F, 7.0F, 8.0F};
  std::array<float, 4> forced_output{71.0F, 71.0F, 71.0F, 71.0F};
  const auto forced_output_before = forced_output;
  expect(!matcore::mdslc::planner::execute_cpu_gemm_plan_v1(
             forced_illegal, forced_lhs.data(), forced_rhs.data(),
             forced_output.data()),
         "forced illegal plan cannot execute a fallback");
  expect(forced_output == forced_output_before,
         "failed forced execution leaves output unchanged");

  auto wrong_type = problem(4, 4, 4);
  wrong_type.element_type =
      matcore::mdslc::planner::CpuScalarTypeV1::invalid;
  const auto invalid =
      matcore::mdslc::planner::plan_cpu_gemm_v1(wrong_type, vector);
  expect(invalid.status == CpuPlanStatusV1::invalid_problem,
         "unsupported planner dtype is rejected");
  expect(invalid.selection_reason == "variant supports only f32 elements",
         "invalid planner problem is actionable");
  auto invalid_alignment = problem(4, 4, 4, 6);
  const auto invalid_alignment_plan =
      matcore::mdslc::planner::plan_cpu_gemm_v1(invalid_alignment, vector);
  expect(invalid_alignment_plan.status == CpuPlanStatusV1::invalid_problem &&
             invalid_alignment_plan.selection_reason ==
                 "minimum alignment must be a power of two",
         "non-power-of-two alignment contract is rejected");

  auto incompatible_capabilities = vector_capabilities();
  incompatible_capabilities.version = 99;
  const auto incompatible_capabilities_plan =
      matcore::mdslc::planner::plan_cpu_gemm_v1(
          problem(4, 4, 4), incompatible_capabilities);
  expect(incompatible_capabilities_plan.status ==
             CpuPlanStatusV1::invalid_capabilities,
         "unsupported capability version is rejected");
  for (const auto &candidate : incompatible_capabilities_plan.candidates) {
    expect(candidate.reason == "CPU capability record version is unsupported",
           "every candidate explains an unsupported capability version");
  }

  const auto saturated = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(std::numeric_limits<std::int64_t>::max(),
              std::numeric_limits<std::int64_t>::max(),
              std::numeric_limits<std::int64_t>::max()),
      vector_capabilities());
  expect_selected(saturated, CpuGemmVariantV1::compiler_vectorized,
                  "saturating cost plan");
  for (const auto &candidate : saturated.candidates) {
    expect(candidate.estimated_cost == std::numeric_limits<std::uint64_t>::max(),
           "large planner costs saturate instead of overflowing");
  }
  expect(saturated.selection_reason ==
             "lowest deterministic cost; ties use priority then registry order",
         "saturating ties retain the documented deterministic rule");

  const auto deterministic_a = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(33, 35, 37), vector);
  const auto deterministic_b = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(33, 35, 37), vector);
  expect(deterministic_a.selected_id == deterministic_b.selected_id,
         "repeated planning selects the same stable ID");
  for (std::size_t index = 0; index < deterministic_a.candidates.size(); ++index)
    expect(deterministic_a.candidates[index].estimated_cost ==
               deterministic_b.candidates[index].estimated_cost,
           "repeated planning produces identical candidate costs");

  std::array<char, 2048> diagnostic_a{};
  std::array<char, 2048> diagnostic_b{};
  const auto required_a = matcore::mdslc::planner::format_cpu_gemm_plan_v1(
      deterministic_a, diagnostic_a.data(), diagnostic_a.size());
  const auto required_b = matcore::mdslc::planner::format_cpu_gemm_plan_v1(
      deterministic_b, diagnostic_b.data(), diagnostic_b.size());
  expect(required_a == required_b && diagnostic_a == diagnostic_b,
         "human-readable plan diagnostic is deterministic");
  const std::string_view diagnostic(diagnostic_a.data());
  expect(diagnostic.find("arch=x86_64") != std::string_view::npos,
         "diagnostic records architecture");
  expect(diagnostic.find("detection_complete=true") != std::string_view::npos,
         "diagnostic records discovery completeness");
  expect(diagnostic.find("features=[portable-scalar-f32,avx2,fma]") !=
             std::string_view::npos,
         "diagnostic records features in canonical order");
  expect(diagnostic.find("vector_bits=256") != std::string_view::npos,
         "diagnostic records vector width");
  expect(diagnostic.find("request=automatic status=selected") !=
             std::string_view::npos,
         "diagnostic records request and status");
  expect(diagnostic.find("selected=cpu.compiler-vectorized") !=
             std::string_view::npos,
         "diagnostic records the selected variant");
  expect(diagnostic.find("candidates=[") != std::string_view::npos,
         "diagnostic records every candidate decision");
  std::array<char, 16> truncated{};
  const auto truncated_required =
      matcore::mdslc::planner::format_cpu_gemm_plan_v1(
          deterministic_a, truncated.data(), truncated.size());
  expect(truncated_required == required_a && truncated.back() == '\0',
         "diagnostic reports required length and safely truncates");
  const auto null_required = matcore::mdslc::planner::format_cpu_gemm_plan_v1(
      deterministic_a, nullptr, 16);
  expect(null_required == required_a,
         "diagnostic accepts a null sizing-only output buffer");
}

void run_forced_variant(CpuGemmVariantV1 variant, CpuGemmRequestV1 request,
                        std::int64_t m, std::int64_t k, std::int64_t n) {
  std::vector<float> a(static_cast<std::size_t>(m * k));
  std::vector<float> b(static_cast<std::size_t>(k * n));
  std::vector<float> c(static_cast<std::size_t>(m * n), -9.0F);
  std::vector<double> oracle(static_cast<std::size_t>(m * n), 0.0);
  for (std::size_t index = 0; index < a.size(); ++index)
    a[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 4.0F;
  for (std::size_t index = 0; index < b.size(); ++index)
    b[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 8.0F;
  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t p = 0; p < k; ++p)
        oracle[static_cast<std::size_t>(i * n + j)] +=
            static_cast<double>(a[static_cast<std::size_t>(i * k + p)]) *
            static_cast<double>(b[static_cast<std::size_t>(p * n + j)]);
  const auto plan = matcore::mdslc::planner::plan_cpu_gemm_v1(
      problem(m, k, n, alignof(float)), vector_capabilities(), request);
  expect_selected(plan, variant, "forced kernel");
  expect(matcore::mdslc::planner::execute_cpu_gemm_plan_v1(
             plan, a.data(), b.data(), c.data()),
         "forced legal kernel executes");
  for (std::size_t index = 0; index < c.size(); ++index)
    expect(std::fabs(static_cast<double>(c[index]) - oracle[index]) < 1.0e-4,
           "forced kernel matches independent oracle");
}

void forced_variant_correctness() {
  run_forced_variant(CpuGemmVariantV1::reference,
                     CpuGemmRequestV1::force_reference, 5, 9, 7);
  run_forced_variant(CpuGemmVariantV1::tiled,
                     CpuGemmRequestV1::force_tiled, 33, 65, 37);
  const auto detected =
      matcore::mdslc::planner::discover_cpu_capabilities_v1();
  if (detected.architecture == CpuArchitectureV1::x86_64 &&
      detected.detection_complete &&
      matcore::mdslc::planner::has_feature(detected, CpuFeatureV1::avx2) &&
      matcore::mdslc::planner::has_feature(detected, CpuFeatureV1::fma) &&
      detected.usable_vector_bits >= 256) {
    run_forced_variant(CpuGemmVariantV1::compiler_vectorized,
                       CpuGemmRequestV1::force_compiler_vectorized, 19, 35, 13);
  }
}

void runtime_plan_report() {
  constexpr std::int64_t m = 17;
  constexpr std::int64_t k = 19;
  constexpr std::int64_t n = 23;
  std::vector<float> a(static_cast<std::size_t>(m * k), 1.0F);
  std::vector<float> b(static_cast<std::size_t>(k * n), 1.0F);
  std::vector<float> c(static_cast<std::size_t>(m * n), 71.0F);
  const auto output_before = c;
  auto out = desc(c.data(), m, n, MATCORE_MUTABILITY_READ_WRITE_V0);
  auto lhs = desc(a.data(), m, k);
  auto rhs = desc(b.data(), k, n);
  const auto p = policy();
  matcore_cpu_gemm_plan_report_v1 report{};
  report.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V1;
  report.struct_size = sizeof(report);
  const auto result =
      matcore_runtime_plan_gemm_f32_v1(&out, &lhs, &rhs, &p, &report);
  expect(result.code == MATCORE_STATUS_OK_V0,
         "runtime plan report succeeds for valid GEMM");
  expect(c == output_before,
         "planning does not modify output");
  expect(report.abi_version == MATCORE_RUNTIME_PLAN_ABI_VERSION_V1 &&
             report.struct_size == sizeof(report) && report.planner_version == 1,
         "runtime plan report is self-versioned");
  expect(report.request == MATCORE_CPU_PLAN_REQUEST_AUTOMATIC_V1 &&
             report.plan_status == MATCORE_CPU_PLAN_STATUS_SELECTED_V1,
         "runtime report exposes request and plan status");
  expect(report.m == m && report.n == n && report.k == k,
         "runtime report exposes canonical M/N/K");
  expect(report.candidate_count ==
             MATCORE_RUNTIME_CPU_GEMM_CANDIDATE_COUNT_V1,
         "runtime report exposes all registry candidates");
  expect(report.selected_stable_id != nullptr &&
             report.selection_reason != nullptr,
         "runtime report exposes process-lifetime selection diagnostics");
  expect(text_or_empty(report.selection_reason).find(
             "lowest deterministic cost") != std::string_view::npos,
         "runtime report exposes a human-readable selection reason");
  for (std::uint32_t index = 0; index < report.candidate_count; ++index) {
    expect(report.candidates[index].stable_id != nullptr &&
               report.candidates[index].reason != nullptr,
           "runtime report candidate has stable diagnostics");
    expect(report.candidates[index].legal != 0 ||
               report.candidates[index].estimated_cost ==
                   std::numeric_limits<std::uint64_t>::max(),
           "illegal candidate has sentinel cost");
  }
  expect(report.feature_bits & MATCORE_CPU_FEATURE_PORTABLE_SCALAR_F32_V1,
         "runtime report exposes portable scalar capability");
  expect(report.minimum_alignment_bytes >= alignof(float),
         "runtime report exposes validated alignment metadata");
#if defined(__x86_64__)
  expect(report.architecture == MATCORE_CPU_ARCHITECTURE_X86_64_V1,
         "runtime report exposes detected x86_64 architecture");
#endif

  matcore_cpu_gemm_plan_report_v1 repeated_report{};
  repeated_report.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V1;
  repeated_report.struct_size = sizeof(repeated_report);
  expect(matcore_runtime_plan_gemm_f32_v1(&out, &lhs, &rhs, &p,
                                          &repeated_report)
                 .code == MATCORE_STATUS_OK_V0,
         "repeated runtime planning succeeds");
  expect(text_or_empty(repeated_report.selected_stable_id) ==
             text_or_empty(report.selected_stable_id) &&
             text_or_empty(repeated_report.selection_reason) ==
                 text_or_empty(report.selection_reason),
         "repeated runtime planning has stable selected metadata");
  for (std::uint32_t index = 0; index < report.candidate_count; ++index) {
    expect(text_or_empty(repeated_report.candidates[index].stable_id) ==
               text_or_empty(report.candidates[index].stable_id) &&
               repeated_report.candidates[index].legal ==
                   report.candidates[index].legal &&
               repeated_report.candidates[index].estimated_cost ==
                   report.candidates[index].estimated_cost &&
               text_or_empty(repeated_report.candidates[index].reason) ==
                   text_or_empty(report.candidates[index].reason),
           "repeated runtime planning has stable candidate diagnostics");
  }

  matcore_cpu_gemm_plan_report_v1 bad_report{};
  bad_report.abi_version = 99;
  bad_report.struct_size = sizeof(bad_report);
  const auto bad_abi_before = bad_report;
  expect(matcore_runtime_plan_gemm_f32_v1(&out, &lhs, &rhs, &p, &bad_report)
             .code == MATCORE_STATUS_ABI_MISMATCH_V0,
         "runtime plan report rejects ABI mismatch");
  expect(std::memcmp(&bad_report, &bad_abi_before, sizeof(bad_report)) == 0,
         "rejected report ABI leaves caller storage unchanged");
  bad_report = {};
  bad_report.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V1;
  bad_report.struct_size = sizeof(bad_report);
  bad_report.m = 1;
  const auto dirty_report_before = bad_report;
  expect(matcore_runtime_plan_gemm_f32_v1(&out, &lhs, &rhs, &p, &bad_report)
             .code == MATCORE_STATUS_INVALID_ARGUMENT_V0,
         "runtime plan report rejects dirty output fields");
  expect(std::memcmp(&bad_report, &dirty_report_before, sizeof(bad_report)) ==
             0,
         "dirty report rejection leaves caller storage unchanged");
  auto invalid_lhs = lhs;
  invalid_lhs.dtype = MATCORE_DTYPE_F64_V0;
  matcore_cpu_gemm_plan_report_v1 invalid_problem_report{};
  invalid_problem_report.abi_version = MATCORE_RUNTIME_PLAN_ABI_VERSION_V1;
  invalid_problem_report.struct_size = sizeof(invalid_problem_report);
  const auto invalid_problem_before = invalid_problem_report;
  expect(matcore_runtime_plan_gemm_f32_v1(
             &out, &invalid_lhs, &rhs, &p, &invalid_problem_report)
             .code == MATCORE_STATUS_UNSUPPORTED_DTYPE_V0,
         "runtime plan query preserves GEMM validation status ordering");
  expect(std::memcmp(&invalid_problem_report, &invalid_problem_before,
                     sizeof(invalid_problem_report)) == 0 &&
             c == output_before,
         "failed plan queries leave report and output unchanged");
  expect(matcore_runtime_plan_gemm_f32_v1(&out, &lhs, &rhs, &p, nullptr).code ==
             MATCORE_STATUS_INVALID_ARGUMENT_V0,
         "runtime plan report rejects null output");
}

void expect_error(matcore_status_code_v0 wanted,
                  const matcore_tensor_desc_v0 &out,
                  const matcore_tensor_desc_v0 &lhs,
                  const matcore_tensor_desc_v0 &rhs,
                  const matcore_policy_v0 &p, std::string_view name) {
  const auto result = matcore_runtime_gemm_f32_v0(&out, &lhs, &rhs, &p);
  expect(result.code == wanted, name);
  expect(result.message != nullptr, "errors carry a static diagnostic");
}

void negative_cases() {
  std::vector<float> a(6), b(6), c(4, 17.0F);
  auto out = desc(c.data(), 2, 2, MATCORE_MUTABILITY_READ_WRITE_V0);
  auto lhs = desc(a.data(), 2, 3);
  auto rhs = desc(b.data(), 3, 2);
  auto p = policy();

  expect(matcore_runtime_gemm_f32_v0(nullptr, &lhs, &rhs, &p).code ==
             MATCORE_STATUS_INVALID_ARGUMENT_V0,
         "null descriptor is rejected");
  auto changed = out;
  changed.abi_version = 1;
  expect_error(MATCORE_STATUS_ABI_MISMATCH_V0, changed, lhs, rhs, p,
               "ABI mismatch is rejected");
  changed = out;
  changed.reserved[0] = 1;
  expect_error(MATCORE_STATUS_INVALID_ARGUMENT_V0, changed, lhs, rhs, p,
               "nonzero tensor reserved fields are rejected");
  auto bad_policy = p;
  bad_policy.struct_size = 0;
  expect_error(MATCORE_STATUS_ABI_MISMATCH_V0, out, lhs, rhs, bad_policy,
               "policy ABI mismatch is rejected");
  bad_policy = p;
  bad_policy.reserved[0] = 1;
  expect_error(MATCORE_STATUS_INVALID_ARGUMENT_V0, out, lhs, rhs, bad_policy,
               "nonzero policy reserved fields are rejected");
  changed = lhs;
  changed.data = nullptr;
  expect_error(MATCORE_STATUS_INVALID_ARGUMENT_V0, out, changed, rhs, p,
               "null tensor data is rejected");
  changed = lhs;
  changed.dtype = MATCORE_DTYPE_F64_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_DTYPE_V0, out, changed, rhs, p,
               "unsupported dtype is rejected");
  changed = lhs;
  changed.rank = 1;
  expect_error(MATCORE_STATUS_UNSUPPORTED_RANK_V0, out, changed, rhs, p,
               "wrong rank is rejected");
  changed = lhs;
  changed.dims[0] = 0;
  expect_error(MATCORE_STATUS_INVALID_SHAPE_V0, out, changed, rhs, p,
               "nonpositive shape is rejected");
  changed = lhs;
  changed.strides[0] += 1;
  expect_error(MATCORE_STATUS_UNSUPPORTED_LAYOUT_V0, out, changed, rhs, p,
               "noncontiguous layout is rejected");
  changed = out;
  changed.mutability = MATCORE_MUTABILITY_READ_ONLY_V0;
  expect_error(MATCORE_STATUS_OUTPUT_NOT_MUTABLE_V0, changed, lhs, rhs, p,
               "const output is rejected");
  changed = lhs;
  changed.mutability = MATCORE_MUTABILITY_INVALID_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_MUTABILITY_V0, out, changed, rhs, p,
               "invalid mutability is rejected");
  changed = lhs;
  changed.memory_space = MATCORE_MEMORY_SPACE_INVALID_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0, out, changed, rhs, p,
               "unknown memory space is rejected");
  changed = rhs;
  changed.memory_space = MATCORE_MEMORY_SPACE_CUDA_DEVICE_V0;
  expect_error(MATCORE_STATUS_MIXED_MEMORY_SPACES_V0, out, lhs, changed, p,
               "mixed residency is rejected");
  auto device_out = out;
  auto device_lhs = lhs;
  auto device_rhs = rhs;
  device_out.memory_space = device_lhs.memory_space = device_rhs.memory_space =
      MATCORE_MEMORY_SPACE_CUDA_DEVICE_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_MEMORY_SPACE_V0, device_out,
               device_lhs, device_rhs, p, "device tensors are rejected");
  bad_policy = p;
  bad_policy.target = MATCORE_TARGET_CUDA_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_TARGET_V0, out, lhs, rhs, bad_policy,
               "non-CPU target is rejected");
  bad_policy = p;
  bad_policy.fallback = MATCORE_FALLBACK_ALLOW_V0;
  expect_error(MATCORE_STATUS_UNSUPPORTED_FALLBACK_V0, out, lhs, rhs,
               bad_policy, "fallback other than error is rejected");
  changed = rhs;
  changed.dims[0] = 2;
  changed.strides[0] = 2;
  expect_error(MATCORE_STATUS_SHAPE_MISMATCH_V0, out, lhs, changed, p,
               "shape mismatch is rejected");
  auto alias_out = desc(a.data(), 2, 2, MATCORE_MUTABILITY_READ_WRITE_V0);
  expect_error(MATCORE_STATUS_ALIAS_VIOLATION_V0, alias_out, lhs, rhs, p,
               "output/input alias is rejected");
  changed = lhs;
  changed.data = static_cast<void *>(reinterpret_cast<char *>(a.data()) + 1);
  expect_error(MATCORE_STATUS_INVALID_ALIGNMENT_V0, out, changed, rhs, p,
               "misaligned f32 storage is rejected");
  auto huge_out = out;
  auto huge_lhs = lhs;
  huge_lhs.dims[0] = std::numeric_limits<std::int64_t>::max();
  huge_out.dims[0] = huge_lhs.dims[0];
  expect_error(MATCORE_STATUS_ARITHMETIC_OVERFLOW_V0, huge_out, huge_lhs, rhs,
               p, "overflowing byte range is rejected");
  expect(c == std::vector<float>(4, 17.0F),
         "failing calls do not modify output");
}

}  // namespace

int main() {
  run_shape(1, 1, 1);
  run_shape(2, 3, 2);
  run_shape(3, 2, 4);
  run_shape(16, 16, 16);
  run_shape(33, 35, 37);
  run_shape(64, 7, 19);
  run_aligned_shape();
  planner_contract();
  forced_variant_correctness();
  runtime_plan_report();
  negative_cases();
  if (failures != 0) return 1;
  std::cout << "runtime CPU GEMM v0: all tests passed\n";
  return 0;
}
