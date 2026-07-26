#include "cpu_gemm_backend.h"
#include "cpu_capability_v2.h"
#include "cpu_packed_b_format.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__clang__) || defined(__GNUC__))
#define MATCORE_MDSLC_PACKED_AVX2_TARGET \
  __attribute__((target("avx2,fma"), noinline))
#define MATCORE_MDSLC_PACKED_AVX2_COMPILED 1
#else
#define MATCORE_MDSLC_PACKED_AVX2_TARGET
#define MATCORE_MDSLC_PACKED_AVX2_COMPILED 0
#endif

namespace matcore::mdslc::runtime {

namespace detail {

// Production packed execution reaches this private symbol only after the
// descriptor, workspace, and complete 4x16 tile contracts have been
// validated. Keeping it separate prevents full tiles from paying the checked
// edge wrapper's stack-buffer setup on every microkernel call.
extern "C" void matcore_cpu_packed_avx2_4x16_full_microkernel_f32_v2(
    const float *packed_a, const float *packed_b, std::size_t k,
    float *output, std::size_t output_stride, bool accumulate) noexcept;

}  // namespace detail

namespace {

struct ByteSpan {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};

bool checked_multiply(std::size_t lhs, std::size_t rhs,
                      std::size_t *result) noexcept {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool checked_round_up(std::size_t value, std::size_t multiple,
                      std::size_t *result) noexcept {
  if (result == nullptr || multiple == 0) return false;
  const std::size_t remainder = value % multiple;
  if (remainder == 0) {
    *result = value;
    return true;
  }
  const std::size_t increment = multiple - remainder;
  if (increment > std::numeric_limits<std::size_t>::max() - value) return false;
  *result = value + increment;
  return true;
}

bool make_span(const void *pointer, std::size_t bytes, ByteSpan *span) noexcept {
  if (pointer == nullptr || span == nullptr) return false;
  const auto address = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
  span->begin = address;
  span->end = address + bytes;
  return true;
}

bool make_matrix_span(const void *pointer, std::size_t rows,
                      std::size_t columns, ByteSpan *span) noexcept {
  std::size_t elements = 0;
  std::size_t bytes = 0;
  return checked_multiply(rows, columns, &elements) &&
         checked_multiply(elements, sizeof(float), &bytes) &&
         make_span(pointer, bytes, span);
}

bool overlaps(const ByteSpan &lhs, const ByteSpan &rhs) noexcept {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

bool pointer_has_alignment(const void *pointer,
                           std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0 &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

CpuPackedGemmStatusV1 dimensions(
    const planner::CpuGemmProblemV1 &problem, std::size_t *m, std::size_t *n,
    std::size_t *k) noexcept {
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  const auto status = cpu_packed_avx2_workspace_requirements_v1(
      problem, CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
      &requirements);
  if (status != CpuPackedGemmStatusV1::success) return status;
  *m = static_cast<std::size_t>(problem.m);
  *n = static_cast<std::size_t>(problem.n);
  *k = static_cast<std::size_t>(problem.k);
  return CpuPackedGemmStatusV1::success;
}

CpuPackedGemmStatusV1 validate_tensor_contract(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, std::size_t m, std::size_t n,
    std::size_t k, ByteSpan *lhs_span, ByteSpan *rhs_span,
    ByteSpan *out_span) noexcept {
  if (lhs == nullptr || rhs == nullptr || out == nullptr) {
    return CpuPackedGemmStatusV1::null_pointer;
  }
  const std::size_t declared_alignment = problem.minimum_alignment_bytes;
  if (!pointer_has_alignment(lhs, alignof(float)) ||
      !pointer_has_alignment(rhs, alignof(float)) ||
      !pointer_has_alignment(out, alignof(float)) ||
      !pointer_has_alignment(lhs, declared_alignment) ||
      !pointer_has_alignment(rhs, declared_alignment) ||
      !pointer_has_alignment(out, declared_alignment)) {
    return CpuPackedGemmStatusV1::invalid_pointer_alignment;
  }
  if (!make_matrix_span(lhs, m, k, lhs_span) ||
      !make_matrix_span(rhs, k, n, rhs_span) ||
      !make_matrix_span(out, m, n, out_span)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }
  if (overlaps(*out_span, *lhs_span) || overlaps(*out_span, *rhs_span)) {
    return CpuPackedGemmStatusV1::alias_violation;
  }
  return CpuPackedGemmStatusV1::success;
}

CpuPackedGemmStatusV1 validate_workspace(
    void *workspace, std::size_t workspace_bytes,
    const CpuPackedGemmWorkspaceRequirementsV1 &requirements,
    const ByteSpan *forbidden_spans, std::size_t forbidden_count,
    ByteSpan *workspace_span) noexcept {
  if (workspace == nullptr) return CpuPackedGemmStatusV1::null_pointer;
  if (!pointer_has_alignment(workspace, requirements.alignment_bytes)) {
    return CpuPackedGemmStatusV1::workspace_misaligned;
  }
  if (workspace_bytes < requirements.total_bytes) {
    return CpuPackedGemmStatusV1::workspace_insufficient;
  }
  if (!make_span(workspace, requirements.total_bytes, workspace_span)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }
  for (std::size_t index = 0; index < forbidden_count; ++index) {
    if (overlaps(*workspace_span, forbidden_spans[index])) {
      return CpuPackedGemmStatusV1::alias_violation;
    }
  }
  return CpuPackedGemmStatusV1::success;
}

void pack_a_block(const float *lhs, std::size_t leading_dimension,
                  std::size_t row_begin, std::size_t k_begin,
                  std::size_t rows, std::size_t depth,
                  float *packed_a) noexcept {
  std::size_t padded_rows = 0;
  (void)checked_round_up(rows, kCpuPackedGemmMrV1, &padded_rows);
  std::size_t destination = 0;
  for (std::size_t row = 0; row < padded_rows;
       row += kCpuPackedGemmMrV1) {
    for (std::size_t p = 0; p < depth; ++p) {
      for (std::size_t lane = 0; lane < kCpuPackedGemmMrV1; ++lane) {
        packed_a[destination++] =
            row + lane < rows
                ? lhs[(row_begin + row + lane) * leading_dimension + k_begin +
                      p]
                : 0.0F;
      }
    }
  }
}

void compute_block(const planner::CpuGemmProblemV1 &problem,
                   const float *lhs, float *out, std::size_t row_begin,
                   std::size_t column_begin, std::size_t k_begin,
                   std::size_t rows, std::size_t columns, std::size_t depth,
                   float *packed_a, const float *packed_b) noexcept {
  const std::size_t lda = static_cast<std::size_t>(problem.k);
  const std::size_t ldc = static_cast<std::size_t>(problem.n);
  pack_a_block(lhs, lda, row_begin, k_begin, rows, depth, packed_a);
  for (std::size_t row = 0; row < rows; row += kCpuPackedGemmMrV1) {
    const auto tile_rows = static_cast<std::uint32_t>(
        std::min(kCpuPackedGemmMrV1, rows - row));
    const float *a_panel =
        packed_a + (row / kCpuPackedGemmMrV1) * depth * kCpuPackedGemmMrV1;
    for (std::size_t column = 0; column < columns;
         column += kCpuPackedGemmNrV1) {
      const auto tile_columns = static_cast<std::uint32_t>(
          std::min(kCpuPackedGemmNrV1, columns - column));
      const float *b_panel = packed_b +
                             (column / kCpuPackedGemmNrV1) * depth *
                                 kCpuPackedGemmNrV1;
      detail::matcore_cpu_packed_avx2_4x16_microkernel_f32_v1(
          a_panel, b_panel, depth,
          out + (row_begin + row) * ldc + column_begin + column, ldc,
          tile_rows, tile_columns, k_begin != 0);
    }
  }
}

CpuPackedGemmStatusV1 validate_prepacked_view(
    const planner::CpuGemmProblemV1 &problem, const CpuPackedBViewV1 &view,
    ByteSpan *source_span, ByteSpan *packed_span) noexcept {
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  const auto status =
      cpu_packed_avx2_prepacked_b_requirements_v1(problem, &requirements);
  if (status != CpuPackedGemmStatusV1::success) return status;
  if (!detail::cpu_validate_packed_b_view_metadata_v1(
          problem, view, requirements.total_bytes,
          requirements.packed_b_bytes / sizeof(float))) {
    return CpuPackedGemmStatusV1::invalid_prepacked_b;
  }
  const auto k = static_cast<std::size_t>(problem.k);
  const auto n = static_cast<std::size_t>(problem.n);
  if (!make_matrix_span(view.source_data, k, n, source_span) ||
      !make_span(view.packed_data, requirements.packed_b_bytes, packed_span)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }
  if (overlaps(*source_span, *packed_span)) {
    return CpuPackedGemmStatusV1::invalid_prepacked_b;
  }
  return CpuPackedGemmStatusV1::success;
}

}  // namespace

bool cpu_packed_avx2_build_available_v1() noexcept {
  return MATCORE_MDSLC_PACKED_AVX2_COMPILED != 0;
}

bool cpu_packed_avx2_runtime_usable_v1() noexcept {
#if MATCORE_MDSLC_PACKED_AVX2_COMPILED
  const auto capabilities =
      platform::discover_cpu_capabilities_v2();
  if (!platform::validate_cpu_capabilities_v2(capabilities)) return false;
  const auto execution_safe = [&capabilities](platform::CpuFeatureV2 feature) {
    return platform::feature_available(capabilities.hardware, feature) &&
           platform::feature_available(capabilities.os_enabled, feature) &&
           platform::feature_available(capabilities.compiler, feature);
  };
  return capabilities.architecture == platform::ArchitectureKindV1::x86_64 &&
         execution_safe(platform::CpuFeatureV2::avx2) &&
         execution_safe(platform::CpuFeatureV2::fma);
#else
  return false;
#endif
}

namespace detail {

namespace {

#if MATCORE_MDSLC_PACKED_AVX2_COMPILED
__attribute__((target("avx2,fma"), always_inline)) inline void
compute_avx2_4x16_full_tile(const float *packed_a, const float *packed_b,
                           std::size_t k, float *output,
                           std::size_t output_stride,
                           bool accumulate) noexcept {
  const __m256 zero = _mm256_setzero_ps();
  __m256 c00 = accumulate ? _mm256_loadu_ps(output) : zero;
  __m256 c01 = accumulate ? _mm256_loadu_ps(output + 8) : zero;
  __m256 c10 = accumulate ? _mm256_loadu_ps(output + output_stride) : zero;
  __m256 c11 =
      accumulate ? _mm256_loadu_ps(output + output_stride + 8) : zero;
  __m256 c20 =
      accumulate ? _mm256_loadu_ps(output + 2 * output_stride) : zero;
  __m256 c21 =
      accumulate ? _mm256_loadu_ps(output + 2 * output_stride + 8) : zero;
  __m256 c30 =
      accumulate ? _mm256_loadu_ps(output + 3 * output_stride) : zero;
  __m256 c31 =
      accumulate ? _mm256_loadu_ps(output + 3 * output_stride + 8) : zero;

  for (std::size_t p = 0; p < k; ++p) {
    const __m256 b0 = _mm256_load_ps(packed_b + p * kCpuPackedGemmNrV1);
    const __m256 b1 =
        _mm256_load_ps(packed_b + p * kCpuPackedGemmNrV1 + 8);
    const float *a = packed_a + p * kCpuPackedGemmMrV1;
    const __m256 a0 = _mm256_broadcast_ss(a);
    const __m256 a1 = _mm256_broadcast_ss(a + 1);
    const __m256 a2 = _mm256_broadcast_ss(a + 2);
    const __m256 a3 = _mm256_broadcast_ss(a + 3);
    c00 = _mm256_fmadd_ps(a0, b0, c00);
    c01 = _mm256_fmadd_ps(a0, b1, c01);
    c10 = _mm256_fmadd_ps(a1, b0, c10);
    c11 = _mm256_fmadd_ps(a1, b1, c11);
    c20 = _mm256_fmadd_ps(a2, b0, c20);
    c21 = _mm256_fmadd_ps(a2, b1, c21);
    c30 = _mm256_fmadd_ps(a3, b0, c30);
    c31 = _mm256_fmadd_ps(a3, b1, c31);
  }

  _mm256_storeu_ps(output, c00);
  _mm256_storeu_ps(output + 8, c01);
  _mm256_storeu_ps(output + output_stride, c10);
  _mm256_storeu_ps(output + output_stride + 8, c11);
  _mm256_storeu_ps(output + 2 * output_stride, c20);
  _mm256_storeu_ps(output + 2 * output_stride + 8, c21);
  _mm256_storeu_ps(output + 3 * output_stride, c30);
  _mm256_storeu_ps(output + 3 * output_stride + 8, c31);
}
#endif

}  // namespace

extern "C" MATCORE_MDSLC_PACKED_AVX2_TARGET void
matcore_cpu_packed_avx2_4x16_full_microkernel_f32_v2(
    const float *packed_a, const float *packed_b, std::size_t k,
    float *output, std::size_t output_stride, bool accumulate) noexcept {
#if MATCORE_MDSLC_PACKED_AVX2_COMPILED
  compute_avx2_4x16_full_tile(packed_a, packed_b, k, output, output_stride,
                             accumulate);
#else
  (void)packed_a;
  (void)packed_b;
  (void)k;
  (void)output;
  (void)output_stride;
  (void)accumulate;
#endif
}

extern "C" MATCORE_MDSLC_PACKED_AVX2_TARGET void
matcore_cpu_packed_avx2_4x16_microkernel_f32_v1(
    const float *packed_a, const float *packed_b, std::size_t k,
    float *output, std::size_t output_stride, std::uint32_t rows,
    std::uint32_t columns, bool accumulate) noexcept {
#if MATCORE_MDSLC_PACKED_AVX2_COMPILED
  if (packed_a == nullptr || packed_b == nullptr || output == nullptr || k == 0 ||
      rows == 0 || rows > kCpuPackedGemmMrV1 || columns == 0 ||
      columns > kCpuPackedGemmNrV1) {
    return;
  }

  const bool complete_tile =
      rows == kCpuPackedGemmMrV1 && columns == kCpuPackedGemmNrV1;
  if (complete_tile) {
    compute_avx2_4x16_full_tile(packed_a, packed_b, k, output, output_stride,
                               accumulate);
    return;
  }

  alignas(32) std::array<float, kCpuPackedGemmMrV1 * kCpuPackedGemmNrV1>
      edge{};
  if (accumulate) {
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t column = 0; column < columns; ++column)
        edge[row * kCpuPackedGemmNrV1 + column] =
            output[row * output_stride + column];
  }

  compute_avx2_4x16_full_tile(packed_a, packed_b, k, edge.data(),
                             kCpuPackedGemmNrV1, accumulate);
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t column = 0; column < columns; ++column)
      output[row * output_stride + column] =
          edge[row * kCpuPackedGemmNrV1 + column];
#else
  (void)packed_a;
  (void)packed_b;
  (void)k;
  (void)output;
  (void)output_stride;
  (void)rows;
  (void)columns;
  (void)accumulate;
#endif
}

}  // namespace detail

CpuPackedGemmStatusV1 cpu_prepare_packed_b_avx2_v1(
    const planner::CpuGemmProblemV1 &problem, const float *rhs,
    void *packed_storage, std::size_t packed_storage_bytes,
    CpuPackedBViewV1 *view) noexcept {
  if (rhs == nullptr || packed_storage == nullptr || view == nullptr) {
    return CpuPackedGemmStatusV1::null_pointer;
  }
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  auto status =
      cpu_packed_avx2_prepacked_b_requirements_v1(problem, &requirements);
  if (status != CpuPackedGemmStatusV1::success) return status;
  if (!pointer_has_alignment(rhs, alignof(float)) ||
      !pointer_has_alignment(rhs, problem.minimum_alignment_bytes)) {
    return CpuPackedGemmStatusV1::invalid_pointer_alignment;
  }
  if (!pointer_has_alignment(packed_storage,
                             kCpuPackedGemmWorkspaceAlignmentV1)) {
    return CpuPackedGemmStatusV1::workspace_misaligned;
  }
  if (packed_storage_bytes < requirements.total_bytes) {
    return CpuPackedGemmStatusV1::workspace_insufficient;
  }

  const auto k = static_cast<std::size_t>(problem.k);
  const auto n = static_cast<std::size_t>(problem.n);
  ByteSpan rhs_span;
  ByteSpan storage_span;
  if (!make_matrix_span(rhs, k, n, &rhs_span) ||
      !make_span(packed_storage, requirements.total_bytes, &storage_span)) {
    return CpuPackedGemmStatusV1::arithmetic_overflow;
  }
  if (overlaps(rhs_span, storage_span)) {
    return CpuPackedGemmStatusV1::alias_violation;
  }

  auto *destination = static_cast<float *>(packed_storage);
  std::size_t packed_elements = 0;
  for (std::size_t column = 0; column < n; column += kCpuPackedGemmNcV1) {
    const std::size_t columns = std::min(kCpuPackedGemmNcV1, n - column);
    for (std::size_t p = 0; p < k; p += kCpuPackedGemmKcV1) {
      const std::size_t depth = std::min(kCpuPackedGemmKcV1, k - p);
      packed_elements += detail::cpu_pack_b_block_v1(
          rhs, n, p, column, depth, columns, destination + packed_elements);
    }
  }

  *view = detail::cpu_make_packed_b_view_v1(
      problem, rhs, static_cast<const float *>(packed_storage),
      packed_storage_bytes, packed_elements);
  return CpuPackedGemmStatusV1::success;
}

CpuPackedGemmStatusV1 cpu_execute_packed_avx2_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes) noexcept {
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  auto status = cpu_packed_avx2_workspace_requirements_v1(
      problem, CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
      &requirements);
  if (status != CpuPackedGemmStatusV1::success) return status;
  if (!cpu_packed_avx2_runtime_usable_v1()) {
    return CpuPackedGemmStatusV1::isa_unavailable;
  }

  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  status = dimensions(problem, &m, &n, &k);
  if (status != CpuPackedGemmStatusV1::success) return status;
  ByteSpan lhs_span;
  ByteSpan rhs_span;
  ByteSpan out_span;
  status = validate_tensor_contract(problem, lhs, rhs, out, m, n, k,
                                    &lhs_span, &rhs_span, &out_span);
  if (status != CpuPackedGemmStatusV1::success) return status;
  const std::array<ByteSpan, 3> forbidden{lhs_span, rhs_span, out_span};
  ByteSpan workspace_span;
  status = validate_workspace(workspace, workspace_bytes, requirements,
                              forbidden.data(), forbidden.size(),
                              &workspace_span);
  if (status != CpuPackedGemmStatusV1::success) return status;

  auto *workspace_bytes_pointer = static_cast<std::byte *>(workspace);
  auto *packed_a = reinterpret_cast<float *>(
      workspace_bytes_pointer + requirements.packed_a_offset);
  auto *packed_b = reinterpret_cast<float *>(
      workspace_bytes_pointer + requirements.packed_b_offset);
  for (std::size_t column = 0; column < n; column += kCpuPackedGemmNcV1) {
    const std::size_t columns = std::min(kCpuPackedGemmNcV1, n - column);
    for (std::size_t p = 0; p < k; p += kCpuPackedGemmKcV1) {
      const std::size_t depth = std::min(kCpuPackedGemmKcV1, k - p);
      (void)detail::cpu_pack_b_block_v1(rhs, n, p, column, depth, columns,
                                        packed_b);
      for (std::size_t row = 0; row < m; row += kCpuPackedGemmMcV1) {
        const std::size_t rows = std::min(kCpuPackedGemmMcV1, m - row);
        compute_block(problem, lhs, out, row, column, p, rows, columns, depth,
                      packed_a, packed_b);
      }
    }
  }
  return CpuPackedGemmStatusV1::success;
}

CpuPackedGemmStatusV1 cpu_execute_packed_avx2_prepacked_b_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs, float *out,
    const CpuPackedBViewV1 &packed_b, void *workspace,
    std::size_t workspace_bytes) noexcept {
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  auto status = cpu_packed_avx2_workspace_requirements_v1(
      problem, CpuPackedGemmWorkspaceModeV1::transient_a_with_prepacked_b,
      &requirements);
  if (status != CpuPackedGemmStatusV1::success) return status;
  if (!cpu_packed_avx2_runtime_usable_v1()) {
    return CpuPackedGemmStatusV1::isa_unavailable;
  }

  std::size_t m = 0;
  std::size_t n = 0;
  std::size_t k = 0;
  status = dimensions(problem, &m, &n, &k);
  if (status != CpuPackedGemmStatusV1::success) return status;
  ByteSpan source_span;
  ByteSpan packed_span;
  status = validate_prepacked_view(problem, packed_b, &source_span, &packed_span);
  if (status != CpuPackedGemmStatusV1::success) return status;

  ByteSpan lhs_span;
  ByteSpan rhs_span;
  ByteSpan out_span;
  status = validate_tensor_contract(problem, lhs, packed_b.source_data, out, m,
                                    n, k, &lhs_span, &rhs_span, &out_span);
  if (status != CpuPackedGemmStatusV1::success) return status;
  if (rhs_span.begin != source_span.begin || rhs_span.end != source_span.end ||
      overlaps(lhs_span, packed_span) || overlaps(out_span, packed_span)) {
    return CpuPackedGemmStatusV1::alias_violation;
  }
  const std::array<ByteSpan, 4> forbidden{lhs_span, rhs_span, out_span,
                                          packed_span};
  ByteSpan workspace_span;
  status = validate_workspace(workspace, workspace_bytes, requirements,
                              forbidden.data(), forbidden.size(),
                              &workspace_span);
  if (status != CpuPackedGemmStatusV1::success) return status;

  auto *packed_a = static_cast<float *>(workspace);
  const float *packed_block = packed_b.packed_data;
  std::size_t packed_offset = 0;
  for (std::size_t column = 0; column < n; column += kCpuPackedGemmNcV1) {
    const std::size_t columns = std::min(kCpuPackedGemmNcV1, n - column);
    std::size_t padded_columns = 0;
    (void)checked_round_up(columns, kCpuPackedGemmNrV1, &padded_columns);
    for (std::size_t p = 0; p < k; p += kCpuPackedGemmKcV1) {
      const std::size_t depth = std::min(kCpuPackedGemmKcV1, k - p);
      const float *current_packed_b = packed_block + packed_offset;
      for (std::size_t row = 0; row < m; row += kCpuPackedGemmMcV1) {
        const std::size_t rows = std::min(kCpuPackedGemmMcV1, m - row);
        compute_block(problem, lhs, out, row, column, p, rows, columns, depth,
                      packed_a, current_packed_b);
      }
      packed_offset += padded_columns * depth;
    }
  }
  return CpuPackedGemmStatusV1::success;
}

}  // namespace matcore::mdslc::runtime

#undef MATCORE_MDSLC_PACKED_AVX2_TARGET
#undef MATCORE_MDSLC_PACKED_AVX2_COMPILED
