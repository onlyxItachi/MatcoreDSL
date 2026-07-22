#include "cpu_packed_avx512.h"

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
#define MATCORE_MDSLC_PACKED_AVX512_TARGET \
  __attribute__((target("avx512f,fma"), noinline))
#define MATCORE_MDSLC_PACKED_AVX512_COMPILED 1
#else
#define MATCORE_MDSLC_PACKED_AVX512_TARGET
#define MATCORE_MDSLC_PACKED_AVX512_COMPILED 0
#endif

namespace matcore::mdslc::runtime {
namespace {

// Kept identical to packed AVX2 because CpuPackedBViewV1 describes one common
// v1 packing format, not an ISA-specific blob.
inline constexpr std::uint64_t kPackedBProvenanceSeed =
    UINT64_C(0x4d43504241563131);

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

std::uint64_t mix_provenance(std::uint64_t state,
                             std::uint64_t value) noexcept {
  state ^= value + UINT64_C(0x9e3779b97f4a7c15) + (state << 6U) +
           (state >> 2U);
  return state;
}

std::uint64_t packed_b_provenance(const CpuPackedBViewV1 &view) noexcept {
  std::uint64_t result = kPackedBProvenanceSeed;
  result = mix_provenance(
      result, static_cast<std::uint64_t>(
                  reinterpret_cast<std::uintptr_t>(view.source_data)));
  result = mix_provenance(
      result, static_cast<std::uint64_t>(
                  reinterpret_cast<std::uintptr_t>(view.packed_data)));
  result = mix_provenance(result, static_cast<std::uint64_t>(view.storage_bytes));
  result =
      mix_provenance(result, static_cast<std::uint64_t>(view.packed_elements));
  result = mix_provenance(result, static_cast<std::uint64_t>(view.k));
  result = mix_provenance(result, static_cast<std::uint64_t>(view.n));
  result = mix_provenance(result, view.kc);
  result = mix_provenance(result, view.nc);
  result = mix_provenance(result, view.nr);
  return result;
}

CpuPackedGemmStatusV1 dimensions(
    const planner::CpuGemmProblemV1 &problem, std::size_t *m, std::size_t *n,
    std::size_t *k) noexcept {
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  const auto status = cpu_packed_avx512_workspace_requirements_v1(
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

std::size_t pack_b_block(const float *rhs, std::size_t leading_dimension,
                         std::size_t k_begin, std::size_t column_begin,
                         std::size_t depth, std::size_t columns,
                         float *packed_b) noexcept {
  std::size_t padded_columns = 0;
  (void)checked_round_up(columns, kCpuPackedGemmNrV1, &padded_columns);
  std::size_t destination = 0;
  for (std::size_t column = 0; column < padded_columns;
       column += kCpuPackedGemmNrV1) {
    for (std::size_t p = 0; p < depth; ++p) {
      for (std::size_t lane = 0; lane < kCpuPackedGemmNrV1; ++lane) {
        packed_b[destination++] =
            column + lane < columns
                ? rhs[(k_begin + p) * leading_dimension + column_begin +
                      column + lane]
                : 0.0F;
      }
    }
  }
  return destination;
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
      detail::matcore_cpu_packed_avx512_4x16_microkernel_f32_v1(
          a_panel, b_panel, depth,
          out + (row_begin + row) * ldc + column_begin + column, ldc,
          tile_rows, tile_columns, k_begin != 0);
    }
  }
}

CpuPackedGemmStatusV1 validate_prepacked_view(
    const planner::CpuGemmProblemV1 &problem, const CpuPackedBViewV1 &view,
    ByteSpan *source_span, ByteSpan *packed_span) noexcept {
  if (view.version != kCpuPackedGemmBackendVersionV1 ||
      view.struct_size != sizeof(CpuPackedBViewV1) ||
      view.source_data == nullptr || view.packed_data == nullptr ||
      view.k != problem.k || view.n != problem.n ||
      view.kc != kCpuPackedGemmKcV1 || view.nc != kCpuPackedGemmNcV1 ||
      view.nr != kCpuPackedGemmNrV1 || view.reserved0 != 0 ||
      !pointer_has_alignment(view.packed_data,
                             kCpuPackedGemmWorkspaceAlignmentV1) ||
      view.provenance != packed_b_provenance(view)) {
    return CpuPackedGemmStatusV1::invalid_prepacked_b;
  }

  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  const auto status =
      cpu_packed_avx512_prepacked_b_requirements_v1(problem, &requirements);
  if (status != CpuPackedGemmStatusV1::success) return status;
  if (view.storage_bytes < requirements.total_bytes ||
      view.packed_elements != requirements.packed_b_bytes / sizeof(float)) {
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

bool cpu_packed_avx512_build_available_v1() noexcept {
  return MATCORE_MDSLC_PACKED_AVX512_COMPILED != 0;
}

bool cpu_packed_avx512_runtime_usable_v1() noexcept {
#if MATCORE_MDSLC_PACKED_AVX512_COMPILED && \
    defined(__has_builtin) && __has_builtin(__builtin_cpu_supports)
  __builtin_cpu_init();
  // Compiler runtime feature discovery includes the operating-system extended
  // state gate for AVX-512. Capability model v2 remains the authoritative
  // planner gate; this check prevents a direct backend call from bypassing it.
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

CpuPackedGemmStatusV1 cpu_packed_avx512_workspace_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceModeV1 mode,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept {
  return cpu_packed_avx2_workspace_requirements_v1(problem, mode, requirements);
}

CpuPackedGemmStatusV1 cpu_packed_avx512_prepacked_b_requirements_v1(
    const planner::CpuGemmProblemV1 &problem,
    CpuPackedGemmWorkspaceRequirementsV1 *requirements) noexcept {
  return cpu_packed_avx2_prepacked_b_requirements_v1(problem, requirements);
}

namespace detail {

extern "C" MATCORE_MDSLC_PACKED_AVX512_TARGET void
matcore_cpu_packed_avx512_4x16_microkernel_f32_v1(
    const float *packed_a, const float *packed_b, std::size_t k,
    float *output, std::size_t output_stride, std::uint32_t rows,
    std::uint32_t columns, bool accumulate) noexcept {
#if MATCORE_MDSLC_PACKED_AVX512_COMPILED
  if (packed_a == nullptr || packed_b == nullptr || output == nullptr || k == 0 ||
      rows == 0 || rows > kCpuPackedGemmMrV1 || columns == 0 ||
      columns > kCpuPackedGemmNrV1) {
    return;
  }

  alignas(64) std::array<float, kCpuPackedGemmMrV1 * kCpuPackedGemmNrV1>
      edge{};
  const bool complete_tile =
      rows == kCpuPackedGemmMrV1 && columns == kCpuPackedGemmNrV1;
  float *tile = output;
  std::size_t tile_stride = output_stride;
  if (!complete_tile) {
    if (accumulate) {
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
          edge[row * kCpuPackedGemmNrV1 + column] =
              output[row * output_stride + column];
    }
    tile = edge.data();
    tile_stride = kCpuPackedGemmNrV1;
  }

  const __m512 zero = _mm512_setzero_ps();
  __m512 c0 = accumulate ? _mm512_loadu_ps(tile) : zero;
  __m512 c1 = accumulate ? _mm512_loadu_ps(tile + tile_stride) : zero;
  __m512 c2 = accumulate ? _mm512_loadu_ps(tile + 2 * tile_stride) : zero;
  __m512 c3 = accumulate ? _mm512_loadu_ps(tile + 3 * tile_stride) : zero;

  for (std::size_t p = 0; p < k; ++p) {
    const __m512 b = _mm512_load_ps(packed_b + p * kCpuPackedGemmNrV1);
    const float *a = packed_a + p * kCpuPackedGemmMrV1;
    c0 = _mm512_fmadd_ps(_mm512_set1_ps(a[0]), b, c0);
    c1 = _mm512_fmadd_ps(_mm512_set1_ps(a[1]), b, c1);
    c2 = _mm512_fmadd_ps(_mm512_set1_ps(a[2]), b, c2);
    c3 = _mm512_fmadd_ps(_mm512_set1_ps(a[3]), b, c3);
  }

  _mm512_storeu_ps(tile, c0);
  _mm512_storeu_ps(tile + tile_stride, c1);
  _mm512_storeu_ps(tile + 2 * tile_stride, c2);
  _mm512_storeu_ps(tile + 3 * tile_stride, c3);

  if (!complete_tile) {
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t column = 0; column < columns; ++column)
        output[row * output_stride + column] =
            edge[row * kCpuPackedGemmNrV1 + column];
  }
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

CpuPackedGemmStatusV1 cpu_prepare_packed_b_avx512_v1(
    const planner::CpuGemmProblemV1 &problem, const float *rhs,
    void *packed_storage, std::size_t packed_storage_bytes,
    CpuPackedBViewV1 *view) noexcept {
  if (rhs == nullptr || packed_storage == nullptr || view == nullptr) {
    return CpuPackedGemmStatusV1::null_pointer;
  }
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  auto status =
      cpu_packed_avx512_prepacked_b_requirements_v1(problem, &requirements);
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
      packed_elements += pack_b_block(rhs, n, p, column, depth, columns,
                                      destination + packed_elements);
    }
  }

  CpuPackedBViewV1 result;
  result.version = kCpuPackedGemmBackendVersionV1;
  result.struct_size = sizeof(CpuPackedBViewV1);
  result.source_data = rhs;
  result.packed_data = static_cast<const float *>(packed_storage);
  result.storage_bytes = packed_storage_bytes;
  result.packed_elements = packed_elements;
  result.k = problem.k;
  result.n = problem.n;
  result.kc = kCpuPackedGemmKcV1;
  result.nc = kCpuPackedGemmNcV1;
  result.nr = kCpuPackedGemmNrV1;
  result.provenance = packed_b_provenance(result);
  *view = result;
  return CpuPackedGemmStatusV1::success;
}

CpuPackedGemmStatusV1 cpu_execute_packed_avx512_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs,
    const float *rhs, float *out, void *workspace,
    std::size_t workspace_bytes) noexcept {
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  auto status = cpu_packed_avx512_workspace_requirements_v1(
      problem, CpuPackedGemmWorkspaceModeV1::transient_a_and_b,
      &requirements);
  if (status != CpuPackedGemmStatusV1::success) return status;
  if (!cpu_packed_avx512_runtime_usable_v1()) {
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
      (void)pack_b_block(rhs, n, p, column, depth, columns, packed_b);
      for (std::size_t row = 0; row < m; row += kCpuPackedGemmMcV1) {
        const std::size_t rows = std::min(kCpuPackedGemmMcV1, m - row);
        compute_block(problem, lhs, out, row, column, p, rows, columns, depth,
                      packed_a, packed_b);
      }
    }
  }
  return CpuPackedGemmStatusV1::success;
}

CpuPackedGemmStatusV1 cpu_execute_packed_avx512_prepacked_b_v1(
    const planner::CpuGemmProblemV1 &problem, const float *lhs, float *out,
    const CpuPackedBViewV1 &packed_b, void *workspace,
    std::size_t workspace_bytes) noexcept {
  CpuPackedGemmWorkspaceRequirementsV1 requirements;
  auto status = cpu_packed_avx512_workspace_requirements_v1(
      problem, CpuPackedGemmWorkspaceModeV1::transient_a_with_prepacked_b,
      &requirements);
  if (status != CpuPackedGemmStatusV1::success) return status;
  if (!cpu_packed_avx512_runtime_usable_v1()) {
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

#undef MATCORE_MDSLC_PACKED_AVX512_TARGET
#undef MATCORE_MDSLC_PACKED_AVX512_COMPILED
