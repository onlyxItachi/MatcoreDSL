#ifndef MATCORE_MDSLC_RUNTIME_CPU_PACKED_B_FORMAT_H
#define MATCORE_MDSLC_RUNTIME_CPU_PACKED_B_FORMAT_H

#include "cpu_gemm_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace matcore::mdslc::runtime::detail {

inline constexpr std::uint64_t kCpuPackedBProvenanceSeedV1 =
    UINT64_C(0x4d43504241563131);

inline std::uint64_t cpu_mix_packed_b_provenance_v1(
    std::uint64_t state, std::uint64_t value) noexcept {
  state ^= value + UINT64_C(0x9e3779b97f4a7c15) + (state << 6U) +
           (state >> 2U);
  return state;
}

inline std::uint64_t cpu_packed_b_provenance_v1(
    const CpuPackedBViewV1 &view) noexcept {
  std::uint64_t result = kCpuPackedBProvenanceSeedV1;
  result = cpu_mix_packed_b_provenance_v1(
      result, static_cast<std::uint64_t>(
                  reinterpret_cast<std::uintptr_t>(view.source_data)));
  result = cpu_mix_packed_b_provenance_v1(
      result, static_cast<std::uint64_t>(
                  reinterpret_cast<std::uintptr_t>(view.packed_data)));
  result = cpu_mix_packed_b_provenance_v1(
      result, static_cast<std::uint64_t>(view.storage_bytes));
  result = cpu_mix_packed_b_provenance_v1(
      result, static_cast<std::uint64_t>(view.packed_elements));
  result = cpu_mix_packed_b_provenance_v1(
      result, static_cast<std::uint64_t>(view.k));
  result = cpu_mix_packed_b_provenance_v1(
      result, static_cast<std::uint64_t>(view.n));
  result = cpu_mix_packed_b_provenance_v1(result, view.kc);
  result = cpu_mix_packed_b_provenance_v1(result, view.nc);
  result = cpu_mix_packed_b_provenance_v1(result, view.nr);
  return result;
}

inline std::size_t cpu_pack_b_block_v1(
    const float *rhs, std::size_t leading_dimension, std::size_t k_begin,
    std::size_t column_begin, std::size_t depth, std::size_t columns,
    float *packed_b) noexcept {
  const std::size_t padded_columns =
      (columns / kCpuPackedGemmNrV1 +
       (columns % kCpuPackedGemmNrV1 != 0 ? 1U : 0U)) *
      kCpuPackedGemmNrV1;
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

inline std::size_t cpu_pack_b_column_panel_v1(
    const float *rhs, std::size_t leading_dimension, std::size_t k,
    std::size_t column_begin, std::size_t columns,
    float *packed_destination) noexcept {
  std::size_t packed_elements = 0;
  for (std::size_t k_begin = 0; k_begin < k;
       k_begin += kCpuPackedGemmKcV1) {
    const std::size_t depth = std::min(kCpuPackedGemmKcV1, k - k_begin);
    packed_elements += cpu_pack_b_block_v1(
        rhs, leading_dimension, k_begin, column_begin, depth, columns,
        packed_destination + packed_elements);
  }
  return packed_elements;
}

inline CpuPackedBViewV1 cpu_make_packed_b_view_v1(
    const planner::CpuGemmProblemV1 &problem, const float *rhs,
    const float *packed_data, std::size_t storage_bytes,
    std::size_t packed_elements) noexcept {
  CpuPackedBViewV1 result;
  result.version = kCpuPackedGemmBackendVersionV1;
  result.struct_size = sizeof(CpuPackedBViewV1);
  result.source_data = rhs;
  result.packed_data = packed_data;
  result.storage_bytes = storage_bytes;
  result.packed_elements = packed_elements;
  result.k = problem.k;
  result.n = problem.n;
  result.kc = kCpuPackedGemmKcV1;
  result.nc = kCpuPackedGemmNcV1;
  result.nr = kCpuPackedGemmNrV1;
  result.provenance = cpu_packed_b_provenance_v1(result);
  return result;
}

inline bool cpu_validate_packed_b_view_metadata_v1(
    const planner::CpuGemmProblemV1 &problem, const CpuPackedBViewV1 &view,
    std::size_t required_storage_bytes,
    std::size_t required_packed_elements) noexcept {
  return view.version == kCpuPackedGemmBackendVersionV1 &&
         view.struct_size == sizeof(CpuPackedBViewV1) &&
         view.source_data != nullptr && view.packed_data != nullptr &&
         view.k == problem.k && view.n == problem.n &&
         view.kc == kCpuPackedGemmKcV1 &&
         view.nc == kCpuPackedGemmNcV1 &&
         view.nr == kCpuPackedGemmNrV1 && view.reserved0 == 0 &&
         reinterpret_cast<std::uintptr_t>(view.packed_data) %
                 kCpuPackedGemmWorkspaceAlignmentV1 ==
             0 &&
         view.storage_bytes >= required_storage_bytes &&
         view.packed_elements == required_packed_elements &&
         view.provenance == cpu_packed_b_provenance_v1(view);
}

}  // namespace matcore::mdslc::runtime::detail

#endif
