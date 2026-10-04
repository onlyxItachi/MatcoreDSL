#ifndef MATCORE_MDSLC_CLOSED_GPU_CAPABILITY_V1_H
#define MATCORE_MDSLC_CLOSED_GPU_CAPABILITY_V1_H
#include <cstdint>
namespace matcore::mdslc::runtime::closed_host_v1::detail {
// Qualification of the initial one-thread-per-output recipe, NOT mathematical
// shape semantics, a performance crossover, or a target-independent schedule.
// Check before allocation and zero-reduction shortcuts as well as at the leaf.
constexpr bool closedGpuShapeCompatibleV1(std::uint64_t m, std::uint64_t n,
                                           std::uint64_t k) noexcept {
  if (m > 65535 || n > 65535 || k > 65535) return false;
  const std::uint64_t output = m * n; // bounded extents make this representable
  return output <= (std::uint64_t{1} << 20) &&
         (k == 0 || output <= (std::uint64_t{1} << 26) / k);
}
// Separate qualification envelope for a single-thread combined row-panel
// recipe, not a source restriction, cost model or dispatch crossover. Both
// original per-GEMM envelopes must still hold before this tighter shared-work
// predicate. The caller checks their source frontiers in order; this aggregate
// predicate belongs only to the second frontier.
constexpr bool closedGpuFusedPairCompatibleV1(std::uint64_t m, std::uint64_t k,
                                              std::uint64_t n,
                                              std::uint64_t p) noexcept {
  if (!closedGpuShapeCompatibleV1(m, n, k) ||
      !closedGpuShapeCompatibleV1(m, p, n)) return false;
  // Each factor is <=65535, so the complete sum is representable in uint64_t.
  return m * n * (k + p) <= (std::uint64_t{1} << 18);
}
}
#endif
