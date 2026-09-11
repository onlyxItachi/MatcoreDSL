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
}
#endif
