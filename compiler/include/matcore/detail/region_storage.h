#ifndef MATCORE_DETAIL_REGION_STORAGE_H
#define MATCORE_DETAIL_REGION_STORAGE_H

// Installed implementation detail for the experimental region result. These
// layouts are shared with the private execution adapter, not a stable ABI.
#include <cstdint>

namespace matcore::mdslc::runtime::closed_host_v1 {
class SessionAbiV2;
using Session = SessionAbiV2;
using Frontier = std::uint64_t;
enum class Access : std::uint8_t { read_only, read_write };
enum class Numeric : std::uint8_t { strict_f32, reassociate_f32 };
enum class Code : std::uint8_t {
  ok, invalid_frontier, invalid_value, invalid_view, shape_mismatch,
  extent_overflow, insufficient_capacity, access_denied, allocation_failure,
  candidate_failure, unsupported_fp_environment, reentrant_use, already_complete,
  invalid_candidate, candidate_unavailable, candidate_incompatible
};
struct Status {
  Code code = Code::ok;
  Frontier failed_frontier = 0;
  Frontier completed_frontier = 0;
  Frontier completed_effect_frontier = 0;
  std::uint64_t publications = 0;
  std::uint64_t observations = 0;
  bool completed = false;
  explicit operator bool() const noexcept { return code == Code::ok; }
};
struct ResourceView {
  float *data = nullptr;
  std::uint64_t rows = 0;
  std::uint64_t columns = 0;
  std::uint64_t capacity_elements = 0;
  Access access = Access::read_only;
};
// Ownership implementation is deliberately opaque: host standard-library
// debug/ABI configuration cannot alter the public result layout or destructor.
struct ObservationBlock;
const char *message(Code) noexcept;
} // namespace matcore::mdslc::runtime::closed_host_v1

#endif
