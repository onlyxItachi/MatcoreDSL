#ifndef MATCORE_MDSLC_PLATFORM_THREAD_AFFINITY_V1_H
#define MATCORE_MDSLC_PLATFORM_THREAD_AFFINITY_V1_H

#include <cstdint>
#include <string_view>
#include <vector>

namespace matcore::mdslc::platform {

inline constexpr std::uint32_t kThreadAffinityVersionV1 = 1;
inline constexpr std::uint32_t kUnknownLogicalCpuV1 = UINT32_MAX;

enum class ThreadAffinityStatusV1 : std::uint8_t {
  not_requested = 0,
  applied = 1,
  unavailable = 2,
  invalid_cpu_id = 3,
  system_error = 4,
};

// Result of binding the calling thread to exactly one logical CPU. This API
// changes only scheduler affinity. It does not allocate, migrate, bind, or
// interleave memory and therefore is not evidence of NUMA page placement.
struct ThreadAffinityApplicationV1 {
  std::uint32_t version = kThreadAffinityVersionV1;
  ThreadAffinityStatusV1 status = ThreadAffinityStatusV1::not_requested;
  std::uint32_t requested_logical_cpu = kUnknownLogicalCpuV1;
  std::int32_t platform_error = 0;
};

struct ThreadAffinityInventoryV1 {
  std::uint32_t version = kThreadAffinityVersionV1;
  bool backend_available = false;
  bool discovery_complete = false;
  std::int32_t platform_error = 0;
  std::vector<std::uint32_t> allowed_logical_cpus;
};

// Snapshot of the logical processor currently executing the calling thread.
// This is deliberately separate from the allowed-CPU inventory: the latter is
// a permission set, while this record authenticates the creator's local NUMA
// node at context-creation time.
struct CurrentLogicalCpuV1 {
  std::uint32_t version = kThreadAffinityVersionV1;
  bool backend_available = false;
  bool discovery_complete = false;
  std::uint32_t logical_cpu = kUnknownLogicalCpuV1;
  std::int32_t platform_error = 0;
};

// Applies a one-CPU affinity mask to the calling thread. Linux uses pthread
// affinity. Windows uses SetThreadGroupAffinity after authenticating the
// process mask; v1 deliberately rejects multi-group (>64 logical CPU) hosts.
// Unsupported platforms fail closed with `unavailable`; there is no no-op
// success path.
ThreadAffinityApplicationV1 apply_current_thread_affinity_v1(
    std::uint32_t logical_cpu) noexcept;

// Returns the calling thread's current scheduler-affinity mask. On Windows the
// result is the intersection of GetProcessAffinityMask and
// GetThreadGroupAffinity. The list is ordered and duplicate-free when
// discovery_complete is true.
ThreadAffinityInventoryV1 discover_current_thread_affinity_v1();

// Returns the logical processor currently executing the calling thread
// (GetCurrentProcessorNumberEx on Windows).
// Unsupported platforms and system failures are reported explicitly; callers
// must not substitute an arbitrary processor when discovery is incomplete.
CurrentLogicalCpuV1 discover_current_logical_cpu_v1() noexcept;

std::string_view to_string(ThreadAffinityStatusV1 status) noexcept;

}  // namespace matcore::mdslc::platform

#endif
