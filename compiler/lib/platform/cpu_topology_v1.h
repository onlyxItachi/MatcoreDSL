#ifndef MATCORE_MDSLC_PLATFORM_CPU_TOPOLOGY_V1_H
#define MATCORE_MDSLC_PLATFORM_CPU_TOPOLOGY_V1_H

#include "platform.h"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace matcore::mdslc::platform {

inline constexpr std::uint32_t kCpuTopologyVersionV1 = 1;
inline constexpr std::uint32_t kCpuPlacementVersionV1 = 1;
inline constexpr std::uint32_t kWindowsCpuTopologySnapshotVersionV1 = 1;
inline constexpr std::uint32_t kWindowsProcessorGroupWidthV1 = 64;
inline constexpr std::uint32_t kUnknownTopologyIdV1 =
    std::numeric_limits<std::uint32_t>::max();

enum class CpuCacheTypeV1 : std::uint8_t {
  unknown = 0,
  data = 1,
  instruction = 2,
  unified = 3,
};

struct CpuLogicalProcessorV1 {
  std::uint32_t logical_cpu = kUnknownTopologyIdV1;
  std::uint32_t core_id = kUnknownTopologyIdV1;
  std::uint32_t package_id = kUnknownTopologyIdV1;
  std::uint32_t numa_node_id = kUnknownTopologyIdV1;
  std::uint32_t thread_index = kUnknownTopologyIdV1;
  bool online = false;
};

struct CpuNumaNodeV1 {
  std::uint32_t node_id = kUnknownTopologyIdV1;
  std::vector<std::uint32_t> logical_cpus;
};

struct CpuCacheGroupV1 {
  std::uint32_t level = 0;
  CpuCacheTypeV1 type = CpuCacheTypeV1::unknown;
  std::uint64_t size_bytes = 0;
  std::uint32_t line_size_bytes = 0;
  std::vector<std::uint32_t> shared_logical_cpus;
};

struct CpuTopologyV1 {
  std::uint32_t version = kCpuTopologyVersionV1;
  ArchitectureKindV1 architecture = ArchitectureKindV1::unknown;
  bool discovery_complete = false;
  std::vector<CpuLogicalProcessorV1> logical_processors;
  std::vector<CpuNumaNodeV1> numa_nodes;
  std::vector<CpuCacheGroupV1> cache_groups;
};

struct CpuTopologyValidationV1 {
  bool valid = false;
  std::string_view reason;

  constexpr explicit operator bool() const noexcept { return valid; }
};

// Portable input model for the Win32 topology collector.  Keeping the
// normalization boundary free of Windows SDK types makes processor-group,
// NUMA, cache, and malformed-record behavior testable on Linux CI.  V1 maps a
// Windows PROCESSOR_NUMBER to its Number only and therefore deliberately
// accepts exactly one active processor group.  Multi-group machines fail
// closed until a future topology/affinity record carries a first-class group
// coordinate through the planner and public diagnostics.
struct WindowsProcessorNumberV1 {
  std::uint16_t group = 0;
  std::uint16_t number = 0;
};

struct WindowsLogicalProcessorRecordV1 {
  WindowsProcessorNumberV1 processor;
  std::uint32_t core_id = kUnknownTopologyIdV1;
  std::uint32_t package_id = kUnknownTopologyIdV1;
  std::uint32_t numa_node_id = kUnknownTopologyIdV1;
  std::uint32_t thread_index = kUnknownTopologyIdV1;
  bool online = false;
};

struct WindowsCacheRecordV1 {
  std::uint32_t level = 0;
  CpuCacheTypeV1 type = CpuCacheTypeV1::unknown;
  std::uint64_t size_bytes = 0;
  std::uint32_t line_size_bytes = 0;
  std::vector<WindowsProcessorNumberV1> shared_processors;
};

struct WindowsCpuTopologySnapshotV1 {
  std::uint32_t version = kWindowsCpuTopologySnapshotVersionV1;
  ArchitectureKindV1 architecture = ArchitectureKindV1::unknown;
  std::uint16_t active_processor_groups = 0;
  bool relationship_discovery_complete = false;
  std::vector<WindowsLogicalProcessorRecordV1> logical_processors;
  std::vector<WindowsCacheRecordV1> cache_groups;
};

enum class CpuAffinityPolicyV1 : std::uint8_t {
  compact = 0,
  scatter = 1,
  local_first = 2,
};

enum class CpuSmtPolicyV1 : std::uint8_t {
  physical_cores_only = 0,
  prefer_physical_cores = 1,
  allow_smt = 2,
};

enum class CpuPlacementStatusV1 : std::uint8_t {
  selected = 0,
  invalid_topology = 1,
  invalid_request = 2,
  insufficient_cpus = 3,
  cross_numa_disallowed = 4,
};

enum class CpuTopologyRestrictionStatusV1 : std::uint8_t {
  selected = 0,
  invalid_topology = 1,
  empty_cpu_set = 2,
  duplicate_cpu_id = 3,
  unavailable_cpu_id = 4,
  resource_exhausted = 5,
};

struct CpuPlacementRequestV1 {
  std::uint32_t version = kCpuPlacementVersionV1;
  std::uint32_t requested_workers = 1;
  CpuAffinityPolicyV1 affinity = CpuAffinityPolicyV1::compact;
  CpuSmtPolicyV1 smt = CpuSmtPolicyV1::prefer_physical_cores;
  std::uint32_t preferred_numa_node = kUnknownTopologyIdV1;
  bool allow_cross_numa = false;
};

struct CpuPlacementPlanV1 {
  std::uint32_t version = kCpuPlacementVersionV1;
  CpuPlacementStatusV1 status = CpuPlacementStatusV1::invalid_request;
  std::uint32_t requested_workers = 0;
  std::uint32_t actual_workers = 0;
  CpuAffinityPolicyV1 affinity = CpuAffinityPolicyV1::compact;
  CpuSmtPolicyV1 smt = CpuSmtPolicyV1::prefer_physical_cores;
  std::vector<std::uint32_t> logical_cpus;
  std::vector<std::uint32_t> numa_nodes;
  bool crosses_numa_nodes = false;
  bool affinity_application_required = false;
  bool caller_first_touch_required = false;
  std::string reason;
};

/*
 * A deterministic, caller-owned projection of a discovered topology onto an
 * externally imposed logical-CPU set (for example sched_getaffinity).  The
 * source topology is never modified.  NUMA and cache groups are intersected
 * with the allowed set and empty groups are removed.  SMT thread indices are
 * recomputed so a permitted sibling remains usable when its lower-numbered
 * sibling is outside the allowed set.
 */
struct CpuTopologyRestrictionV1 {
  CpuTopologyRestrictionStatusV1 status =
      CpuTopologyRestrictionStatusV1::invalid_topology;
  CpuTopologyV1 topology;
  std::string reason;

  explicit operator bool() const noexcept {
    return status == CpuTopologyRestrictionStatusV1::selected;
  }
};

CpuTopologyValidationV1 validate_cpu_topology_v1(
    const CpuTopologyV1 &record) noexcept;

CpuTopologyV1 discover_linux_cpu_topology_v1(
    const std::filesystem::path &sys_devices_root =
        "/sys/devices/system");

CpuTopologyV1 normalize_windows_cpu_topology_v1(
    const WindowsCpuTopologySnapshotV1 &snapshot);

// Uses GetLogicalProcessorInformationEx with RelationProcessorCore,
// RelationProcessorPackage, RelationCache, and RelationNumaNodeEx.  Unsupported
// processor-group layouts and incomplete relationship data return a valid but
// incomplete topology; they are never accepted by placement.
CpuTopologyV1 discover_windows_cpu_topology_v1();

// Selects the native collector without exposing OS preprocessor branches to
// planner/runtime consumers.  Unknown platforms return an incomplete record.
CpuTopologyV1 discover_host_cpu_topology_v1();

CpuTopologyRestrictionV1 restrict_cpu_topology_v1(
    const CpuTopologyV1 &topology,
    const std::vector<std::uint32_t> &allowed_logical_cpus);

std::uint32_t logical_cpu_count_v1(const CpuTopologyV1 &record) noexcept;
std::uint32_t physical_core_count_v1(const CpuTopologyV1 &record) noexcept;
std::uint32_t socket_count_v1(const CpuTopologyV1 &record) noexcept;
std::uint32_t numa_node_count_v1(const CpuTopologyV1 &record) noexcept;

CpuPlacementPlanV1 plan_cpu_placement_v1(
    const CpuTopologyV1 &topology,
    const CpuPlacementRequestV1 &request);

std::string_view to_string(CpuCacheTypeV1 value) noexcept;
std::string_view to_string(CpuAffinityPolicyV1 value) noexcept;
std::string_view to_string(CpuSmtPolicyV1 value) noexcept;
std::string_view to_string(CpuPlacementStatusV1 value) noexcept;
std::string_view to_string(CpuTopologyRestrictionStatusV1 value) noexcept;
std::string format_cpu_topology_v1(const CpuTopologyV1 &record);
std::string format_cpu_placement_v1(const CpuPlacementPlanV1 &plan);

}  // namespace matcore::mdslc::platform

#endif
