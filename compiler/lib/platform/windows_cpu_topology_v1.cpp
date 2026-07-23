#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "cpu_topology_v1.h"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

#if defined(_WIN32)
#include <windows.h>

#include <cstddef>
#include <vector>
#endif

namespace matcore::mdslc::platform {
namespace {

CpuTopologyV1 incomplete_topology(ArchitectureKindV1 architecture) {
  CpuTopologyV1 record;
  record.architecture = architecture;
  return record;
}

bool known_architecture(ArchitectureKindV1 architecture) noexcept {
  return architecture == ArchitectureKindV1::x86_64 ||
         architecture == ArchitectureKindV1::aarch64;
}

bool valid_windows_processor(
    const WindowsProcessorNumberV1 &processor) noexcept {
  return processor.group == 0 &&
         processor.number < kWindowsProcessorGroupWidthV1;
}

#if defined(_WIN32)
ArchitectureKindV1 windows_compile_architecture() noexcept {
#if defined(_M_X64)
  return ArchitectureKindV1::x86_64;
#elif defined(_M_ARM64)
  return ArchitectureKindV1::aarch64;
#else
  return ArchitectureKindV1::unknown;
#endif
}

bool query_relationship(LOGICAL_PROCESSOR_RELATIONSHIP relationship,
                        std::vector<unsigned char> *buffer) {
  if (buffer == nullptr) return false;
  DWORD bytes = 0;
  ::SetLastError(ERROR_SUCCESS);
  if (::GetLogicalProcessorInformationEx(relationship, nullptr, &bytes) !=
          FALSE ||
      ::GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) {
    return false;
  }
  try {
    buffer->assign(bytes, 0);
  } catch (...) {
    buffer->clear();
    return false;
  }
  if (::GetLogicalProcessorInformationEx(
          relationship,
          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
              buffer->data()),
          &bytes) == FALSE) {
    buffer->clear();
    return false;
  }
  buffer->resize(bytes);
  return true;
}

template <class Callback>
bool visit_relationships(const std::vector<unsigned char> &buffer,
                         Callback &&callback) {
  constexpr std::size_t kRelationshipHeaderBytes =
      offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor);
  std::size_t offset = 0;
  while (offset < buffer.size()) {
    if (buffer.size() - offset < kRelationshipHeaderBytes) {
      return false;
    }
    const auto *entry =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(
            buffer.data() + offset);
    if (entry->Size < kRelationshipHeaderBytes ||
        entry->Size > buffer.size() - offset || !callback(*entry)) {
      return false;
    }
    offset += entry->Size;
  }
  return offset == buffer.size();
}

bool append_group_affinity(
    const GROUP_AFFINITY &affinity,
    std::vector<WindowsProcessorNumberV1> *processors) {
  if (processors == nullptr || affinity.Group != 0 || affinity.Mask == 0) {
    return false;
  }
  constexpr unsigned kAffinityBits = sizeof(KAFFINITY) * 8U;
  for (unsigned number = 0; number < kAffinityBits; ++number) {
    const KAFFINITY bit = static_cast<KAFFINITY>(1) << number;
    if ((affinity.Mask & bit) != 0) {
      processors->push_back(
          {0, static_cast<std::uint16_t>(number)});
    }
  }
  return true;
}

WindowsLogicalProcessorRecordV1 *find_processor(
    WindowsCpuTopologySnapshotV1 *snapshot,
    const WindowsProcessorNumberV1 &processor) {
  if (snapshot == nullptr || !valid_windows_processor(processor)) {
    return nullptr;
  }
  const auto found = std::find_if(
      snapshot->logical_processors.begin(),
      snapshot->logical_processors.end(), [&processor](const auto &candidate) {
        return candidate.processor.group == processor.group &&
               candidate.processor.number == processor.number;
      });
  return found == snapshot->logical_processors.end() ? nullptr : &*found;
}

bool assign_processor_relationship(
    const std::vector<unsigned char> &buffer,
    LOGICAL_PROCESSOR_RELATIONSHIP expected,
    WindowsCpuTopologySnapshotV1 *snapshot) {
  std::uint32_t relationship_count = 0;
  return visit_relationships(buffer, [&](const auto &entry) {
    if (entry.Relationship != expected || entry.Processor.GroupCount != 1) {
      return false;
    }
    constexpr std::size_t kProcessorOffset =
        offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor);
    constexpr std::size_t kMasksOffset =
        offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
    const std::size_t required =
        kProcessorOffset + kMasksOffset +
        static_cast<std::size_t>(entry.Processor.GroupCount) *
            sizeof(GROUP_AFFINITY);
    if (entry.Size < required) return false;
    std::vector<WindowsProcessorNumberV1> processors;
    if (!append_group_affinity(entry.Processor.GroupMask[0], &processors) ||
        processors.empty()) {
      return false;
    }
    std::sort(processors.begin(), processors.end(), [](const auto &left,
                                                       const auto &right) {
      return std::tie(left.group, left.number) <
             std::tie(right.group, right.number);
    });
    const std::uint32_t stable_id = processors.front().number;
    for (std::size_t index = 0; index < processors.size(); ++index) {
      auto *logical = find_processor(snapshot, processors[index]);
      if (logical == nullptr) return false;
      std::uint32_t *field = expected == RelationProcessorCore
                                 ? &logical->core_id
                                 : &logical->package_id;
      if (*field != kUnknownTopologyIdV1) return false;
      *field = stable_id;
      if (expected == RelationProcessorCore) {
        logical->thread_index = static_cast<std::uint32_t>(index);
      }
    }
    ++relationship_count;
    return true;
  }) && relationship_count != 0;
}

bool assign_numa_relationships(const std::vector<unsigned char> &buffer,
                               WindowsCpuTopologySnapshotV1 *snapshot) {
  std::uint32_t node_count = 0;
  return visit_relationships(buffer, [&](const auto &entry) {
    // RelationNumaNodeEx is an input request.  Windows reports
    // RelationNumaNode entries whose GroupMasks array carries full affinity.
    if (entry.Relationship != RelationNumaNode ||
        entry.NumaNode.GroupCount == 0) {
      return false;
    }
    constexpr std::size_t kNumaOffset =
        offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, NumaNode);
    constexpr std::size_t kMasksOffset =
        offsetof(NUMA_NODE_RELATIONSHIP, GroupMasks);
    const std::size_t required =
        kNumaOffset + kMasksOffset +
        static_cast<std::size_t>(entry.NumaNode.GroupCount) *
            sizeof(GROUP_AFFINITY);
    if (entry.Size < required) return false;
    std::vector<WindowsProcessorNumberV1> processors;
    for (WORD index = 0; index < entry.NumaNode.GroupCount; ++index) {
      if (!append_group_affinity(entry.NumaNode.GroupMasks[index],
                                 &processors)) {
        return false;
      }
    }
    if (processors.empty()) return false;
    for (const auto &processor : processors) {
      auto *logical = find_processor(snapshot, processor);
      if (logical == nullptr ||
          logical->numa_node_id != kUnknownTopologyIdV1) {
        return false;
      }
      logical->numa_node_id = entry.NumaNode.NodeNumber;
    }
    ++node_count;
    return true;
  }) && node_count != 0;
}

CpuCacheTypeV1 normalize_windows_cache_type(PROCESSOR_CACHE_TYPE type) {
  switch (type) {
    case CacheData:
      return CpuCacheTypeV1::data;
    case CacheInstruction:
      return CpuCacheTypeV1::instruction;
    case CacheUnified:
      return CpuCacheTypeV1::unified;
    case CacheTrace:
    case CacheUnknown:
      return CpuCacheTypeV1::unknown;
  }
  return CpuCacheTypeV1::unknown;
}

bool collect_cache_relationships(const std::vector<unsigned char> &buffer,
                                 WindowsCpuTopologySnapshotV1 *snapshot) {
  std::uint32_t represented_cache_count = 0;
  const bool valid = visit_relationships(buffer, [&](const auto &entry) {
    if (entry.Relationship != RelationCache || entry.Cache.GroupCount != 1) {
      return false;
    }
    constexpr std::size_t kCacheOffset =
        offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Cache);
    constexpr std::size_t kMasksOffset =
        offsetof(CACHE_RELATIONSHIP, GroupMasks);
    const std::size_t required =
        kCacheOffset + kMasksOffset +
        static_cast<std::size_t>(entry.Cache.GroupCount) *
            sizeof(GROUP_AFFINITY);
    if (entry.Size < required) return false;
    const CpuCacheTypeV1 type =
        normalize_windows_cache_type(entry.Cache.Type);
    if (type == CpuCacheTypeV1::unknown) {
      // Trace caches are not part of the v1 data/instruction/unified model.
      return true;
    }
    WindowsCacheRecordV1 cache;
    cache.level = entry.Cache.Level;
    cache.type = type;
    cache.size_bytes = entry.Cache.CacheSize;
    cache.line_size_bytes = entry.Cache.LineSize;
    if (!append_group_affinity(entry.Cache.GroupMasks[0],
                               &cache.shared_processors) ||
        cache.level == 0 || cache.size_bytes == 0 ||
        cache.line_size_bytes == 0) {
      return false;
    }
    snapshot->cache_groups.push_back(std::move(cache));
    ++represented_cache_count;
    return true;
  });
  return valid && represented_cache_count != 0;
}
#endif

}  // namespace

CpuTopologyV1 normalize_windows_cpu_topology_v1(
    const WindowsCpuTopologySnapshotV1 &snapshot) {
  const ArchitectureKindV1 architecture =
      known_architecture(snapshot.architecture) ? snapshot.architecture
                                                : ArchitectureKindV1::unknown;
  CpuTopologyV1 result = incomplete_topology(architecture);
  if (snapshot.version != kWindowsCpuTopologySnapshotVersionV1 ||
      !known_architecture(snapshot.architecture) ||
      snapshot.active_processor_groups != 1 ||
      !snapshot.relationship_discovery_complete ||
      snapshot.logical_processors.empty() || snapshot.cache_groups.empty()) {
    return result;
  }

  try {
    for (const WindowsLogicalProcessorRecordV1 &input :
         snapshot.logical_processors) {
      if (!valid_windows_processor(input.processor) || !input.online ||
          input.core_id == kUnknownTopologyIdV1 ||
          input.package_id == kUnknownTopologyIdV1 ||
          input.numa_node_id == kUnknownTopologyIdV1 ||
          input.thread_index == kUnknownTopologyIdV1) {
        return incomplete_topology(architecture);
      }
      result.logical_processors.push_back(
          {input.processor.number, input.core_id, input.package_id,
           input.numa_node_id, input.thread_index, true});
    }
    std::sort(result.logical_processors.begin(),
              result.logical_processors.end(), [](const auto &left,
                                                   const auto &right) {
                return left.logical_cpu < right.logical_cpu;
              });
    if (std::adjacent_find(
            result.logical_processors.begin(),
            result.logical_processors.end(), [](const auto &left,
                                                 const auto &right) {
              return left.logical_cpu == right.logical_cpu;
            }) != result.logical_processors.end()) {
      return incomplete_topology(architecture);
    }

    std::map<std::uint32_t, std::vector<std::uint32_t>> nodes;
    for (const CpuLogicalProcessorV1 &processor :
         result.logical_processors) {
      nodes[processor.numa_node_id].push_back(processor.logical_cpu);
    }
    for (auto &[node_id, logical_cpus] : nodes) {
      result.numa_nodes.push_back({node_id, std::move(logical_cpus)});
    }

    for (const WindowsCacheRecordV1 &input : snapshot.cache_groups) {
      if (input.type == CpuCacheTypeV1::unknown || input.level == 0 ||
          input.size_bytes == 0 || input.line_size_bytes == 0 ||
          input.shared_processors.empty()) {
        return incomplete_topology(architecture);
      }
      CpuCacheGroupV1 cache;
      cache.level = input.level;
      cache.type = input.type;
      cache.size_bytes = input.size_bytes;
      cache.line_size_bytes = input.line_size_bytes;
      for (const WindowsProcessorNumberV1 &processor :
           input.shared_processors) {
        if (!valid_windows_processor(processor)) {
          return incomplete_topology(architecture);
        }
        cache.shared_logical_cpus.push_back(processor.number);
      }
      std::sort(cache.shared_logical_cpus.begin(),
                cache.shared_logical_cpus.end());
      if (std::adjacent_find(cache.shared_logical_cpus.begin(),
                             cache.shared_logical_cpus.end()) !=
          cache.shared_logical_cpus.end()) {
        return incomplete_topology(architecture);
      }
      result.cache_groups.push_back(std::move(cache));
    }
    std::sort(result.cache_groups.begin(), result.cache_groups.end(),
              [](const CpuCacheGroupV1 &left,
                 const CpuCacheGroupV1 &right) {
                return std::tie(left.level, left.type,
                                left.shared_logical_cpus, left.size_bytes,
                                left.line_size_bytes) <
                       std::tie(right.level, right.type,
                                right.shared_logical_cpus, right.size_bytes,
                                right.line_size_bytes);
              });
    result.cache_groups.erase(
        std::unique(result.cache_groups.begin(), result.cache_groups.end(),
                    [](const auto &left, const auto &right) {
                      return left.level == right.level &&
                             left.type == right.type &&
                             left.size_bytes == right.size_bytes &&
                             left.line_size_bytes == right.line_size_bytes &&
                             left.shared_logical_cpus ==
                                 right.shared_logical_cpus;
                    }),
        result.cache_groups.end());
  } catch (...) {
    return incomplete_topology(architecture);
  }

  result.discovery_complete = true;
  if (!validate_cpu_topology_v1(result)) {
    return incomplete_topology(architecture);
  }
  return result;
}

CpuTopologyV1 discover_windows_cpu_topology_v1() {
#if defined(_WIN32)
  WindowsCpuTopologySnapshotV1 snapshot;
  snapshot.architecture = windows_compile_architecture();
  snapshot.active_processor_groups = ::GetActiveProcessorGroupCount();

  // CpuTopologyV1 and ThreadAffinityV1 deliberately model a flat single-group
  // logical CPU ID.  Returning an incomplete record is safer than truncating
  // or aliasing processor coordinates on machines with more than 64 CPUs.
  if (!known_architecture(snapshot.architecture) ||
      snapshot.active_processor_groups != 1) {
    return normalize_windows_cpu_topology_v1(snapshot);
  }
  const DWORD active_processors = ::GetActiveProcessorCount(0);
  if (active_processors == 0 ||
      active_processors > kWindowsProcessorGroupWidthV1) {
    return normalize_windows_cpu_topology_v1(snapshot);
  }
  try {
    snapshot.logical_processors.reserve(active_processors);
    for (DWORD number = 0; number < active_processors; ++number) {
      WindowsLogicalProcessorRecordV1 processor;
      processor.processor.number = static_cast<std::uint16_t>(number);
      processor.online = true;
      snapshot.logical_processors.push_back(processor);
    }
  } catch (...) {
    return normalize_windows_cpu_topology_v1(snapshot);
  }

  std::vector<unsigned char> cores;
  std::vector<unsigned char> packages;
  std::vector<unsigned char> nodes;
  std::vector<unsigned char> caches;
  snapshot.relationship_discovery_complete =
      query_relationship(RelationProcessorCore, &cores) &&
      query_relationship(RelationProcessorPackage, &packages) &&
      query_relationship(RelationNumaNodeEx, &nodes) &&
      query_relationship(RelationCache, &caches) &&
      assign_processor_relationship(cores, RelationProcessorCore, &snapshot) &&
      assign_processor_relationship(packages, RelationProcessorPackage,
                                    &snapshot) &&
      assign_numa_relationships(nodes, &snapshot) &&
      collect_cache_relationships(caches, &snapshot);
  return normalize_windows_cpu_topology_v1(snapshot);
#else
  return {};
#endif
}

CpuTopologyV1 discover_host_cpu_topology_v1() {
#if defined(__linux__)
  return discover_linux_cpu_topology_v1();
#elif defined(_WIN32)
  return discover_windows_cpu_topology_v1();
#else
  return {};
#endif
}

}  // namespace matcore::mdslc::platform
