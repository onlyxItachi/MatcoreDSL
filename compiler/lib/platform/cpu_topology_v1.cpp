#include "cpu_topology_v1.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <system_error>
#include <tuple>
#include <utility>

namespace matcore::mdslc::platform {
namespace {

std::string trim(std::string value) {
  const auto not_space = [](unsigned char character) {
    return std::isspace(character) == 0;
  };
  const auto begin = std::find_if(value.begin(), value.end(), not_space);
  const auto end = std::find_if(value.rbegin(), value.rend(), not_space).base();
  if (begin >= end) return {};
  return std::string(begin, end);
}

std::optional<std::string> read_text(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) return std::nullopt;
  std::ostringstream contents;
  contents << input.rdbuf();
  if (input.bad()) return std::nullopt;
  return trim(contents.str());
}

std::optional<std::uint32_t> parse_u32(std::string_view text) noexcept {
  if (text.empty()) return std::nullopt;
  std::uint32_t value = 0;
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end) return std::nullopt;
  return value;
}

std::optional<std::vector<std::uint32_t>> parse_cpu_list(
    std::string_view text) {
  std::vector<std::uint32_t> cpus;
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    const std::size_t comma = text.find(',', cursor);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    const std::string_view token = text.substr(cursor, end - cursor);
    if (token.empty()) return std::nullopt;
    const std::size_t dash = token.find('-');
    const auto first = parse_u32(token.substr(0, dash));
    if (!first) return std::nullopt;
    std::uint32_t last = *first;
    if (dash != std::string_view::npos) {
      if (token.find('-', dash + 1) != std::string_view::npos) {
        return std::nullopt;
      }
      const auto parsed_last = parse_u32(token.substr(dash + 1));
      if (!parsed_last || *parsed_last < *first) return std::nullopt;
      last = *parsed_last;
    }
    if (static_cast<std::uint64_t>(last) - *first > UINT64_C(1048576)) {
      return std::nullopt;
    }
    for (std::uint32_t cpu = *first;; ++cpu) {
      cpus.push_back(cpu);
      if (cpu == last) break;
      if (cpu == std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
      }
    }
    if (comma == std::string_view::npos) break;
    if (comma + 1 == text.size()) return std::nullopt;
    cursor = comma + 1;
  }
  std::sort(cpus.begin(), cpus.end());
  if (std::adjacent_find(cpus.begin(), cpus.end()) != cpus.end()) {
    return std::nullopt;
  }
  return cpus;
}

std::optional<std::uint64_t> parse_cache_size(std::string_view text) noexcept {
  if (text.empty()) return std::nullopt;
  std::uint64_t multiplier = 1;
  const char suffix = text.back();
  if (suffix == 'K' || suffix == 'k') {
    multiplier = UINT64_C(1024);
    text.remove_suffix(1);
  } else if (suffix == 'M' || suffix == 'm') {
    multiplier = UINT64_C(1024) * UINT64_C(1024);
    text.remove_suffix(1);
  } else if (suffix == 'G' || suffix == 'g') {
    multiplier = UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024);
    text.remove_suffix(1);
  }
  if (text.empty()) return std::nullopt;
  std::uint64_t value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      value > std::numeric_limits<std::uint64_t>::max() / multiplier) {
    return std::nullopt;
  }
  return value * multiplier;
}

std::optional<std::uint32_t> id_from_name(std::string_view name,
                                          std::string_view prefix) noexcept {
  if (!name.starts_with(prefix)) return std::nullopt;
  return parse_u32(name.substr(prefix.size()));
}

CpuCacheTypeV1 parse_cache_type(std::string_view value) noexcept {
  if (value == "Data") return CpuCacheTypeV1::data;
  if (value == "Instruction") return CpuCacheTypeV1::instruction;
  if (value == "Unified") return CpuCacheTypeV1::unified;
  return CpuCacheTypeV1::unknown;
}

bool contains_cpu(const std::vector<std::uint32_t> &cpus,
                  std::uint32_t cpu) noexcept {
  return std::binary_search(cpus.begin(), cpus.end(), cpu);
}

bool known(CpuCacheTypeV1 value) noexcept {
  switch (value) {
    case CpuCacheTypeV1::unknown:
    case CpuCacheTypeV1::data:
    case CpuCacheTypeV1::instruction:
    case CpuCacheTypeV1::unified:
      return true;
  }
  return false;
}

bool known(CpuAffinityPolicyV1 value) noexcept {
  switch (value) {
    case CpuAffinityPolicyV1::compact:
    case CpuAffinityPolicyV1::scatter:
    case CpuAffinityPolicyV1::local_first:
      return true;
  }
  return false;
}

bool known(CpuSmtPolicyV1 value) noexcept {
  switch (value) {
    case CpuSmtPolicyV1::physical_cores_only:
    case CpuSmtPolicyV1::prefer_physical_cores:
    case CpuSmtPolicyV1::allow_smt:
      return true;
  }
  return false;
}

using CoreKey = std::tuple<std::uint32_t, std::uint32_t>;

std::vector<const CpuLogicalProcessorV1 *> topology_candidates(
    const CpuTopologyV1 &topology, CpuSmtPolicyV1 smt,
    const std::set<std::uint32_t> &allowed_nodes) {
  std::vector<const CpuLogicalProcessorV1 *> base;
  for (const CpuLogicalProcessorV1 &processor : topology.logical_processors) {
    if (!processor.online ||
        allowed_nodes.find(processor.numa_node_id) == allowed_nodes.end()) {
      continue;
    }
    if (smt == CpuSmtPolicyV1::physical_cores_only &&
        processor.thread_index != 0) {
      continue;
    }
    base.push_back(&processor);
  }
  std::sort(base.begin(), base.end(), [](const auto *left, const auto *right) {
    return std::tie(left->numa_node_id, left->package_id, left->core_id,
                    left->thread_index, left->logical_cpu) <
           std::tie(right->numa_node_id, right->package_id, right->core_id,
                    right->thread_index, right->logical_cpu);
  });
  if (smt != CpuSmtPolicyV1::prefer_physical_cores) return base;

  std::stable_sort(base.begin(), base.end(), [](const auto *left,
                                                const auto *right) {
    const bool left_primary = left->thread_index == 0;
    const bool right_primary = right->thread_index == 0;
    return left_primary != right_primary && left_primary;
  });
  return base;
}

std::vector<const CpuLogicalProcessorV1 *> scatter_candidates(
    const std::vector<const CpuLogicalProcessorV1 *> &input) {
  using BucketKey = std::pair<std::uint32_t, std::uint32_t>;
  std::map<BucketKey, std::vector<const CpuLogicalProcessorV1 *>> buckets;
  for (const auto *processor : input) {
    buckets[{processor->numa_node_id, processor->package_id}].push_back(
        processor);
  }
  std::vector<const CpuLogicalProcessorV1 *> output;
  std::size_t position = 0;
  while (output.size() < input.size()) {
    bool emitted = false;
    for (const auto &[key, bucket] : buckets) {
      (void)key;
      if (position < bucket.size()) {
        output.push_back(bucket[position]);
        emitted = true;
      }
    }
    if (!emitted) break;
    ++position;
  }
  return output;
}

ArchitectureKindV1 compile_architecture() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  return ArchitectureKindV1::x86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
  return ArchitectureKindV1::aarch64;
#else
  return ArchitectureKindV1::unknown;
#endif
}

}  // namespace

CpuTopologyValidationV1 validate_cpu_topology_v1(
    const CpuTopologyV1 &record) noexcept {
  if (record.version != kCpuTopologyVersionV1) {
    return {false, "CPU topology record version is unsupported"};
  }
  if (!known(record.architecture)) {
    return {false, "CPU topology architecture is invalid"};
  }
  std::uint32_t previous_cpu = 0;
  bool first_cpu = true;
  for (const CpuLogicalProcessorV1 &processor : record.logical_processors) {
    if (processor.logical_cpu == kUnknownTopologyIdV1) {
      return {false, "CPU topology contains an unknown logical CPU ID"};
    }
    if (!first_cpu && processor.logical_cpu <= previous_cpu) {
      return {false, "CPU topology logical CPUs must be unique and sorted"};
    }
    first_cpu = false;
    previous_cpu = processor.logical_cpu;
    if (record.discovery_complete &&
        (!processor.online || processor.core_id == kUnknownTopologyIdV1 ||
         processor.package_id == kUnknownTopologyIdV1 ||
         processor.numa_node_id == kUnknownTopologyIdV1 ||
         processor.thread_index == kUnknownTopologyIdV1)) {
      return {false, "complete topology requires every online CPU mapping"};
    }
  }

  std::uint32_t previous_node = 0;
  bool first_node = true;
  for (const CpuNumaNodeV1 &node : record.numa_nodes) {
    if (node.node_id == kUnknownTopologyIdV1 ||
        (!first_node && node.node_id <= previous_node)) {
      return {false, "NUMA node IDs must be known, unique, and sorted"};
    }
    first_node = false;
    previous_node = node.node_id;
    if (!std::is_sorted(node.logical_cpus.begin(), node.logical_cpus.end()) ||
        std::adjacent_find(node.logical_cpus.begin(),
                           node.logical_cpus.end()) != node.logical_cpus.end()) {
      return {false, "NUMA logical CPU lists must be unique and sorted"};
    }
    for (const std::uint32_t cpu : node.logical_cpus) {
      const auto processor = std::lower_bound(
          record.logical_processors.begin(), record.logical_processors.end(),
          cpu, [](const CpuLogicalProcessorV1 &candidate, std::uint32_t id) {
            return candidate.logical_cpu < id;
          });
      if (processor == record.logical_processors.end() ||
          processor->logical_cpu != cpu ||
          processor->numa_node_id != node.node_id) {
        return {false, "NUMA mapping disagrees with logical CPU records"};
      }
    }
  }
  for (const CpuLogicalProcessorV1 &processor : record.logical_processors) {
    if (processor.numa_node_id == kUnknownTopologyIdV1) continue;
    const auto node = std::lower_bound(
        record.numa_nodes.begin(), record.numa_nodes.end(),
        processor.numa_node_id,
        [](const CpuNumaNodeV1 &candidate, std::uint32_t id) {
          return candidate.node_id < id;
        });
    if (node == record.numa_nodes.end() ||
        node->node_id != processor.numa_node_id ||
        !contains_cpu(node->logical_cpus, processor.logical_cpu)) {
      return {false, "logical CPU references an absent NUMA node mapping"};
    }
  }

  for (const CpuCacheGroupV1 &cache : record.cache_groups) {
    if (!known(cache.type) || cache.type == CpuCacheTypeV1::unknown ||
        cache.level == 0 || cache.size_bytes == 0 ||
        cache.line_size_bytes == 0 || cache.shared_logical_cpus.empty()) {
      return {false, "CPU cache group is incomplete or invalid"};
    }
    if (!std::is_sorted(cache.shared_logical_cpus.begin(),
                        cache.shared_logical_cpus.end()) ||
        std::adjacent_find(cache.shared_logical_cpus.begin(),
                           cache.shared_logical_cpus.end()) !=
            cache.shared_logical_cpus.end()) {
      return {false, "cache-sharing CPU lists must be unique and sorted"};
    }
    for (const std::uint32_t cpu : cache.shared_logical_cpus) {
      const auto processor = std::lower_bound(
          record.logical_processors.begin(), record.logical_processors.end(),
          cpu, [](const CpuLogicalProcessorV1 &candidate, std::uint32_t id) {
            return candidate.logical_cpu < id;
          });
      if (processor == record.logical_processors.end() ||
          processor->logical_cpu != cpu) {
        return {false, "cache group references an absent logical CPU"};
      }
    }
  }

  if (record.discovery_complete &&
      (record.architecture == ArchitectureKindV1::unknown ||
       record.logical_processors.empty() || record.numa_nodes.empty() ||
       record.cache_groups.empty())) {
    return {false, "complete CPU topology requires architecture, CPU, NUMA, and cache records"};
  }
  return {true, {}};
}

CpuTopologyV1 discover_linux_cpu_topology_v1(
    const std::filesystem::path &sys_devices_root) {
  CpuTopologyV1 record;
  record.architecture = compile_architecture();
  bool complete = record.architecture != ArchitectureKindV1::unknown;
  const std::filesystem::path cpu_root = sys_devices_root / "cpu";
  const std::filesystem::path node_root = sys_devices_root / "node";

  std::vector<std::uint32_t> online_cpus;
  if (const auto online = read_text(cpu_root / "online")) {
    const auto parsed = parse_cpu_list(*online);
    if (parsed) {
      online_cpus = *parsed;
    } else {
      complete = false;
    }
  } else {
    complete = false;
    std::error_code error;
    for (const auto &entry : std::filesystem::directory_iterator(cpu_root, error)) {
      if (const auto id = id_from_name(entry.path().filename().string(), "cpu")) {
        online_cpus.push_back(*id);
      }
    }
    std::sort(online_cpus.begin(), online_cpus.end());
  }

  std::vector<std::uint32_t> online_nodes;
  if (const auto online = read_text(node_root / "online")) {
    const auto parsed = parse_cpu_list(*online);
    if (parsed) {
      online_nodes = *parsed;
    } else {
      complete = false;
    }
  } else {
    complete = false;
    std::error_code error;
    for (const auto &entry : std::filesystem::directory_iterator(node_root, error)) {
      if (const auto id = id_from_name(entry.path().filename().string(), "node")) {
        online_nodes.push_back(*id);
      }
    }
    std::sort(online_nodes.begin(), online_nodes.end());
  }

  for (const std::uint32_t node_id : online_nodes) {
    CpuNumaNodeV1 node;
    node.node_id = node_id;
    const auto cpulist = read_text(node_root /
                                   ("node" + std::to_string(node_id)) /
                                   "cpulist");
    if (cpulist) {
      const auto parsed = parse_cpu_list(*cpulist);
      if (parsed) {
        node.logical_cpus = *parsed;
      } else {
        complete = false;
      }
    } else {
      complete = false;
    }
    record.numa_nodes.push_back(std::move(node));
  }

  for (const std::uint32_t cpu_id : online_cpus) {
    CpuLogicalProcessorV1 processor;
    processor.logical_cpu = cpu_id;
    processor.online = true;
    const std::filesystem::path path =
        cpu_root / ("cpu" + std::to_string(cpu_id));
    const auto core = read_text(path / "topology/core_id");
    const auto package = read_text(path / "topology/physical_package_id");
    const auto core_id = core ? parse_u32(*core) : std::nullopt;
    const auto package_id = package ? parse_u32(*package) : std::nullopt;
    if (core_id) {
      processor.core_id = *core_id;
    } else {
      complete = false;
    }
    if (package_id) {
      processor.package_id = *package_id;
    } else {
      complete = false;
    }

    for (const CpuNumaNodeV1 &node : record.numa_nodes) {
      if (contains_cpu(node.logical_cpus, cpu_id)) {
        if (processor.numa_node_id != kUnknownTopologyIdV1) complete = false;
        processor.numa_node_id = node.node_id;
      }
    }
    if (processor.numa_node_id == kUnknownTopologyIdV1) complete = false;
    record.logical_processors.push_back(processor);
  }
  std::sort(record.logical_processors.begin(),
            record.logical_processors.end(),
            [](const auto &left, const auto &right) {
              return left.logical_cpu < right.logical_cpu;
            });

  for (CpuLogicalProcessorV1 &processor : record.logical_processors) {
    std::uint32_t index = 0;
    for (const CpuLogicalProcessorV1 &candidate : record.logical_processors) {
      if (candidate.package_id == processor.package_id &&
          candidate.core_id == processor.core_id &&
          candidate.logical_cpu < processor.logical_cpu) {
        ++index;
      }
    }
    processor.thread_index = index;
  }

  for (const std::uint32_t cpu_id : online_cpus) {
    const std::filesystem::path cache_root =
        cpu_root / ("cpu" + std::to_string(cpu_id)) / "cache";
    std::error_code error;
    bool found_cache = false;
    for (const auto &entry :
         std::filesystem::directory_iterator(cache_root, error)) {
      if (!id_from_name(entry.path().filename().string(), "index")) continue;
      found_cache = true;
      const auto level_text = read_text(entry.path() / "level");
      const auto type_text = read_text(entry.path() / "type");
      const auto size_text = read_text(entry.path() / "size");
      const auto line_text = read_text(entry.path() / "coherency_line_size");
      const auto shared_text = read_text(entry.path() / "shared_cpu_list");
      const auto level = level_text ? parse_u32(*level_text) : std::nullopt;
      const auto size = size_text ? parse_cache_size(*size_text) : std::nullopt;
      const auto line = line_text ? parse_u32(*line_text) : std::nullopt;
      const auto shared =
          shared_text ? parse_cpu_list(*shared_text) : std::nullopt;
      const CpuCacheTypeV1 type =
          type_text ? parse_cache_type(*type_text) : CpuCacheTypeV1::unknown;
      if (!level || !size || !line || !shared || shared->empty() ||
          type == CpuCacheTypeV1::unknown) {
        complete = false;
        continue;
      }
      CpuCacheGroupV1 cache{*level, type, *size, *line, *shared};
      const auto duplicate = std::find_if(
          record.cache_groups.begin(), record.cache_groups.end(),
          [&cache](const CpuCacheGroupV1 &candidate) {
            return candidate.level == cache.level &&
                   candidate.type == cache.type &&
                   candidate.size_bytes == cache.size_bytes &&
                   candidate.line_size_bytes == cache.line_size_bytes &&
                   candidate.shared_logical_cpus == cache.shared_logical_cpus;
          });
      if (duplicate == record.cache_groups.end()) {
        record.cache_groups.push_back(std::move(cache));
      }
    }
    if (error || !found_cache) complete = false;
  }
  std::sort(record.cache_groups.begin(), record.cache_groups.end(),
            [](const CpuCacheGroupV1 &left, const CpuCacheGroupV1 &right) {
              return std::tie(left.level, left.type,
                              left.shared_logical_cpus, left.size_bytes,
                              left.line_size_bytes) <
                     std::tie(right.level, right.type,
                              right.shared_logical_cpus, right.size_bytes,
                              right.line_size_bytes);
            });

  record.discovery_complete = complete;
  if (!validate_cpu_topology_v1(record)) record.discovery_complete = false;
  return record;
}

std::uint32_t logical_cpu_count_v1(const CpuTopologyV1 &record) noexcept {
  std::uint32_t count = 0;
  for (const CpuLogicalProcessorV1 &processor : record.logical_processors) {
    if (processor.online && count != std::numeric_limits<std::uint32_t>::max()) {
      ++count;
    }
  }
  return count;
}

std::uint32_t physical_core_count_v1(const CpuTopologyV1 &record) noexcept {
  std::uint32_t count = 0;
  for (std::size_t index = 0; index < record.logical_processors.size(); ++index) {
    const auto &processor = record.logical_processors[index];
    if (!processor.online || processor.core_id == kUnknownTopologyIdV1 ||
        processor.package_id == kUnknownTopologyIdV1) {
      continue;
    }
    bool seen = false;
    for (std::size_t prior = 0; prior < index; ++prior) {
      const auto &candidate = record.logical_processors[prior];
      if (candidate.online && candidate.core_id == processor.core_id &&
          candidate.package_id == processor.package_id) {
        seen = true;
        break;
      }
    }
    if (!seen && count != std::numeric_limits<std::uint32_t>::max()) ++count;
  }
  return count;
}

std::uint32_t socket_count_v1(const CpuTopologyV1 &record) noexcept {
  std::uint32_t count = 0;
  for (std::size_t index = 0; index < record.logical_processors.size(); ++index) {
    const auto &processor = record.logical_processors[index];
    if (!processor.online || processor.package_id == kUnknownTopologyIdV1) {
      continue;
    }
    bool seen = false;
    for (std::size_t prior = 0; prior < index; ++prior) {
      const auto &candidate = record.logical_processors[prior];
      if (candidate.online && candidate.package_id == processor.package_id) {
        seen = true;
        break;
      }
    }
    if (!seen && count != std::numeric_limits<std::uint32_t>::max()) ++count;
  }
  return count;
}

std::uint32_t numa_node_count_v1(const CpuTopologyV1 &record) noexcept {
  return record.numa_nodes.size() > std::numeric_limits<std::uint32_t>::max()
             ? std::numeric_limits<std::uint32_t>::max()
             : static_cast<std::uint32_t>(record.numa_nodes.size());
}

CpuPlacementPlanV1 plan_cpu_placement_v1(
    const CpuTopologyV1 &topology,
    const CpuPlacementRequestV1 &request) {
  CpuPlacementPlanV1 plan;
  plan.requested_workers = request.requested_workers;
  plan.affinity = request.affinity;
  plan.smt = request.smt;

  const CpuTopologyValidationV1 validation =
      validate_cpu_topology_v1(topology);
  if (!validation || !topology.discovery_complete) {
    plan.status = CpuPlacementStatusV1::invalid_topology;
    plan.reason = validation ? "CPU topology discovery is incomplete"
                             : std::string(validation.reason);
    return plan;
  }
  if (request.version != kCpuPlacementVersionV1 ||
      request.requested_workers == 0 || !known(request.affinity) ||
      !known(request.smt)) {
    plan.status = CpuPlacementStatusV1::invalid_request;
    plan.reason = "CPU placement request is invalid or unsupported";
    return plan;
  }

  std::set<std::uint32_t> all_nodes;
  for (const CpuNumaNodeV1 &node : topology.numa_nodes) {
    all_nodes.insert(node.node_id);
  }
  if (request.preferred_numa_node != kUnknownTopologyIdV1 &&
      all_nodes.find(request.preferred_numa_node) == all_nodes.end()) {
    plan.status = CpuPlacementStatusV1::invalid_request;
    plan.reason = "preferred NUMA node is absent from the topology";
    return plan;
  }
  if (request.affinity == CpuAffinityPolicyV1::local_first &&
      request.preferred_numa_node == kUnknownTopologyIdV1) {
    plan.status = CpuPlacementStatusV1::invalid_request;
    plan.reason = "local-first placement requires an explicit NUMA node";
    return plan;
  }

  std::set<std::uint32_t> allowed_nodes;
  if (request.allow_cross_numa) {
    allowed_nodes = all_nodes;
  } else if (request.preferred_numa_node != kUnknownTopologyIdV1) {
    allowed_nodes.insert(request.preferred_numa_node);
  } else if (!all_nodes.empty()) {
    allowed_nodes.insert(*all_nodes.begin());
  }

  std::vector<const CpuLogicalProcessorV1 *> candidates =
      topology_candidates(topology, request.smt, allowed_nodes);
  if (request.affinity == CpuAffinityPolicyV1::local_first &&
      request.allow_cross_numa) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [&request](const auto *left, const auto *right) {
                       const bool left_local =
                           left->numa_node_id == request.preferred_numa_node;
                       const bool right_local =
                           right->numa_node_id == request.preferred_numa_node;
                       return left_local != right_local && left_local;
                     });
  } else if (request.affinity == CpuAffinityPolicyV1::scatter) {
    candidates = scatter_candidates(candidates);
  }

  if (candidates.size() < request.requested_workers) {
    if (!request.allow_cross_numa && all_nodes.size() > allowed_nodes.size()) {
      const auto cross_node_candidates =
          topology_candidates(topology, request.smt, all_nodes);
      if (cross_node_candidates.size() >= request.requested_workers) {
        plan.status = CpuPlacementStatusV1::cross_numa_disallowed;
        plan.reason =
            "requested workers require explicit cross-NUMA authorization";
        return plan;
      }
    }
    plan.status = CpuPlacementStatusV1::insufficient_cpus;
    plan.reason = "topology has insufficient legal CPUs for requested workers";
    return plan;
  }

  for (std::uint32_t index = 0; index < request.requested_workers; ++index) {
    plan.logical_cpus.push_back(candidates[index]->logical_cpu);
    plan.numa_nodes.push_back(candidates[index]->numa_node_id);
  }
  std::sort(plan.numa_nodes.begin(), plan.numa_nodes.end());
  plan.numa_nodes.erase(
      std::unique(plan.numa_nodes.begin(), plan.numa_nodes.end()),
      plan.numa_nodes.end());
  plan.status = CpuPlacementStatusV1::selected;
  plan.actual_workers = request.requested_workers;
  plan.crosses_numa_nodes = plan.numa_nodes.size() > 1;
  plan.affinity_application_required = true;
  plan.caller_first_touch_required = request.requested_workers > 1;
  plan.reason =
      "selected deterministic placement; affinity and first-touch remain caller-owned";
  return plan;
}

std::string_view to_string(CpuCacheTypeV1 value) noexcept {
  switch (value) {
    case CpuCacheTypeV1::unknown:
      return "unknown";
    case CpuCacheTypeV1::data:
      return "data";
    case CpuCacheTypeV1::instruction:
      return "instruction";
    case CpuCacheTypeV1::unified:
      return "unified";
  }
  return "invalid";
}

std::string_view to_string(CpuAffinityPolicyV1 value) noexcept {
  switch (value) {
    case CpuAffinityPolicyV1::compact:
      return "compact";
    case CpuAffinityPolicyV1::scatter:
      return "scatter";
    case CpuAffinityPolicyV1::local_first:
      return "local-first";
  }
  return "invalid";
}

std::string_view to_string(CpuSmtPolicyV1 value) noexcept {
  switch (value) {
    case CpuSmtPolicyV1::physical_cores_only:
      return "physical-cores-only";
    case CpuSmtPolicyV1::prefer_physical_cores:
      return "prefer-physical-cores";
    case CpuSmtPolicyV1::allow_smt:
      return "allow-smt";
  }
  return "invalid";
}

std::string_view to_string(CpuPlacementStatusV1 value) noexcept {
  switch (value) {
    case CpuPlacementStatusV1::selected:
      return "selected";
    case CpuPlacementStatusV1::invalid_topology:
      return "invalid-topology";
    case CpuPlacementStatusV1::invalid_request:
      return "invalid-request";
    case CpuPlacementStatusV1::insufficient_cpus:
      return "insufficient-cpus";
    case CpuPlacementStatusV1::cross_numa_disallowed:
      return "cross-numa-disallowed";
  }
  return "invalid";
}

std::string format_cpu_topology_v1(const CpuTopologyV1 &record) {
  std::ostringstream output;
  output << "cpu-topology-v1{version=" << record.version
         << ",architecture=" << to_string(record.architecture)
         << ",discovery="
         << (record.discovery_complete ? "complete" : "incomplete")
         << ",logical-cpus=" << logical_cpu_count_v1(record)
         << ",physical-cores=" << physical_core_count_v1(record)
         << ",sockets=" << socket_count_v1(record)
         << ",numa-nodes=" << numa_node_count_v1(record)
         << ",cpu-map=[";
  bool first = true;
  for (const CpuLogicalProcessorV1 &processor : record.logical_processors) {
    if (!first) output << ',';
    first = false;
    output << processor.logical_cpu << ":core=" << processor.core_id
           << "/socket=" << processor.package_id
           << "/node=" << processor.numa_node_id
           << "/thread=" << processor.thread_index;
  }
  output << "],cache-groups=" << record.cache_groups.size() << '}';
  return output.str();
}

std::string format_cpu_placement_v1(const CpuPlacementPlanV1 &plan) {
  std::ostringstream output;
  output << "cpu-placement-v1{version=" << plan.version
         << ",status=" << to_string(plan.status)
         << ",requested=" << plan.requested_workers
         << ",actual=" << plan.actual_workers
         << ",affinity=" << to_string(plan.affinity)
         << ",smt=" << to_string(plan.smt)
         << ",cross-numa=" << (plan.crosses_numa_nodes ? "yes" : "no")
         << ",affinity-application="
         << (plan.affinity_application_required ? "caller" : "none")
         << ",first-touch="
         << (plan.caller_first_touch_required ? "caller" : "not-required")
         << ",cpus=[";
  for (std::size_t index = 0; index < plan.logical_cpus.size(); ++index) {
    if (index != 0) output << ',';
    output << plan.logical_cpus[index];
  }
  output << "],nodes=[";
  for (std::size_t index = 0; index < plan.numa_nodes.size(); ++index) {
    if (index != 0) output << ',';
    output << plan.numa_nodes[index];
  }
  output << "],reason=" << plan.reason << '}';
  return output.str();
}

}  // namespace matcore::mdslc::platform
