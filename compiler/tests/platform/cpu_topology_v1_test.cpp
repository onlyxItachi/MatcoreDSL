#include "cpu_topology_v1.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace platform = matcore::mdslc::platform;

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

platform::CpuTopologyV1 synthetic_topology() {
  platform::CpuTopologyV1 record;
  record.architecture = platform::ArchitectureKindV1::x86_64;
  record.discovery_complete = true;
  record.logical_processors = {
      {0, 0, 0, 0, 0, true}, {1, 1, 0, 0, 0, true},
      {2, 0, 1, 1, 0, true}, {3, 1, 1, 1, 0, true},
      {4, 0, 0, 0, 1, true}, {5, 1, 0, 0, 1, true},
      {6, 0, 1, 1, 1, true}, {7, 1, 1, 1, 1, true},
  };
  record.numa_nodes = {{0, {0, 1, 4, 5}}, {1, {2, 3, 6, 7}}};
  record.cache_groups = {
      {1, platform::CpuCacheTypeV1::data, 32768, 64, {0, 4}},
      {1, platform::CpuCacheTypeV1::data, 32768, 64, {1, 5}},
      {1, platform::CpuCacheTypeV1::data, 32768, 64, {2, 6}},
      {1, platform::CpuCacheTypeV1::data, 32768, 64, {3, 7}},
      {3, platform::CpuCacheTypeV1::unified, 8388608, 64, {0, 1, 4, 5}},
      {3, platform::CpuCacheTypeV1::unified, 8388608, 64, {2, 3, 6, 7}},
  };
  return record;
}

void write_file(const std::filesystem::path &path, std::string_view value) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  output << value;
}

std::filesystem::path make_synthetic_sysfs() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "matcore-cpu-topology-v1-synthetic";
  std::error_code error;
  std::filesystem::remove_all(root, error);
  write_file(root / "cpu/online", "0-3\n");
  write_file(root / "node/online", "0-1\n");
  write_file(root / "node/node0/cpulist", "0-1\n");
  write_file(root / "node/node1/cpulist", "2-3\n");
  for (std::uint32_t cpu = 0; cpu < 4; ++cpu) {
    const std::filesystem::path cpu_path =
        root / "cpu" / ("cpu" + std::to_string(cpu));
    write_file(cpu_path / "topology/core_id",
               std::to_string(cpu % 2) + "\n");
    write_file(cpu_path / "topology/physical_package_id",
               std::to_string(cpu / 2) + "\n");
    write_file(cpu_path / "cache/index0/level", "1\n");
    write_file(cpu_path / "cache/index0/type", "Data\n");
    write_file(cpu_path / "cache/index0/size", "32K\n");
    write_file(cpu_path / "cache/index0/coherency_line_size", "64\n");
    write_file(cpu_path / "cache/index0/shared_cpu_list",
               std::to_string(cpu) + "\n");
  }
  return root;
}

}  // namespace

int main() {
  const platform::CpuTopologyV1 synthetic = synthetic_topology();
  expect(platform::validate_cpu_topology_v1(synthetic).valid,
         "synthetic two-node topology validates");
  expect(platform::logical_cpu_count_v1(synthetic) == 8,
         "synthetic topology reports eight logical CPUs");
  expect(platform::physical_core_count_v1(synthetic) == 4,
         "synthetic topology reports four physical cores");
  expect(platform::socket_count_v1(synthetic) == 2,
         "synthetic topology reports two sockets");
  expect(platform::numa_node_count_v1(synthetic) == 2,
         "synthetic topology reports two NUMA nodes");

  const auto sibling_restriction =
      platform::restrict_cpu_topology_v1(synthetic, {0, 4});
  expect(sibling_restriction &&
             platform::logical_cpu_count_v1(sibling_restriction.topology) == 2 &&
             platform::physical_core_count_v1(sibling_restriction.topology) == 1 &&
             platform::numa_node_count_v1(sibling_restriction.topology) == 1 &&
             sibling_restriction.topology.logical_processors[0].thread_index == 0 &&
             sibling_restriction.topology.logical_processors[1].thread_index == 1,
         "topology restriction preserves one allowed SMT sibling pair");
  expect(sibling_restriction.topology.numa_nodes[0].logical_cpus ==
             std::vector<std::uint32_t>({0, 4}) &&
             std::all_of(
                 sibling_restriction.topology.cache_groups.begin(),
                 sibling_restriction.topology.cache_groups.end(),
                 [](const platform::CpuCacheGroupV1 &cache) {
                   return std::all_of(
                       cache.shared_logical_cpus.begin(),
                       cache.shared_logical_cpus.end(),
                       [](std::uint32_t cpu) { return cpu == 0 || cpu == 4; });
                 }),
         "topology restriction recomputes NUMA and cache memberships");

  const auto secondary_only =
      platform::restrict_cpu_topology_v1(synthetic, {5, 4});
  expect(secondary_only &&
             platform::physical_core_count_v1(secondary_only.topology) == 2 &&
             secondary_only.topology.logical_processors[0].logical_cpu == 4 &&
             secondary_only.topology.logical_processors[0].thread_index == 0 &&
             secondary_only.topology.logical_processors[1].logical_cpu == 5 &&
             secondary_only.topology.logical_processors[1].thread_index == 0,
         "remaining SMT siblings become deterministic primary threads");
  platform::CpuPlacementRequestV1 restricted_physical_request;
  restricted_physical_request.requested_workers = 2;
  restricted_physical_request.smt =
      platform::CpuSmtPolicyV1::physical_cores_only;
  const auto restricted_physical = platform::plan_cpu_placement_v1(
      secondary_only.topology, restricted_physical_request);
  expect(restricted_physical.status ==
                 platform::CpuPlacementStatusV1::selected &&
             restricted_physical.logical_cpus ==
                 std::vector<std::uint32_t>({4, 5}),
         "physical-only placement uses permitted secondary siblings safely");

  const auto invalid_restriction =
      platform::restrict_cpu_topology_v1(synthetic, {0, 99});
  expect(invalid_restriction.status ==
                 platform::CpuTopologyRestrictionStatusV1::unavailable_cpu_id &&
             invalid_restriction.topology.logical_processors.empty(),
         "topology restriction rejects unavailable CPU IDs without partial output");
  const auto duplicate_restriction =
      platform::restrict_cpu_topology_v1(synthetic, {0, 0});
  expect(duplicate_restriction.status ==
             platform::CpuTopologyRestrictionStatusV1::duplicate_cpu_id,
         "topology restriction rejects duplicate CPU IDs");
  const auto empty_restriction =
      platform::restrict_cpu_topology_v1(synthetic, {});
  expect(empty_restriction.status ==
             platform::CpuTopologyRestrictionStatusV1::empty_cpu_set,
         "topology restriction rejects an empty CPU set");

  platform::CpuPlacementRequestV1 compact_request;
  compact_request.requested_workers = 2;
  const platform::CpuPlacementPlanV1 compact =
      platform::plan_cpu_placement_v1(synthetic, compact_request);
  expect(compact.status == platform::CpuPlacementStatusV1::selected,
         "compact placement selects a legal plan");
  expect(compact.logical_cpus == std::vector<std::uint32_t>({0, 1}),
         "compact placement selects primary threads on one node first");
  expect(!compact.crosses_numa_nodes,
         "default compact placement never crosses NUMA nodes silently");

  platform::CpuPlacementRequestV1 scatter_request;
  scatter_request.requested_workers = 4;
  scatter_request.affinity = platform::CpuAffinityPolicyV1::scatter;
  scatter_request.smt = platform::CpuSmtPolicyV1::physical_cores_only;
  scatter_request.allow_cross_numa = true;
  const platform::CpuPlacementPlanV1 scatter =
      platform::plan_cpu_placement_v1(synthetic, scatter_request);
  expect(scatter.status == platform::CpuPlacementStatusV1::selected,
         "explicit cross-node scatter placement selects");
  expect(scatter.logical_cpus ==
             std::vector<std::uint32_t>({0, 2, 1, 3}),
         "scatter placement round-robins deterministic socket/node buckets");
  expect(scatter.crosses_numa_nodes,
         "scatter diagnostics report cross-NUMA placement");
  expect(scatter.caller_first_touch_required,
         "multi-worker placement keeps first-touch caller-owned");

  platform::CpuPlacementRequestV1 local_request;
  local_request.requested_workers = 3;
  local_request.affinity = platform::CpuAffinityPolicyV1::local_first;
  local_request.preferred_numa_node = 1;
  local_request.allow_cross_numa = true;
  const platform::CpuPlacementPlanV1 local =
      platform::plan_cpu_placement_v1(synthetic, local_request);
  expect(local.status == platform::CpuPlacementStatusV1::selected,
         "local-first placement selects");
  expect(local.logical_cpus == std::vector<std::uint32_t>({2, 3, 6}),
         "local-first exhausts explicitly preferred node before crossing");
  expect(!local.crosses_numa_nodes,
         "local-first does not claim a crossing when local SMT is sufficient");

  auto creator_node_compact_request = compact_request;
  creator_node_compact_request.preferred_numa_node = 1;
  const auto creator_node_compact = platform::plan_cpu_placement_v1(
      synthetic, creator_node_compact_request);
  expect(creator_node_compact.status ==
                 platform::CpuPlacementStatusV1::selected &&
             creator_node_compact.affinity ==
                 platform::CpuAffinityPolicyV1::compact &&
             creator_node_compact.logical_cpus ==
                 std::vector<std::uint32_t>({2, 3}),
         "preferred creator node preserves explicit compact ordering");

  auto creator_node_scatter_request = creator_node_compact_request;
  creator_node_scatter_request.affinity =
      platform::CpuAffinityPolicyV1::scatter;
  const auto creator_node_scatter = platform::plan_cpu_placement_v1(
      synthetic, creator_node_scatter_request);
  expect(creator_node_scatter.status ==
                 platform::CpuPlacementStatusV1::selected &&
             creator_node_scatter.affinity ==
                 platform::CpuAffinityPolicyV1::scatter &&
             std::all_of(creator_node_scatter.numa_nodes.begin(),
                         creator_node_scatter.numa_nodes.end(),
                         [](std::uint32_t node) { return node == 1; }),
         "preferred creator node preserves explicit scatter policy within the node");

  platform::CpuPlacementRequestV1 no_cross_request;
  no_cross_request.requested_workers = 3;
  no_cross_request.smt = platform::CpuSmtPolicyV1::physical_cores_only;
  const platform::CpuPlacementPlanV1 no_cross =
      platform::plan_cpu_placement_v1(synthetic, no_cross_request);
  expect(no_cross.status ==
             platform::CpuPlacementStatusV1::cross_numa_disallowed,
         "cross-node capacity requires explicit authorization");
  expect(no_cross.logical_cpus.empty(),
         "rejected placement never emits a partial affinity plan");

  platform::CpuPlacementRequestV1 missing_node_request;
  missing_node_request.affinity = platform::CpuAffinityPolicyV1::local_first;
  missing_node_request.preferred_numa_node = 99;
  const auto missing_node =
      platform::plan_cpu_placement_v1(synthetic, missing_node_request);
  expect(missing_node.status == platform::CpuPlacementStatusV1::invalid_request,
         "absent preferred NUMA node is actionable rejection");

  auto mismatched_node = synthetic;
  mismatched_node.numa_nodes[0].logical_cpus.push_back(2);
  std::sort(mismatched_node.numa_nodes[0].logical_cpus.begin(),
            mismatched_node.numa_nodes[0].logical_cpus.end());
  expect(!platform::validate_cpu_topology_v1(mismatched_node).valid,
         "conflicting CPU-to-node mapping is rejected");

  auto incomplete = synthetic;
  incomplete.discovery_complete = false;
  const auto incomplete_plan =
      platform::plan_cpu_placement_v1(incomplete, compact_request);
  expect(incomplete_plan.status ==
             platform::CpuPlacementStatusV1::invalid_topology,
         "placement fails closed on incomplete discovery");

  auto future = synthetic;
  ++future.version;
  expect(!platform::validate_cpu_topology_v1(future).valid,
         "future topology version fails closed");

  const std::filesystem::path synthetic_root = make_synthetic_sysfs();
  const platform::CpuTopologyV1 discovered_synthetic =
      platform::discover_linux_cpu_topology_v1(synthetic_root);
  expect(discovered_synthetic.discovery_complete,
         "injected Linux sysfs topology discovery is complete");
  expect(platform::validate_cpu_topology_v1(discovered_synthetic).valid,
         "injected Linux sysfs topology validates");
  expect(platform::logical_cpu_count_v1(discovered_synthetic) == 4 &&
             platform::physical_core_count_v1(discovered_synthetic) == 4 &&
             platform::socket_count_v1(discovered_synthetic) == 2 &&
             platform::numa_node_count_v1(discovered_synthetic) == 2,
         "injected Linux sysfs counts are deterministic");
  std::error_code cleanup_error;
  std::filesystem::remove_all(synthetic_root, cleanup_error);

#if defined(__linux__)
  const platform::CpuTopologyV1 host =
      platform::discover_linux_cpu_topology_v1();
  expect(platform::validate_cpu_topology_v1(host).valid,
         "host Linux topology record validates");
  expect(host.discovery_complete,
         "validation host provides complete CPU/NUMA/cache sysfs topology");
  expect(platform::logical_cpu_count_v1(host) > 0,
         "host topology exposes online logical CPUs");
  expect(platform::physical_core_count_v1(host) > 0,
         "host topology exposes physical cores");
  expect(platform::numa_node_count_v1(host) > 0,
         "host topology exposes at least one NUMA node");
  std::cout << platform::format_cpu_topology_v1(host) << '\n';
#endif

  const std::string first = platform::format_cpu_topology_v1(synthetic);
  const std::string second = platform::format_cpu_topology_v1(synthetic);
  expect(first == second, "topology diagnostics are deterministic");
  const std::string placement_first =
      platform::format_cpu_placement_v1(scatter);
  const std::string placement_second =
      platform::format_cpu_placement_v1(scatter);
  expect(placement_first == placement_second,
         "placement diagnostics are deterministic");
  expect(placement_first.find("cross-numa=yes") != std::string::npos &&
             placement_first.find("affinity-application=caller") !=
                 std::string::npos &&
             placement_first.find("first-touch=caller") != std::string::npos,
         "placement diagnostics expose NUMA, affinity, and memory ownership");

  if (failures != 0) return 1;
  std::cout << first << '\n';
  std::cout << placement_first << '\n';
  std::cout << "CPU topology v1 tests PASS\n";
  return 0;
}
