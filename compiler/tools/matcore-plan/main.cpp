#include "matcore/runtime_c.h"

#include "cpu_backend_registry.h"
#include "cpu_capability_v2.h"
#include "cpu_execution_context.h"
#include "cpu_openblas.h"
#include "cpu_planner_v3.h"
#include "cpu_planner_v3_resources.h"
#include "cpu_topology_v1.h"
#include "platform.h"
#include "thread_affinity_v1.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace planner = matcore::mdslc::planner;
namespace platform = matcore::mdslc::platform;
namespace runtime = matcore::mdslc::runtime;

enum class AffinityOption { none, compact, scatter };
enum class NumaOption { single_node, local_first };

struct Options {
  std::int64_t m = 0;
  std::int64_t k = 0;
  std::int64_t n = 0;
  std::uint32_t alignment = alignof(float);
  std::uint32_t threads = 1;
  std::uint32_t maximum_threads = 0;
  planner::CpuGemmRequestV3 request = planner::CpuGemmRequestV3::automatic;
  AffinityOption affinity = AffinityOption::none;
  NumaOption numa = NumaOption::single_node;
  bool allow_smt = false;
  bool platform_info = false;
};

void usage(std::ostream &output) {
  output << "usage: matcore-plan --m M --k K --n N [options]\n\n"
            "Options:\n"
            "  --alignment BYTES   minimum data alignment (default: 4)\n"
            "  --threads COUNT     requested execution threads (default: 1)\n"
            "  --max-threads COUNT explicit deterministic ceiling\n"
            "  --smt physical|allow (default: physical)\n"
            "  --affinity none|compact|scatter (default: none)\n"
            "  --numa single-node|local-first (default: single-node)\n"
            "  --variant ID        auto or one of the eight stable IDs:\n"
            "                      cpu.reference.f32.v1\n"
            "                      cpu.tiled.f32.v1\n"
            "                      cpu.compiler-vectorized.avx2-fma.f32.v1\n"
            "                      cpu.external.openblas.f32.v1\n"
            "                      cpu.native-packed.avx2-fma.f32.v1\n"
            "                      cpu.native-packed.avx512-fma.f32.v1\n"
            "                      cpu.native-parallel.avx2-fma.f32.v1\n"
            "                      cpu.native-parallel.avx512-fma.f32.v1\n"
            "  --platform-info     print the versioned compile-platform record\n"
            "  --help              show this help\n";
}

template <typename Integer>
bool parse_positive(std::string_view encoded, Integer &value) {
  if (encoded.empty()) return false;
  Integer parsed = 0;
  const auto result = std::from_chars(encoded.data(),
                                      encoded.data() + encoded.size(), parsed);
  if (result.ec != std::errc{} ||
      result.ptr != encoded.data() + encoded.size() || parsed <= 0)
    return false;
  value = parsed;
  return true;
}

std::optional<std::string_view> take_value(int argc, char **argv, int &index,
                                           std::string_view option) {
  if (index + 1 >= argc) {
    std::cerr << "matcore-plan: " << option << " requires a value\n";
    return std::nullopt;
  }
  return std::string_view(argv[++index]);
}

bool set_variant(std::string_view value, Options &options) {
  using Request = planner::CpuGemmRequestV3;
  if (value == "auto")
    options.request = Request::automatic;
  else if (value == "reference" || value == "cpu.reference.f32.v1")
    options.request = Request::force_reference;
  else if (value == "tiled" || value == "cpu.tiled.f32.v1")
    options.request = Request::force_tiled;
  else if (value == "compiler-vectorized" ||
           value == "cpu.compiler-vectorized.avx2-fma.f32.v1")
    options.request = Request::force_compiler_vectorized;
  else if (value == "openblas" ||
           value == "cpu.external.openblas.f32.v1")
    options.request = Request::force_external_openblas;
  else if (value == "native-packed-avx2-fma" ||
           value == "cpu.native-packed.avx2-fma.f32.v1")
    options.request = Request::force_native_packed_avx2_fma;
  else if (value == "native-packed-avx512-fma" ||
           value == "cpu.native-packed.avx512-fma.f32.v1")
    options.request = Request::force_native_packed_avx512_fma;
  else if (value == "native-parallel-avx2-fma" ||
           value == "cpu.native-parallel.avx2-fma.f32.v1")
    options.request = Request::force_native_parallel_avx2_fma;
  else if (value == "native-parallel-avx512-fma" ||
           value == "cpu.native-parallel.avx512-fma.f32.v1")
    options.request = Request::force_native_parallel_avx512_fma;
  else {
    std::cerr << "matcore-plan: unsupported variant '" << value
              << "'; expected auto or a registered stable CPU GEMM variant "
                 "ID (see --help)\n";
    return false;
  }
  return true;
}

std::optional<Options> parse_command_line(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      usage(std::cout);
      std::exit(0);
    }
    if (argument == "--platform-info") {
      options.platform_info = true;
      continue;
    }
    const bool takes_value =
        argument == "--m" || argument == "--k" || argument == "--n" ||
        argument == "--alignment" || argument == "--threads" ||
        argument == "--max-threads" || argument == "--variant" ||
        argument == "--affinity" || argument == "--numa" ||
        argument == "--smt";
    if (!takes_value) {
      std::cerr << "matcore-plan: unknown option: " << argument << '\n';
      return std::nullopt;
    }
    const auto value = take_value(argc, argv, index, argument);
    if (!value) return std::nullopt;

    if (argument == "--m") {
      if (!parse_positive(*value, options.m)) {
        std::cerr << "matcore-plan: M must be a positive int64\n";
        return std::nullopt;
      }
    } else if (argument == "--k") {
      if (!parse_positive(*value, options.k)) {
        std::cerr << "matcore-plan: K must be a positive int64\n";
        return std::nullopt;
      }
    } else if (argument == "--n") {
      if (!parse_positive(*value, options.n)) {
        std::cerr << "matcore-plan: N must be a positive int64\n";
        return std::nullopt;
      }
    } else if (argument == "--alignment") {
      if (!parse_positive(*value, options.alignment) ||
          options.alignment < alignof(float) ||
          (options.alignment & (options.alignment - 1U)) != 0) {
        std::cerr << "matcore-plan: alignment must be a power of two and at "
                     "least 4 bytes\n";
        return std::nullopt;
      }
    } else if (argument == "--threads") {
      if (!parse_positive(*value, options.threads)) {
        std::cerr << "matcore-plan: threads must be a positive uint32\n";
        return std::nullopt;
      }
    } else if (argument == "--max-threads") {
      if (!parse_positive(*value, options.maximum_threads)) {
        std::cerr << "matcore-plan: max-threads must be a positive uint32\n";
        return std::nullopt;
      }
    } else if (argument == "--variant") {
      if (!set_variant(*value, options)) return std::nullopt;
    } else if (argument == "--affinity") {
      if (*value == "none")
        options.affinity = AffinityOption::none;
      else if (*value == "compact")
        options.affinity = AffinityOption::compact;
      else if (*value == "scatter")
        options.affinity = AffinityOption::scatter;
      else {
        std::cerr << "matcore-plan: affinity must be none, compact, or scatter\n";
        return std::nullopt;
      }
    } else if (argument == "--numa") {
      if (*value == "single-node")
        options.numa = NumaOption::single_node;
      else if (*value == "local-first")
        options.numa = NumaOption::local_first;
      else {
        std::cerr << "matcore-plan: numa must be single-node or local-first\n";
        return std::nullopt;
      }
    } else if (argument == "--smt") {
      if (*value == "physical")
        options.allow_smt = false;
      else if (*value == "allow")
        options.allow_smt = true;
      else {
        std::cerr << "matcore-plan: smt must be physical or allow\n";
        return std::nullopt;
      }
    }
  }
  if (!options.platform_info &&
      (options.m == 0 || options.k == 0 || options.n == 0)) {
    std::cerr << "matcore-plan: --m, --k, and --n are required\n";
    return std::nullopt;
  }
  if (options.numa == NumaOption::local_first &&
      options.affinity != AffinityOption::none) {
    std::cerr << "matcore-plan: local-first NUMA policy owns worker placement; "
                 "use --affinity none to avoid a conflicting policy\n";
    return std::nullopt;
  }
  return options;
}

platform::ArchitectureKindV1 architecture_from_c(
    matcore_cpu_architecture_v1 value) noexcept {
  if (value == MATCORE_CPU_ARCHITECTURE_X86_64_V1)
    return platform::ArchitectureKindV1::x86_64;
  if (value == MATCORE_CPU_ARCHITECTURE_AARCH64_V1)
    return platform::ArchitectureKindV1::aarch64;
  return platform::ArchitectureKindV1::unknown;
}

bool query_capabilities(platform::CpuCapabilitiesV2 &result,
                        std::string &error) {
  matcore_cpu_capabilities_v2 output{};
  output.abi_version = MATCORE_RUNTIME_CAPABILITY_ABI_VERSION_V2;
  output.struct_size = sizeof(output);
  const matcore_status_v0 status =
      matcore_runtime_query_cpu_capabilities_v2(&output);
  if (status.code != MATCORE_STATUS_OK_V0) {
    error = status.message == nullptr ? "capability query failed" : status.message;
    return false;
  }
  result.architecture = architecture_from_c(output.architecture);
  result.hardware = {output.hardware_known_features,
                     output.hardware_available_features};
  result.os_enabled = {output.os_known_features,
                       output.os_available_features};
  result.compiler = {output.compiler_known_features,
                     output.compiler_available_features};
  result.implementation = {output.implementation_known_features,
                           output.implementation_available_features};
  result.runtime_validation = {
      output.runtime_validation_known_features,
      output.runtime_validated_features};
  result.os_xstate_mask = output.os_xstate_mask;
  if (output.usable_vector_bits >
      std::numeric_limits<std::uint16_t>::max()) {
    error = "capability query returned an unrepresentable vector width";
    return false;
  }
  result.usable_vector_bits =
      static_cast<std::uint16_t>(output.usable_vector_bits);
  result.os_xstate_mask_known = output.os_xstate_mask_known != 0;
  result.amx_permission_known = output.amx_permission_known != 0;
  result.amx_permission_granted = output.amx_permission_granted != 0;
  const auto validation = platform::validate_cpu_capabilities_v2(result);
  if (!validation) {
    error = std::string(validation.reason);
    return false;
  }
  return true;
}

std::uint32_t worker_ceiling(const Options &options,
                             const platform::CpuTopologyV1 &topology) {
  std::uint32_t ceiling = options.allow_smt
                              ? platform::logical_cpu_count_v1(topology)
                              : platform::physical_core_count_v1(topology);
  if (options.maximum_threads != 0)
    ceiling = std::min(ceiling, options.maximum_threads);
  return std::min(ceiling, options.threads);
}

platform::CpuPlacementPlanV1 placement_for(
    const Options &options, const platform::CpuTopologyV1 &topology,
    std::uint32_t workers, bool &requested) {
  requested = options.affinity != AffinityOption::none ||
              options.numa == NumaOption::local_first;
  platform::CpuPlacementRequestV1 request;
  request.requested_workers = workers;
  request.smt = options.allow_smt ? platform::CpuSmtPolicyV1::allow_smt
                                  : platform::CpuSmtPolicyV1::physical_cores_only;
  request.allow_cross_numa = options.numa == NumaOption::local_first;
  if (options.numa == NumaOption::local_first) {
    request.affinity = platform::CpuAffinityPolicyV1::local_first;
    if (!topology.numa_nodes.empty())
      request.preferred_numa_node = topology.numa_nodes.front().node_id;
  } else if (options.affinity == AffinityOption::scatter) {
    request.affinity = platform::CpuAffinityPolicyV1::scatter;
  } else {
    request.affinity = platform::CpuAffinityPolicyV1::compact;
  }
  if (!requested) {
    platform::CpuPlacementPlanV1 plan;
    plan.status = platform::CpuPlacementStatusV1::selected;
    plan.requested_workers = workers;
    plan.actual_workers = workers;
    plan.affinity = platform::CpuAffinityPolicyV1::compact;
    plan.smt = request.smt;
    if (!topology.numa_nodes.empty())
      plan.numa_nodes.push_back(topology.numa_nodes.front().node_id);
    plan.caller_first_touch_required = workers > 1;
    plan.reason = "scheduler affinity was not requested";
    return plan;
  }
  return platform::plan_cpu_placement_v1(topology, request);
}

platform::CpuTopologyV1 discover_topology(
    platform::ArchitectureKindV1 architecture) {
#if defined(__linux__)
  (void)architecture;
  return platform::discover_linux_cpu_topology_v1();
#else
  platform::CpuTopologyV1 result;
  result.architecture = architecture;
  return result;
#endif
}

planner::CpuPlannerPlacementEvidenceV1 placement_evidence_for(
    const Options &options, const platform::CpuTopologyV1 &topology,
    const platform::CpuPlacementPlanV1 &placement,
    bool process_mask_complete, bool placement_requested,
    const runtime::CpuExecutionContextV1 *context,
    const runtime::CpuWorkerAffinityReportV1 &affinity_report) {
  planner::CpuPlannerPlacementEvidenceV1 result;
  const auto validation = platform::validate_cpu_topology_v1(topology);
  if (!validation || !topology.discovery_complete || !process_mask_complete ||
      topology.numa_nodes.empty() ||
      placement.status != platform::CpuPlacementStatusV1::selected) {
    return result;
  }

  result.affinity_requested = placement_requested;
  result.affinity_applied =
      placement_requested && context != nullptr && affinity_report.complete &&
      affinity_report.status == runtime::CpuWorkerAffinityStatusV1::complete;
  result.affinity = placement.affinity;
  result.numa = options.numa == NumaOption::local_first
                    ? planner::CpuPlannerNumaPolicyV1::local_first
                    : planner::CpuPlannerNumaPolicyV1::single_node;
  result.crosses_numa_nodes = placement.crosses_numa_nodes;
  result.caller_first_touch_required =
      placement.caller_first_touch_required;
  if (placement.numa_nodes.empty() ||
      placement.numa_nodes.size() > result.selected_numa_nodes.size()) {
    return result;
  }
  result.selected_numa_node_count =
      static_cast<std::uint32_t>(placement.numa_nodes.size());
  for (std::size_t index = 0; index < placement.numa_nodes.size(); ++index)
    result.selected_numa_nodes[index] = placement.numa_nodes[index];

  const std::uint32_t local_node = result.selected_numa_nodes.front();
  const auto node = std::find_if(
      topology.numa_nodes.begin(), topology.numa_nodes.end(),
      [local_node](const platform::CpuNumaNodeV1 &candidate) {
        return candidate.node_id == local_node;
      });
  if (node == topology.numa_nodes.end()) return {};
  const auto local =
      platform::restrict_cpu_topology_v1(topology, node->logical_cpus);
  if (!local) return {};
  result.local_logical_processor_capacity =
      platform::logical_cpu_count_v1(local.topology);
  result.local_physical_core_capacity =
      platform::physical_core_count_v1(local.topology);
  result.evidence_complete =
      result.local_logical_processor_capacity != 0 &&
      result.local_physical_core_capacity != 0 &&
      (!placement_requested || result.affinity_applied);
  return result;
}

std::string_view affinity_name(AffinityOption value) noexcept {
  switch (value) {
    case AffinityOption::none: return "none";
    case AffinityOption::compact: return "compact";
    case AffinityOption::scatter: return "scatter";
  }
  return "invalid";
}

std::string_view numa_name(NumaOption value) noexcept {
  return value == NumaOption::local_first ? "local-first" : "single-node";
}

}  // namespace

int main(int argc, char **argv) {
  const auto options = parse_command_line(argc, argv);
  if (!options) {
    usage(std::cerr);
    return 2;
  }
  if (options->platform_info) {
    const auto record = platform::discover_compile_platform_v1();
    std::cout << platform::format_platform_record_v1(record) << '\n';
    return platform::validate_platform_record_v1(record) ? 0 : 1;
  }

  platform::CpuCapabilitiesV2 capabilities;
  std::string capability_error;
  if (!query_capabilities(capabilities, capability_error)) {
    std::cerr << "matcore-plan: " << capability_error << '\n';
    return 1;
  }
  const platform::CpuTopologyV1 system_topology =
      discover_topology(capabilities.architecture);
  platform::CpuTopologyV1 topology = system_topology;
  const auto system_topology_validation =
      platform::validate_cpu_topology_v1(system_topology);
  const auto affinity_inventory =
      platform::discover_current_thread_affinity_v1();
  bool process_mask_complete = false;
  if (system_topology_validation && system_topology.discovery_complete &&
      affinity_inventory.discovery_complete) {
    const auto restricted = platform::restrict_cpu_topology_v1(
        system_topology, affinity_inventory.allowed_logical_cpus);
    if (restricted) {
      topology = restricted.topology;
      process_mask_complete = true;
    } else if (options->affinity != AffinityOption::none ||
               options->numa == NumaOption::local_first) {
      std::cerr << "matcore-plan: process-affinity topology restriction "
                   "failed: "
                << restricted.reason << '\n';
      return 1;
    }
  }
  const auto topology_validation = platform::validate_cpu_topology_v1(topology);
  const bool topology_complete =
      topology_validation && topology.discovery_complete &&
      !topology.numa_nodes.empty();
  const std::uint32_t workers =
      topology_complete ? worker_ceiling(*options, topology) : 0;
  bool placement_requested = false;
  platform::CpuPlacementPlanV1 placement;
  if (topology_complete && workers != 0) {
    placement = placement_for(*options, topology, workers, placement_requested);
  }
  if (placement_requested &&
      placement.status != platform::CpuPlacementStatusV1::selected) {
    std::cerr << "matcore-plan: CPU placement rejected: " << placement.reason
              << '\n';
    return 1;
  }
  runtime::CpuExecutionStatusV1 context_status{};
  runtime::CpuWorkerAffinityReportV1 affinity_report;
  std::unique_ptr<runtime::CpuExecutionContextV1> context;
  if (topology_complete && workers != 0) {
    runtime::CpuExecutionContextConfigV1 context_config;
    context_config.requested_threads = options->threads;
    context_config.maximum_threads = workers;
    if (placement_requested)
      context_config.worker_cpu_ids = placement.logical_cpus;
    context = runtime::CpuExecutionContextV1::create(
        context_config, &context_status, &affinity_report);
  }
  if (!context && placement_requested) {
    std::cerr << "matcore-plan: "
              << runtime::cpu_execution_status_message_v1(context_status)
              << '\n';
    return 1;
  }

  const auto placement_evidence = placement_evidence_for(
      *options, topology, placement, process_mask_complete,
      placement_requested, context.get(), affinity_report);

  const planner::CpuGemmProblemV1 problem{
      options->m, options->n, options->k, planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous, options->alignment};
  const auto baseline =
      runtime::discover_cpu_gemm_implementation_resources_v1(
          problem, options->threads);
  runtime::CpuRuntimeValidationEvidenceV1 evidence;
  evidence.packed_avx2_f32_runtime_validated =
      platform::has_runtime_validated_feature_v2(
          capabilities, platform::CpuFeatureV2::avx2) &&
      platform::has_runtime_validated_feature_v2(
          capabilities, platform::CpuFeatureV2::fma);
  evidence.packed_avx512_f32_runtime_validated =
      platform::has_runtime_validated_feature_v2(
          capabilities, platform::CpuFeatureV2::avx512f) &&
      platform::has_runtime_validated_feature_v2(
          capabilities, platform::CpuFeatureV2::fma);
  const auto resources = runtime::augment_cpu_gemm_implementation_resources_v2(
      problem, baseline, context.get(), evidence);
  planner::CpuThreadPolicyV1 thread_policy;
  thread_policy.requested_threads = options->threads;
  thread_policy.maximum_threads = options->maximum_threads;
  thread_policy.allow_smt = options->allow_smt;
  const auto plan = planner::plan_cpu_gemm_v3(
      problem, capabilities, topology, thread_policy, resources,
      options->request, 0, placement_evidence);
  const std::size_t required =
      planner::format_cpu_gemm_plan_v3(plan, nullptr, 0);
  std::string diagnostic(required + 1, '\0');
  planner::format_cpu_gemm_plan_v3(plan, diagnostic.data(), diagnostic.size());
  diagnostic.resize(required);

  const auto provider = runtime::openblas_provider_info_v1();
  std::cout << platform::format_cpu_capabilities_v2(capabilities) << '\n'
            << platform::format_cpu_topology_v1(topology) << '\n';
  if (placement_requested)
    std::cout << platform::format_cpu_placement_v1(placement) << '\n';
  std::cout << "cpu-execution-policy-v1 requested-threads="
            << options->threads << " actual-workers="
            << (context ? context->info().actual_worker_count : 0)
            << " smt=" << (options->allow_smt ? "allow" : "physical")
            << " smt-placement-enforced="
            << (placement_requested ? "true" : "false")
            << " affinity=" << affinity_name(options->affinity)
            << " affinity-status="
            << runtime::cpu_worker_affinity_status_message_v1(
                   affinity_report.status)
            << " affinity-applied="
            << (placement_evidence.affinity_applied ? "true" : "false")
            << " process-mask-complete="
            << (process_mask_complete ? "true" : "false")
            << " numa=" << numa_name(options->numa)
            << " numa-memory-placement=false"
            << " openblas-linked=" << (provider.linked ? "true" : "false")
            << " openblas-version=" << provider.package_version << '\n'
            << diagnostic << '\n';
  return plan.status == planner::CpuPlanStatusV1::selected ? 0 : 1;
}
