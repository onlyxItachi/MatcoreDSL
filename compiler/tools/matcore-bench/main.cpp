#include "benchmark.h"

#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace {

namespace bench = matcore::mdslc::bench;

void usage(std::ostream &output) {
  output <<
      "usage: matcore-bench [--m M --n N --k K | --quick | --standard | --full] [options]\n"
      "\n"
      "Problem and selection:\n"
      "  --m M --n N --k K       benchmark one positive GEMM shape\n"
      "  --quick                  bounded CI/correctness profile (default)\n"
      "  --standard               declared local standard profile\n"
      "  --full                   opt-in profile including 4096 and 8192 squares\n"
      "  --variant ID             stable variant ID or auto (default: auto)\n"
      "  --threads N              requested implementation threads (default: 1)\n"
      "  --physical-cores-only    cap planning to physical cores (default)\n"
      "  --allow-smt              explicitly allow logical SMT workers\n"
      "  --affinity POLICY        none|compact|scatter|local-first (default: none)\n"
      "  --alignment BYTES        exact minimum input/output alignment\n"
      "\n"
      "Measurement contract:\n"
      "  --warmup N               untimed warmup executions (default: 2)\n"
      "  --iterations N           measured aggregate samples (default: 9)\n"
      "  --hot-cache              hot-cache aggregate timing (default)\n"
      "  --cold-cache             best-effort 64 MiB eviction before each sample\n"
      "  --include-packing        packing occurs inside each interval (default)\n"
      "  --exclude-packing        prepare packing before measured intervals\n"
      "  --prepack-b              prepack B once; requires variant support\n"
      "  --reuse-workspace        allocate output/workspace before timing (default)\n"
      "  --include-allocation     include plan and output/workspace allocation\n"
      "  --timer-floor-us N       minimum aggregate timer interval (default: 1000)\n"
      "  --max-memory-mib N       hard pre-allocation cap (default: 2048)\n"
      "  --seed N                 deterministic unsigned input seed\n"
      "  --guard                  reject any invalid timing or correctness result\n"
      "  --compare-one-thread     add same-family one-thread speedup/efficiency\n"
      "  --planner-regret         time every legal complete-call candidate; auto only\n"
      "  --json-out PATH          write schema-v2 JSON ('-' writes stdout)\n"
      "  --list-variants          list the runner's registered stable IDs\n"
      "  --help                   show this help\n";
}

template <typename Integer>
bool parse_integer(std::string_view encoded, Integer &value,
                   bool allow_zero = false) {
  Integer parsed{};
  const auto result = std::from_chars(encoded.data(),
                                      encoded.data() + encoded.size(), parsed);
  if (encoded.empty() || result.ec != std::errc{} ||
      result.ptr != encoded.data() + encoded.size() ||
      (!allow_zero && parsed == 0))
    return false;
  value = parsed;
  return true;
}

std::optional<std::string_view> take_value(int argc, char **argv, int &index,
                                           std::string_view option) {
  if (index + 1 >= argc) {
    std::cerr << "matcore-bench: " << option << " requires a value\n";
    return std::nullopt;
  }
  return std::string_view(argv[++index]);
}

struct ParsedCommandLine {
  bench::BenchmarkOptionsV1 options;
  bool list_variants = false;
};

std::optional<ParsedCommandLine> parse_command_line(int argc, char **argv) {
  ParsedCommandLine parsed;
  std::int64_t m = 0, n = 0, k = 0;
  bool profile_seen = false;
  bool custom_seen = false;
  bool cache_seen = false;
  bool packing_seen = false;
  bool allocation_seen = false;
  bool smt_seen = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      usage(std::cout);
      std::exit(0);
    }
    if (argument == "--list-variants") {
      parsed.list_variants = true;
      continue;
    }
    if (argument == "--quick" || argument == "--standard" ||
        argument == "--full") {
      if (profile_seen || custom_seen) {
        std::cerr << "matcore-bench: choose exactly one profile or custom shape\n";
        return std::nullopt;
      }
      profile_seen = true;
      parsed.options.profile = argument == "--quick"
                                   ? bench::ProfileV1::quick
                                   : argument == "--standard"
                                         ? bench::ProfileV1::standard
                                         : bench::ProfileV1::full;
      continue;
    }
    if (argument == "--hot-cache" || argument == "--cold-cache") {
      if (cache_seen) {
        std::cerr << "matcore-bench: cache modes are mutually exclusive\n";
        return std::nullopt;
      }
      cache_seen = true;
      parsed.options.cache_mode = argument == "--cold-cache"
                                      ? bench::CacheModeV1::cold
                                      : bench::CacheModeV1::hot;
      continue;
    }
    if (argument == "--include-packing" || argument == "--exclude-packing" ||
        argument == "--prepack-b") {
      if (packing_seen) {
        std::cerr << "matcore-bench: packing modes are mutually exclusive\n";
        return std::nullopt;
      }
      packing_seen = true;
      parsed.options.packing_mode =
          argument == "--include-packing"
              ? bench::PackingModeV1::include
              : argument == "--exclude-packing"
                    ? bench::PackingModeV1::exclude
                    : bench::PackingModeV1::prepack_b;
      continue;
    }
    if (argument == "--reuse-workspace" || argument == "--include-allocation") {
      if (allocation_seen) {
        std::cerr << "matcore-bench: allocation modes are mutually exclusive\n";
        return std::nullopt;
      }
      allocation_seen = true;
      parsed.options.allocation_mode =
          argument == "--include-allocation"
              ? bench::AllocationModeV1::include_allocation
              : bench::AllocationModeV1::reuse_workspace;
      continue;
    }
    if (argument == "--guard") {
      parsed.options.guard = true;
      continue;
    }
    if (argument == "--compare-one-thread") {
      parsed.options.compare_one_thread = true;
      continue;
    }
    if (argument == "--planner-regret") {
      parsed.options.planner_regret = true;
      continue;
    }
    if (argument == "--allow-smt" ||
        argument == "--physical-cores-only") {
      if (smt_seen) {
        std::cerr << "matcore-bench: SMT policies are mutually exclusive\n";
        return std::nullopt;
      }
      smt_seen = true;
      parsed.options.smt_policy =
          argument == "--allow-smt" ? bench::SmtPolicyV2::allow_smt
                                     : bench::SmtPolicyV2::physical_cores_only;
      continue;
    }

    const bool takes_value =
        argument == "--m" || argument == "--n" || argument == "--k" ||
        argument == "--variant" || argument == "--threads" ||
        argument == "--warmup" || argument == "--iterations" ||
        argument == "--alignment" || argument == "--json-out" ||
        argument == "--affinity" ||
        argument == "--max-memory-mib" || argument == "--timer-floor-us" ||
        argument == "--seed";
    if (!takes_value) {
      std::cerr << "matcore-bench: unknown option: " << argument << '\n';
      return std::nullopt;
    }
    const auto value = take_value(argc, argv, index, argument);
    if (!value) return std::nullopt;

    if (argument == "--m" || argument == "--n" || argument == "--k") {
      if (profile_seen) {
        std::cerr << "matcore-bench: custom dimensions cannot accompany a profile\n";
        return std::nullopt;
      }
      custom_seen = true;
      std::int64_t *destination = argument == "--m" ? &m : argument == "--n" ? &n : &k;
      if (!parse_integer(*value, *destination) || *destination <= 0) {
        std::cerr << "matcore-bench: M, N, and K must be positive int64 values\n";
        return std::nullopt;
      }
    } else if (argument == "--variant") {
      parsed.options.requested_variant = std::string(*value);
    } else if (argument == "--threads") {
      if (!parse_integer(*value, parsed.options.requested_threads)) {
        std::cerr << "matcore-bench: threads must be positive\n";
        return std::nullopt;
      }
    } else if (argument == "--warmup") {
      if (!parse_integer(*value, parsed.options.warmup_iterations, true)) {
        std::cerr << "matcore-bench: warmup must be a nonnegative uint32\n";
        return std::nullopt;
      }
    } else if (argument == "--iterations") {
      if (!parse_integer(*value, parsed.options.measured_iterations)) {
        std::cerr << "matcore-bench: iterations must be positive\n";
        return std::nullopt;
      }
    } else if (argument == "--alignment") {
      if (!parse_integer(*value, parsed.options.alignment_bytes)) {
        std::cerr << "matcore-bench: alignment must be positive\n";
        return std::nullopt;
      }
    } else if (argument == "--affinity") {
      if (*value == "none") {
        parsed.options.affinity_policy = bench::AffinityPolicyV2::none;
      } else if (*value == "compact") {
        parsed.options.affinity_policy = bench::AffinityPolicyV2::compact;
      } else if (*value == "scatter") {
        parsed.options.affinity_policy = bench::AffinityPolicyV2::scatter;
      } else if (*value == "local-first") {
        parsed.options.affinity_policy = bench::AffinityPolicyV2::local_first;
      } else {
        std::cerr << "matcore-bench: affinity must be none, compact, scatter, "
                     "or local-first\n";
        return std::nullopt;
      }
    } else if (argument == "--json-out") {
      parsed.options.json_output = std::string(*value);
    } else if (argument == "--max-memory-mib") {
      std::uint64_t mib = 0;
      if (!parse_integer(*value, mib) ||
          mib > std::numeric_limits<std::uint64_t>::max() / (1024 * 1024)) {
        std::cerr << "matcore-bench: max-memory-mib is invalid\n";
        return std::nullopt;
      }
      parsed.options.maximum_memory_bytes = mib * 1024 * 1024;
    } else if (argument == "--timer-floor-us") {
      std::uint64_t microseconds = 0;
      if (!parse_integer(*value, microseconds) ||
          microseconds > std::numeric_limits<std::uint64_t>::max() / 1000) {
        std::cerr << "matcore-bench: timer-floor-us is invalid\n";
        return std::nullopt;
      }
      parsed.options.timer_floor_nanoseconds = microseconds * 1000;
    } else if (argument == "--seed") {
      if (!parse_integer(*value, parsed.options.seed, true)) {
        std::cerr << "matcore-bench: seed must be an unsigned integer\n";
        return std::nullopt;
      }
    }
  }

  if (custom_seen) {
    if (m == 0 || n == 0 || k == 0) {
      std::cerr << "matcore-bench: custom shape requires --m, --n, and --k\n";
      return std::nullopt;
    }
    parsed.options.profile = bench::ProfileV1::custom;
    parsed.options.shapes = {{m, n, k}};
  }
  return parsed;
}

}  // namespace

int main(int argc, char **argv) {
  const auto parsed = parse_command_line(argc, argv);
  if (!parsed) {
    usage(std::cerr);
    return 2;
  }
  const auto runner = bench::make_planner_runner_v1();
  if (parsed->list_variants) {
    std::cout << "auto\n";
    for (const auto &variant : runner->variant_ids()) std::cout << variant << '\n';
    return 0;
  }

  bench::BenchmarkReportV1 report;
  std::string error;
  if (!bench::run_benchmarks_v1(parsed->options, *runner, report, error)) {
    std::cerr << "matcore-bench: " << error << '\n';
    return 1;
  }
  if (parsed->options.json_output != "-") {
    for (const auto &result : report.results) {
      std::cout << "m=" << result.shape.m << " n=" << result.shape.n
              << " k=" << result.shape.k
              << " variant=" << result.plan.selected_variant
              << " threads=" << result.plan.actual_threads
              << " smt_policy=" << result.plan.smt_policy
              << " affinity_policy=" << result.plan.affinity_policy
              << " worker_affinity_applied="
              << (result.plan.worker_affinity_applied ? "true" : "false")
              << " worker_affinity_origin="
              << (result.plan.worker_affinity_user_requested
                      ? "user-requested"
                      : result.plan.worker_affinity_policy_induced
                            ? "smt-policy"
                            : "none")
              << " workspace_bytes=" << result.plan.workspace_bytes
              << " shared_workspace_bytes="
              << result.plan.shared_workspace_bytes
              << " per_worker_workspace_bytes="
              << result.plan.per_worker_workspace_bytes
              << " comparison="
              << (result.plan.complete_implementation_comparison
                      ? "complete-implementation"
                      : "diagnostic-only")
              << " timing_scope=\"" << result.plan.timing_scope << '"'
              << " median_ms=" << result.timing.median_seconds * 1.0e3
              << " p95_ms=" << result.timing.p95_seconds * 1.0e3
              << " gflops=" << result.gflops
              << " timing=" << (result.timing.valid ? "valid" : "rejected")
              << " correctness="
              << (result.correctness.passed ? "pass" : "FAIL");
      if (result.scaling.requested) {
        std::cout << " speedup=" << result.scaling.speedup_over_one_thread
                  << " efficiency=" << result.scaling.parallel_efficiency
                  << " scaling=" << (result.scaling.valid ? "valid" : "n/a");
      }
      if (result.planner_regret.requested) {
        std::cout << " regret=" << result.planner_regret.regret
                  << " fastest="
                  << result.planner_regret.fastest_legal_variant
                  << " regret_status="
                  << (result.planner_regret.valid ? "valid" : "n/a");
      }
      std::cout << '\n';
    }
  }

  if (!parsed->options.json_output.empty()) {
    if (parsed->options.json_output == "-") {
      bench::write_json_v1(report, std::cout);
    } else {
      std::ofstream output(parsed->options.json_output,
                           std::ios::out | std::ios::trunc);
      if (!output) {
        std::cerr << "matcore-bench: cannot open JSON output: "
                  << parsed->options.json_output << '\n';
        return 1;
      }
      bench::write_json_v1(report, output);
      output.flush();
      if (!output) {
        std::cerr << "matcore-bench: failed to write complete JSON output\n";
        return 1;
      }
    }
  }
  if (parsed->options.guard) {
    for (const auto &result : report.results) {
      if (!result.timing.valid || !result.correctness.passed ||
          (parsed->options.compare_one_thread && !result.scaling.valid) ||
          (parsed->options.planner_regret && !result.planner_regret.valid))
        return 1;
    }
  }
  return 0;
}
