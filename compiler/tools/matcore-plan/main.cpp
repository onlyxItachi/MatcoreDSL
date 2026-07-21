#include "cpu_planner.h"

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

namespace planner = matcore::mdslc::planner;

struct Options {
  std::int64_t m = 0;
  std::int64_t k = 0;
  std::int64_t n = 0;
  std::uint32_t alignment = alignof(float);
  planner::CpuGemmRequestV1 request =
      planner::CpuGemmRequestV1::automatic;
};

void usage(std::ostream &output) {
  output << "usage: matcore-plan --m M --k K --n N [options]\n"
            "\n"
            "Options:\n"
            "  --alignment BYTES  minimum data alignment (default: 4)\n"
            "  --variant NAME     auto, reference, tiled, or "
            "compiler-vectorized\n"
            "  --help             show this help\n";
}

template <typename Integer>
bool parsePositive(std::string_view encoded, Integer &value) {
  if (encoded.empty()) {
    return false;
  }
  Integer parsed = 0;
  const auto result = std::from_chars(encoded.data(),
                                      encoded.data() + encoded.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != encoded.data() + encoded.size() ||
      parsed <= 0) {
    return false;
  }
  value = parsed;
  return true;
}

std::optional<std::string_view> takeValue(int argc, char **argv, int &index,
                                          std::string_view option) {
  if (index + 1 >= argc) {
    std::cerr << "matcore-plan: " << option << " requires a value\n";
    return std::nullopt;
  }
  return std::string_view(argv[++index]);
}

bool setVariant(std::string_view encoded, Options &options) {
  if (encoded == "auto") {
    options.request = planner::CpuGemmRequestV1::automatic;
  } else if (encoded == "reference") {
    options.request = planner::CpuGemmRequestV1::force_reference;
  } else if (encoded == "tiled") {
    options.request = planner::CpuGemmRequestV1::force_tiled;
  } else if (encoded == "compiler-vectorized") {
    options.request = planner::CpuGemmRequestV1::force_compiler_vectorized;
  } else {
    std::cerr << "matcore-plan: unsupported variant '" << encoded
              << "'; expected auto, reference, tiled, or "
                 "compiler-vectorized\n";
    return false;
  }
  return true;
}

std::optional<Options> parseCommandLine(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    std::optional<std::string_view> value;
    if (argument == "--help" || argument == "-h") {
      usage(std::cout);
      std::exit(0);
    } else if (argument == "--m" || argument == "--k" || argument == "--n" ||
               argument == "--alignment" || argument == "--variant") {
      value = takeValue(argc, argv, index, argument);
      if (!value) {
        return std::nullopt;
      }
    }

    if (argument == "--m") {
      if (!parsePositive(*value, options.m)) {
        std::cerr << "matcore-plan: M must be a positive int64\n";
        return std::nullopt;
      }
    } else if (argument == "--k") {
      if (!parsePositive(*value, options.k)) {
        std::cerr << "matcore-plan: K must be a positive int64\n";
        return std::nullopt;
      }
    } else if (argument == "--n") {
      if (!parsePositive(*value, options.n)) {
        std::cerr << "matcore-plan: N must be a positive int64\n";
        return std::nullopt;
      }
    } else if (argument == "--alignment") {
      if (!parsePositive(*value, options.alignment) ||
          options.alignment < alignof(float) ||
          (options.alignment & (options.alignment - 1U)) != 0) {
        std::cerr << "matcore-plan: alignment must be a power of two and at "
                     "least 4 bytes\n";
        return std::nullopt;
      }
    } else if (argument == "--variant") {
      if (!setVariant(*value, options)) {
        return std::nullopt;
      }
    } else if (argument != "--help" && argument != "-h") {
      std::cerr << "matcore-plan: unknown option: " << argument << '\n';
      return std::nullopt;
    }
  }
  if (options.m == 0 || options.k == 0 || options.n == 0) {
    std::cerr << "matcore-plan: --m, --k, and --n are required\n";
    return std::nullopt;
  }
  return options;
}

}  // namespace

int main(int argc, char **argv) {
  const std::optional<Options> options = parseCommandLine(argc, argv);
  if (!options) {
    usage(std::cerr);
    return 2;
  }

  const planner::CpuGemmProblemV1 problem{
      options->m,
      options->n,
      options->k,
      planner::CpuScalarTypeV1::f32,
      planner::CpuScalarTypeV1::f32,
      planner::CpuLayoutV1::row_major_contiguous,
      options->alignment,
  };
  const planner::CpuGemmPlanV1 plan = planner::plan_cpu_gemm_v1(
      problem, planner::discover_cpu_capabilities_v1(), options->request);
  const std::size_t required =
      planner::format_cpu_gemm_plan_v1(plan, nullptr, 0);
  std::string diagnostic(required + 1, '\0');
  planner::format_cpu_gemm_plan_v1(plan, diagnostic.data(), diagnostic.size());
  diagnostic.resize(required);
  std::cout << diagnostic << '\n';
  return plan.status == planner::CpuPlanStatusV1::selected ? 0 : 1;
}
