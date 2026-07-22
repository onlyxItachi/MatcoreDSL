#include "cpu_backend_registry.h"
#include "cpu_gemm_backend.h"
#include "cpu_openblas.h"
#include "cpu_planner_v2.h"
#include "platform.h"

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
  std::uint32_t threads = 1;
  planner::CpuGemmRequestV2 request = planner::CpuGemmRequestV2::automatic;
  bool platform_info = false;
};

void usage(std::ostream &output) {
  output << "usage: matcore-plan --m M --k K --n N [options]\n"
            "\n"
            "Options:\n"
            "  --alignment BYTES  minimum data alignment (default: 4)\n"
            "  --threads COUNT    requested provider thread count (default: 1)\n"
            "  --variant ID       auto or one of the five stable variant IDs\n"
            "                     cpu.reference.f32.v1\n"
            "                     cpu.tiled.f32.v1\n"
            "                     cpu.compiler-vectorized.avx2-fma.f32.v1\n"
            "                     cpu.external.openblas.f32.v1\n"
            "                     cpu.native-packed.avx2-fma.f32.v1\n"
            "                     short legacy aliases remain accepted\n"
            "  --platform-info    print the versioned compile-platform record\n"
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
    options.request = planner::CpuGemmRequestV2::automatic;
  } else if (encoded == "reference" ||
             encoded == "cpu.reference.f32.v1") {
    options.request = planner::CpuGemmRequestV2::force_reference;
  } else if (encoded == "tiled" || encoded == "cpu.tiled.f32.v1") {
    options.request = planner::CpuGemmRequestV2::force_tiled;
  } else if (encoded == "compiler-vectorized" ||
             encoded == "cpu.compiler-vectorized.avx2-fma.f32.v1") {
    options.request = planner::CpuGemmRequestV2::force_compiler_vectorized;
  } else if (encoded == "openblas" ||
             encoded == "cpu.external.openblas.f32.v1") {
    options.request = planner::CpuGemmRequestV2::force_external_openblas;
  } else if (encoded == "native-packed-avx2-fma" ||
             encoded == "cpu.native-packed.avx2-fma.f32.v1") {
    options.request =
        planner::CpuGemmRequestV2::force_native_packed_avx2_fma;
  } else {
    std::cerr << "matcore-plan: unsupported variant '" << encoded
              << "'; expected auto or a registered stable CPU GEMM variant "
                 "ID (see --help)\n";
    return false;
  }
  return true;
}

std::string_view architectureName(planner::CpuArchitectureV1 architecture) {
  switch (architecture) {
    case planner::CpuArchitectureV1::x86_64:
      return "x86_64";
    case planner::CpuArchitectureV1::aarch64:
      return "aarch64";
    case planner::CpuArchitectureV1::unknown:
      return "unknown";
  }
  return "invalid";
}

void printImplementationResources(
    const planner::CpuCapabilitiesV1 &capabilities,
    const planner::CpuGemmImplementationResourcesV1 &resources) {
  const auto provider =
      matcore::mdslc::runtime::openblas_provider_info_v1();
  std::cout << "cpu-implementation-resources-v1"
            << " architecture=" << architectureName(capabilities.architecture)
            << " capability-detection-complete="
            << (capabilities.detection_complete ? "true" : "false")
            << " feature-bits=" << capabilities.features
            << " usable-vector-bits=" << capabilities.usable_vector_bits
            << " requested-threads=" << resources.requested_threads
            << " openblas-linked="
            << (provider.linked ? "true" : "false")
            << " openblas-package=" << provider.package_version
            << " openblas-parallel-model=" << provider.parallel_model
            << " openblas-max-threads=" << provider.maximum_reported_threads
            << " planned-openblas-thread-limit="
            << resources.openblas_maximum_threads
            << " openblas-core=" << provider.runtime_core
            << " openblas-config=[" << provider.runtime_config << ']'
            << " native-packed-compiled="
            << (resources.native_packed_avx2_fma_compiled ? "true" : "false")
            << " native-packed-runtime-usable="
            << (matcore::mdslc::runtime::cpu_packed_avx2_runtime_usable_v1()
                    ? "true"
                    : "false")
            << " native-packed-workspace-valid="
            << (resources.native_packed_workspace_size_valid ? "true"
                                                              : "false")
            << " native-packed-workspace-bytes="
            << resources.native_packed_workspace_bytes
            << " native-packed-workspace-alignment="
            << resources.native_packed_workspace_alignment << '\n';
}

std::optional<Options> parseCommandLine(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    std::optional<std::string_view> value;
    if (argument == "--help" || argument == "-h") {
      usage(std::cout);
      std::exit(0);
    } else if (argument == "--platform-info") {
      options.platform_info = true;
    } else if (argument == "--m" || argument == "--k" || argument == "--n" ||
               argument == "--alignment" || argument == "--threads" ||
               argument == "--variant") {
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
    } else if (argument == "--threads") {
      if (!parsePositive(*value, options.threads)) {
        std::cerr << "matcore-plan: threads must be a positive uint32\n";
        return std::nullopt;
      }
    } else if (argument == "--variant") {
      if (!setVariant(*value, options)) {
        return std::nullopt;
      }
    } else if (argument != "--help" && argument != "-h" &&
               argument != "--platform-info") {
      std::cerr << "matcore-plan: unknown option: " << argument << '\n';
      return std::nullopt;
    }
  }
  if (!options.platform_info &&
      (options.m == 0 || options.k == 0 || options.n == 0)) {
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
  if (options->platform_info) {
    const auto record =
        matcore::mdslc::platform::discover_compile_platform_v1();
    const auto validation =
        matcore::mdslc::platform::validate_platform_record_v1(record);
    std::cout << matcore::mdslc::platform::format_platform_record_v1(record)
              << '\n';
    return validation ? 0 : 1;
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
  const planner::CpuCapabilitiesV1 capabilities =
      planner::discover_cpu_capabilities_v1();
  const planner::CpuGemmImplementationResourcesV1 resources =
      matcore::mdslc::runtime::discover_cpu_gemm_implementation_resources_v1(
          problem, options->threads);
  const planner::CpuGemmPlanV2 plan = planner::plan_cpu_gemm_v2(
      problem, capabilities, resources, options->request);
  const std::size_t required =
      planner::format_cpu_gemm_plan_v2(plan, nullptr, 0);
  std::string diagnostic(required + 1, '\0');
  planner::format_cpu_gemm_plan_v2(plan, diagnostic.data(), diagnostic.size());
  diagnostic.resize(required);
  printImplementationResources(capabilities, resources);
  std::cout << diagnostic << '\n';
  return plan.status == planner::CpuPlanStatusV1::selected ? 0 : 1;
}
