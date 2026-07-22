#include "cpu_capability_v2.h"

#include <iostream>
#include <string>
#include <string_view>

namespace platform = matcore::mdslc::platform;

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

constexpr std::uint64_t bit(platform::CpuFeatureV2 feature) {
  return platform::feature_bit(feature);
}

platform::CpuFeatureDomainV2 domain(std::uint64_t available,
                                    std::uint64_t known =
                                        platform::kKnownCpuFeatureBitsV2) {
  return {known, available};
}

platform::CpuCapabilitiesV2 synthetic_aarch64() {
  const std::uint64_t portable =
      bit(platform::CpuFeatureV2::portable_scalar_f32);
  platform::CpuCapabilitiesV2 record;
  record.architecture = platform::ArchitectureKindV1::aarch64;
  record.hardware = domain(portable, portable);
  record.os_enabled = domain(portable, portable);
  record.compiler = domain(portable, portable);
  record.implementation = domain(portable, portable);
  record.runtime_validation = domain(portable, portable);
  return record;
}

platform::CpuCapabilitiesV2 synthetic_avx512() {
  const std::uint64_t portable =
      bit(platform::CpuFeatureV2::portable_scalar_f32);
  const std::uint64_t vector =
      bit(platform::CpuFeatureV2::avx2) |
      bit(platform::CpuFeatureV2::fma) |
      bit(platform::CpuFeatureV2::avx512f) |
      bit(platform::CpuFeatureV2::avx512dq) |
      bit(platform::CpuFeatureV2::avx512bw) |
      bit(platform::CpuFeatureV2::avx512vl);
  platform::CpuCapabilitiesV2 record;
  record.architecture = platform::ArchitectureKindV1::x86_64;
  record.hardware = domain(portable | vector);
  record.os_enabled = domain(portable | vector);
  record.compiler = domain(portable | vector);
  record.implementation = domain(portable | vector);
  record.runtime_validation = domain(portable);
  record.os_xstate_mask_known = true;
  record.os_xstate_mask = (UINT64_C(1) << 1) | (UINT64_C(1) << 2) |
                          (UINT64_C(1) << 5) | (UINT64_C(1) << 6) |
                          (UINT64_C(1) << 7);
  record.amx_permission_known = true;
  record.usable_vector_bits = 512;
  return record;
}

}  // namespace

int main() {
  static_assert(platform::kCpuFeatureOrderV2.front() ==
                platform::CpuFeatureV2::portable_scalar_f32);
  static_assert(platform::kCpuFeatureOrderV2.back() ==
                platform::CpuFeatureV2::amx_int8);
  static_assert(platform::kCpuFeatureOrderV2.size() == 12);

  const auto aarch64 = synthetic_aarch64();
  expect(platform::validate_cpu_capabilities_v2(aarch64).valid,
         "synthetic AArch64 portable record validates");
  expect(platform::has_runtime_validated_feature_v2(
             aarch64, platform::CpuFeatureV2::portable_scalar_f32),
         "synthetic AArch64 portable implementation is validated");
  expect(!platform::has_usable_feature_v2(aarch64,
                                          platform::CpuFeatureV2::avx2),
         "AArch64 record never exposes an x86 feature");

  const auto avx512 = synthetic_avx512();
  expect(platform::validate_cpu_capabilities_v2(avx512).valid,
         "synthetic AVX-512 record validates");
  expect(platform::has_usable_feature_v2(avx512,
                                         platform::CpuFeatureV2::avx512f),
         "AVX-512 is usable only across all legality domains");
  expect(!platform::has_runtime_validated_feature_v2(
             avx512, platform::CpuFeatureV2::avx512f),
         "compile-capable AVX-512 is distinct from runtime validation");

  auto unknown_feature = avx512;
  unknown_feature.hardware.known |= UINT64_C(1) << 63;
  expect(!platform::validate_cpu_capabilities_v2(unknown_feature).valid,
         "unknown feature bits fail closed under v2");

  auto os_exceeds_hardware = avx512;
  os_exceeds_hardware.hardware.available &=
      ~bit(platform::CpuFeatureV2::avx512dq);
  expect(!platform::validate_cpu_capabilities_v2(os_exceeds_hardware).valid,
         "OS state cannot exceed hardware support");

  auto missing_xstate = avx512;
  missing_xstate.os_xstate_mask = (UINT64_C(1) << 1) | (UINT64_C(1) << 2);
  expect(!platform::validate_cpu_capabilities_v2(missing_xstate).valid,
         "AVX-512 requires complete ZMM XSTATE");

  auto unimplemented = avx512;
  unimplemented.implementation.available &=
      ~(bit(platform::CpuFeatureV2::avx512f) |
        bit(platform::CpuFeatureV2::avx512dq) |
        bit(platform::CpuFeatureV2::avx512bw) |
        bit(platform::CpuFeatureV2::avx512vl));
  unimplemented.usable_vector_bits = 256;
  expect(platform::validate_cpu_capabilities_v2(unimplemented).valid,
         "hardware support remains valid when implementation is absent");
  expect(!platform::has_usable_feature_v2(
             unimplemented, platform::CpuFeatureV2::avx512f),
         "implementation availability participates in usability");

  auto false_validation = unimplemented;
  false_validation.runtime_validation.available |=
      bit(platform::CpuFeatureV2::avx512f);
  expect(!platform::validate_cpu_capabilities_v2(false_validation).valid,
         "runtime validation cannot exceed usable implementation state");

  auto wrong_architecture = aarch64;
  wrong_architecture.hardware.known |= bit(platform::CpuFeatureV2::avx2);
  wrong_architecture.hardware.available |= bit(platform::CpuFeatureV2::avx2);
  expect(!platform::validate_cpu_capabilities_v2(wrong_architecture).valid,
         "AArch64 record rejects x86 hardware claims");

  auto malformed_amx = avx512;
  const std::uint64_t amx = bit(platform::CpuFeatureV2::amx_tile) |
                            bit(platform::CpuFeatureV2::amx_bf16);
  malformed_amx.hardware.available |= amx;
  malformed_amx.os_enabled.available |= amx;
  malformed_amx.compiler.available |= amx;
  malformed_amx.implementation.available |= amx;
  malformed_amx.os_xstate_mask |=
      (UINT64_C(1) << 17) | (UINT64_C(1) << 18);
  malformed_amx.amx_permission_known = true;
  malformed_amx.amx_permission_granted = false;
  expect(!platform::validate_cpu_capabilities_v2(malformed_amx).valid,
         "AMX OS usability requires explicit per-process permission");

  auto bad_version = avx512;
  ++bad_version.version;
  expect(!platform::validate_cpu_capabilities_v2(bad_version).valid,
         "future capability version fails closed");

  platform::CpuCapabilitiesV2 unknown;
  expect(platform::validate_cpu_capabilities_v2(unknown).valid,
         "well-formed unknown CPU record remains representable");

  platform::CpuImplementationAvailabilityV2 availability;
  availability.compiled = domain(
      bit(platform::CpuFeatureV2::portable_scalar_f32) |
      bit(platform::CpuFeatureV2::avx2) |
      bit(platform::CpuFeatureV2::fma));
  availability.runtime_validated = domain(
      bit(platform::CpuFeatureV2::portable_scalar_f32));
  const platform::CpuCapabilitiesV2 detected =
      platform::discover_cpu_capabilities_v2(availability);
  const auto detected_validation =
      platform::validate_cpu_capabilities_v2(detected);
  expect(detected_validation.valid,
         "physical capability discovery produces a valid record");
  expect(platform::feature_available(
             detected.hardware,
             platform::CpuFeatureV2::portable_scalar_f32),
         "physical discovery always exposes portable scalar hardware");
  expect(platform::feature_available(
             detected.os_enabled,
             platform::CpuFeatureV2::portable_scalar_f32),
         "physical discovery always exposes portable scalar OS state");
  expect(platform::has_runtime_validated_feature_v2(
             detected, platform::CpuFeatureV2::portable_scalar_f32),
         "injected portable validation remains explicit");

#if defined(__x86_64__) || defined(_M_X64)
  expect(detected.architecture == platform::ArchitectureKindV1::x86_64,
         "x86-64 host reports x86-64 architecture");
  expect(platform::domain_complete_v2(detected.hardware,
                                      detected.architecture),
         "x86 CPUID hardware discovery is complete for the v2 vocabulary");
  expect(detected.os_xstate_mask_known,
         "x86 validation host exposes XCR0 state");
  if (platform::feature_available(detected.hardware,
                                  platform::CpuFeatureV2::avx2)) {
    expect(platform::feature_available(detected.os_enabled,
                                       platform::CpuFeatureV2::avx2),
           "host AVX2 hardware is OS-enabled before usability");
  }
#endif

  const std::string first = platform::format_cpu_capabilities_v2(detected);
  const std::string second = platform::format_cpu_capabilities_v2(detected);
  expect(first == second, "capability diagnostics are deterministic");
  const auto portable_position = first.find("portable-scalar-f32:");
  const auto avx2_position = first.find("avx2:");
  const auto avx512_position = first.find("avx512f:");
  const auto amx_position = first.find("amx-tile:");
  expect(portable_position < avx2_position && avx2_position < avx512_position &&
             avx512_position < amx_position,
         "capability diagnostics preserve canonical feature order");
  expect(first.find("hardware=") != std::string::npos &&
             first.find("/os=") != std::string::npos &&
             first.find("/compiler=") != std::string::npos &&
             first.find("/implementation=") != std::string::npos &&
             first.find("/runtime=") != std::string::npos,
         "diagnostics expose every distinct legality domain");

  if (failures != 0) return 1;
  std::cout << first << '\n';
  std::cout << "CPU capability v2 tests PASS\n";
  return 0;
}
