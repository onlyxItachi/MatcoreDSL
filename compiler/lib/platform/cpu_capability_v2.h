#ifndef MATCORE_MDSLC_PLATFORM_CPU_CAPABILITY_V2_H
#define MATCORE_MDSLC_PLATFORM_CPU_CAPABILITY_V2_H

#include "platform.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace matcore::mdslc::platform {

inline constexpr std::uint32_t kCpuCapabilitiesVersionV2 = 2;
inline constexpr std::uint32_t kCpuImplementationAvailabilityVersionV2 = 2;

// The order is a serialization and diagnostic contract. Append future features
// only with a new capability-record version.
enum class CpuFeatureV2 : std::uint64_t {
  portable_scalar_f32 = UINT64_C(1) << 0,
  avx2 = UINT64_C(1) << 1,
  fma = UINT64_C(1) << 2,
  avx512f = UINT64_C(1) << 3,
  avx512dq = UINT64_C(1) << 4,
  avx512bw = UINT64_C(1) << 5,
  avx512vl = UINT64_C(1) << 6,
  avx512vnni = UINT64_C(1) << 7,
  avx512bf16 = UINT64_C(1) << 8,
  amx_tile = UINT64_C(1) << 9,
  amx_bf16 = UINT64_C(1) << 10,
  amx_int8 = UINT64_C(1) << 11,
};

inline constexpr std::array<CpuFeatureV2, 12> kCpuFeatureOrderV2{{
    CpuFeatureV2::portable_scalar_f32,
    CpuFeatureV2::avx2,
    CpuFeatureV2::fma,
    CpuFeatureV2::avx512f,
    CpuFeatureV2::avx512dq,
    CpuFeatureV2::avx512bw,
    CpuFeatureV2::avx512vl,
    CpuFeatureV2::avx512vnni,
    CpuFeatureV2::avx512bf16,
    CpuFeatureV2::amx_tile,
    CpuFeatureV2::amx_bf16,
    CpuFeatureV2::amx_int8,
}};

constexpr std::uint64_t feature_bit(CpuFeatureV2 feature) noexcept {
  return static_cast<std::uint64_t>(feature);
}

inline constexpr std::uint64_t kKnownCpuFeatureBitsV2 = [] {
  std::uint64_t bits = 0;
  for (const CpuFeatureV2 feature : kCpuFeatureOrderV2) {
    bits |= feature_bit(feature);
  }
  return bits;
}();

inline constexpr std::uint64_t kX86CpuFeatureBitsV2 =
    kKnownCpuFeatureBitsV2 &
    ~feature_bit(CpuFeatureV2::portable_scalar_f32);

// A domain represents three states per feature: unknown (not in known), known
// unavailable, and known available. This prevents incomplete detection from
// being interpreted as an affirmative "unsupported" result.
struct CpuFeatureDomainV2 {
  std::uint64_t known = 0;
  std::uint64_t available = 0;
};

struct CpuImplementationAvailabilityV2 {
  std::uint32_t version = kCpuImplementationAvailabilityVersionV2;
  CpuFeatureDomainV2 compiled;
  CpuFeatureDomainV2 runtime_validated;
};

struct CpuCapabilitiesV2 {
  std::uint32_t version = kCpuCapabilitiesVersionV2;
  ArchitectureKindV1 architecture = ArchitectureKindV1::unknown;

  CpuFeatureDomainV2 hardware;
  CpuFeatureDomainV2 os_enabled;
  CpuFeatureDomainV2 compiler;
  CpuFeatureDomainV2 implementation;
  CpuFeatureDomainV2 runtime_validation;

  // Raw XCR0 is evidence, not a substitute for per-feature OS legality.
  std::uint64_t os_xstate_mask = 0;
  bool os_xstate_mask_known = false;
  bool amx_permission_known = false;
  bool amx_permission_granted = false;
  std::uint16_t usable_vector_bits = 0;
};

struct CpuCapabilitiesValidationV2 {
  bool valid = false;
  std::string_view reason;

  constexpr explicit operator bool() const noexcept { return valid; }
};

constexpr bool feature_known(const CpuFeatureDomainV2 &domain,
                             CpuFeatureV2 feature) noexcept {
  return (domain.known & feature_bit(feature)) != 0;
}

constexpr bool feature_available(const CpuFeatureDomainV2 &domain,
                                 CpuFeatureV2 feature) noexcept {
  return feature_known(domain, feature) &&
         (domain.available & feature_bit(feature)) != 0;
}

constexpr bool has_usable_feature_v2(const CpuCapabilitiesV2 &record,
                                     CpuFeatureV2 feature) noexcept {
  return feature_available(record.hardware, feature) &&
         feature_available(record.os_enabled, feature) &&
         feature_available(record.compiler, feature) &&
         feature_available(record.implementation, feature);
}

constexpr bool has_runtime_validated_feature_v2(
    const CpuCapabilitiesV2 &record, CpuFeatureV2 feature) noexcept {
  return has_usable_feature_v2(record, feature) &&
         feature_available(record.runtime_validation, feature);
}

constexpr std::uint64_t required_feature_bits_v2(
    ArchitectureKindV1 architecture) noexcept {
  switch (architecture) {
    case ArchitectureKindV1::x86_64:
      return kKnownCpuFeatureBitsV2;
    case ArchitectureKindV1::aarch64:
      return feature_bit(CpuFeatureV2::portable_scalar_f32);
    case ArchitectureKindV1::unknown:
      return 0;
  }
  return 0;
}

constexpr bool domain_complete_v2(const CpuFeatureDomainV2 &domain,
                                  ArchitectureKindV1 architecture) noexcept {
  const std::uint64_t required = required_feature_bits_v2(architecture);
  return required != 0 && (domain.known & required) == required;
}

CpuCapabilitiesValidationV2 validate_cpu_capabilities_v2(
    const CpuCapabilitiesV2 &record) noexcept;

CpuImplementationAvailabilityV2
unknown_cpu_implementation_availability_v2() noexcept;

// Normalizes compiler feature support independently from the host that runs
// discovery.  This keeps clang-cl in the Clang capability family while
// retaining the MSVC ABI identity recorded by PlatformRecordV1.  Unknown and
// native MSVC frontends fail closed for ISA-specific code generation; the
// runtime implementation domain remains a separate, mandatory legality gate.
CpuFeatureDomainV2 cpu_compiler_feature_domain_v2(
    ArchitectureKindV1 architecture, CompilerFrontendV1 frontend,
    std::uint32_t compiler_major_version) noexcept;

CpuCapabilitiesV2 discover_cpu_capabilities_v2(
    const CpuImplementationAvailabilityV2 &implementation =
        unknown_cpu_implementation_availability_v2()) noexcept;

std::string_view to_string(CpuFeatureV2 feature) noexcept;
std::string format_cpu_capabilities_v2(const CpuCapabilitiesV2 &record);

}  // namespace matcore::mdslc::platform

#endif
