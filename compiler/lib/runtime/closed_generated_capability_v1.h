#ifndef MATCORE_MDSLC_CLOSED_GENERATED_CAPABILITY_V1_H
#define MATCORE_MDSLC_CLOSED_GENERATED_CAPABILITY_V1_H

#include "../platform/cpu_capability_v2.h"

namespace matcore::mdslc::runtime::closed_host_v1::detail {

// This candidate's object is separately issued for baseline x86-64 + AVX2/FMA.
// Inspect direct physical/OS discovery, not another implementation's compiled
// or runtime-selftest domains. This pure predicate creates no source authority.
constexpr bool generatedReassociateCpuSupported(
    const platform::CpuCapabilitiesV2 &capabilities) noexcept {
  using platform::CpuFeatureV2;
  constexpr auto required = platform::feature_bit(CpuFeatureV2::avx2) |
                            platform::feature_bit(CpuFeatureV2::fma);
  const auto permits = [](platform::CpuFeatureDomainV2 domain) {
    return (domain.known & ~platform::kKnownCpuFeatureBitsV2) == 0 &&
           (domain.available & ~domain.known) == 0 &&
           (domain.known & required) == required &&
           (domain.available & required) == required;
  };
  return capabilities.version == platform::kCpuCapabilitiesVersionV2 &&
         capabilities.architecture == platform::ArchitectureKindV1::x86_64 &&
         permits(capabilities.hardware) && permits(capabilities.os_enabled) &&
         capabilities.os_xstate_mask_known &&
         (capabilities.os_xstate_mask & UINT64_C(6)) == UINT64_C(6);
}

} // namespace matcore::mdslc::runtime::closed_host_v1::detail
#endif
