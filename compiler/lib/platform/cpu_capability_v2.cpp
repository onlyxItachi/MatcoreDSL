#include "cpu_capability_v2.h"

#include <sstream>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#elif defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
#include <cpuid.h>
#endif

#if defined(__linux__) && defined(__x86_64__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace matcore::mdslc::platform {
namespace {

inline constexpr std::uint64_t kXcr0Xmm = UINT64_C(1) << 1;
inline constexpr std::uint64_t kXcr0Ymm = UINT64_C(1) << 2;
inline constexpr std::uint64_t kXcr0Opmask = UINT64_C(1) << 5;
inline constexpr std::uint64_t kXcr0ZmmHi256 = UINT64_C(1) << 6;
inline constexpr std::uint64_t kXcr0Hi16Zmm = UINT64_C(1) << 7;
inline constexpr std::uint64_t kXcr0TileConfig = UINT64_C(1) << 17;
inline constexpr std::uint64_t kXcr0TileData = UINT64_C(1) << 18;
inline constexpr std::uint64_t kAvxXstate = kXcr0Xmm | kXcr0Ymm;
inline constexpr std::uint64_t kAvx512Xstate =
    kAvxXstate | kXcr0Opmask | kXcr0ZmmHi256 | kXcr0Hi16Zmm;
inline constexpr std::uint64_t kAmxXstate =
    kXcr0TileConfig | kXcr0TileData;

constexpr bool domain_well_formed(const CpuFeatureDomainV2 &domain) noexcept {
  return (domain.known & ~kKnownCpuFeatureBitsV2) == 0 &&
         (domain.available & ~domain.known) == 0;
}

constexpr bool dependencies_satisfied(
    const CpuFeatureDomainV2 &domain) noexcept {
  const auto available = [&domain](CpuFeatureV2 feature) {
    return feature_available(domain, feature);
  };
  if ((available(CpuFeatureV2::avx512dq) ||
       available(CpuFeatureV2::avx512bw) ||
       available(CpuFeatureV2::avx512vl) ||
       available(CpuFeatureV2::avx512vnni) ||
       available(CpuFeatureV2::avx512bf16)) &&
      !available(CpuFeatureV2::avx512f)) {
    return false;
  }
  if ((available(CpuFeatureV2::amx_bf16) ||
       available(CpuFeatureV2::amx_int8)) &&
      !available(CpuFeatureV2::amx_tile)) {
    return false;
  }
  return true;
}

std::string_view domain_state(const CpuFeatureDomainV2 &domain,
                              CpuFeatureV2 feature) noexcept {
  if (!feature_known(domain, feature)) return "unknown";
  return feature_available(domain, feature) ? "yes" : "no";
}

void mark_known(CpuFeatureDomainV2 &domain, std::uint64_t bits) noexcept {
  domain.known |= bits;
}

void mark_available(CpuFeatureDomainV2 &domain,
                    CpuFeatureV2 feature) noexcept {
  domain.known |= feature_bit(feature);
  domain.available |= feature_bit(feature);
}

struct CpuidResult {
  std::uint32_t eax = 0;
  std::uint32_t ebx = 0;
  std::uint32_t ecx = 0;
  std::uint32_t edx = 0;
};

bool read_cpuid(std::uint32_t leaf, std::uint32_t subleaf,
                CpuidResult *output) noexcept {
  if (output == nullptr) return false;
#if defined(_MSC_VER) && defined(_M_X64)
  int maximum[4]{};
  __cpuidex(maximum, 0, 0);
  if (leaf > static_cast<std::uint32_t>(maximum[0])) return false;
  int registers[4]{};
  __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
  *output = {static_cast<std::uint32_t>(registers[0]),
             static_cast<std::uint32_t>(registers[1]),
             static_cast<std::uint32_t>(registers[2]),
             static_cast<std::uint32_t>(registers[3])};
  return true;
#elif defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
  const unsigned int maximum = __get_cpuid_max(0, nullptr);
  if (leaf > maximum) return false;
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  *output = {eax, ebx, ecx, edx};
  return true;
#else
  (void)leaf;
  (void)subleaf;
  return false;
#endif
}

bool read_xcr0(std::uint64_t *output) noexcept {
  if (output == nullptr) return false;
#if defined(_MSC_VER) && defined(_M_X64)
  *output = _xgetbv(0);
  return true;
#elif defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
  std::uint32_t eax = 0;
  std::uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  *output = (static_cast<std::uint64_t>(edx) << 32U) | eax;
  return true;
#else
  return false;
#endif
}

bool read_amx_permission(bool *granted) noexcept {
  if (granted == nullptr) return false;
#if defined(__linux__) && defined(__x86_64__) && \
    defined(SYS_arch_prctl) && defined(ARCH_GET_XCOMP_PERM)
  unsigned long permissions = 0;
  if (::syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &permissions) != 0) {
    return false;
  }
  *granted = (permissions & (1UL << 18U)) != 0;
  return true;
#else
  return false;
#endif
}

CompilerFrontendV1 compile_frontend() noexcept {
#if defined(__clang__) && defined(_MSC_VER)
  return CompilerFrontendV1::clang_cl;
#elif defined(__clang__)
  return CompilerFrontendV1::clang_gnu;
#elif defined(_MSC_VER)
  return CompilerFrontendV1::msvc;
#elif defined(__GNUC__)
  return CompilerFrontendV1::gcc;
#else
  return CompilerFrontendV1::unknown;
#endif
}

std::uint32_t compile_frontend_major_version() noexcept {
#if defined(__clang__)
  return __clang_major__;
#elif defined(__GNUC__) && !defined(_MSC_VER)
  return __GNUC__;
#elif defined(_MSC_VER)
  return _MSC_VER;
#else
  return 0;
#endif
}

std::uint16_t derive_usable_vector_bits(
    const CpuCapabilitiesV2 &record) noexcept {
  if (has_usable_feature_v2(record, CpuFeatureV2::avx512f)) return 512;
  if (has_usable_feature_v2(record, CpuFeatureV2::avx2)) return 256;
  return 0;
}

}  // namespace

CpuCapabilitiesValidationV2 validate_cpu_capabilities_v2(
    const CpuCapabilitiesV2 &record) noexcept {
  if (record.version != kCpuCapabilitiesVersionV2) {
    return {false, "CPU capability record version is unsupported"};
  }
  if (!known(record.architecture)) {
    return {false, "CPU capability record architecture is invalid"};
  }
  for (const CpuFeatureDomainV2 domain :
       {record.hardware, record.os_enabled, record.compiler,
        record.implementation, record.runtime_validation}) {
    if (!domain_well_formed(domain)) {
      return {false, "CPU feature domain contains unknown or unproven bits"};
    }
    if (!dependencies_satisfied(domain)) {
      return {false, "CPU feature domain violates ISA feature dependencies"};
    }
  }

  if (record.architecture == ArchitectureKindV1::unknown) {
    if (record.hardware.known != 0 || record.os_enabled.known != 0 ||
        record.compiler.known != 0 || record.implementation.known != 0 ||
        record.runtime_validation.known != 0 ||
        record.os_xstate_mask_known || record.amx_permission_known ||
        record.usable_vector_bits != 0) {
      return {false, "unknown CPU architecture cannot claim feature state"};
    }
    return {true, {}};
  }

  if ((record.hardware.available &
       feature_bit(CpuFeatureV2::portable_scalar_f32)) == 0 ||
      (record.os_enabled.available &
       feature_bit(CpuFeatureV2::portable_scalar_f32)) == 0 ||
      (record.compiler.available &
       feature_bit(CpuFeatureV2::portable_scalar_f32)) == 0) {
    return {false, "known CPU requires portable scalar hardware, OS, and compiler support"};
  }
  if ((record.os_enabled.available & ~record.hardware.available) != 0) {
    return {false, "OS-enabled feature cannot exceed hardware support"};
  }
  if ((record.runtime_validation.available &
       ~(record.hardware.available & record.os_enabled.available &
         record.compiler.available & record.implementation.available)) != 0) {
    return {false, "runtime validation requires a usable implementation"};
  }

  if (record.architecture != ArchitectureKindV1::x86_64) {
    const std::uint64_t available_x86 =
        (record.hardware.available | record.os_enabled.available |
         record.compiler.available | record.implementation.available |
         record.runtime_validation.available) &
        kX86CpuFeatureBitsV2;
    if (available_x86 != 0) {
      return {false, "x86 ISA feature requires x86_64 architecture"};
    }
    if (record.os_xstate_mask_known || record.amx_permission_known) {
      return {false, "non-x86 record cannot claim x86 OS state"};
    }
  }

  if (record.amx_permission_granted && !record.amx_permission_known) {
    return {false, "AMX permission cannot be granted when its state is unknown"};
  }
  if ((feature_available(record.os_enabled, CpuFeatureV2::amx_tile) ||
       feature_available(record.os_enabled, CpuFeatureV2::amx_bf16) ||
       feature_available(record.os_enabled, CpuFeatureV2::amx_int8)) &&
      (!record.amx_permission_known || !record.amx_permission_granted)) {
    return {false, "OS-enabled AMX requires explicit architectural-state permission"};
  }

  if (record.os_xstate_mask_known) {
    if ((feature_available(record.os_enabled, CpuFeatureV2::avx2) ||
         feature_available(record.os_enabled, CpuFeatureV2::fma)) &&
        (record.os_xstate_mask & kAvxXstate) != kAvxXstate) {
      return {false, "OS-enabled AVX2/FMA requires XMM and YMM XSTATE"};
    }
    if (feature_available(record.os_enabled, CpuFeatureV2::avx512f) &&
        (record.os_xstate_mask & kAvx512Xstate) != kAvx512Xstate) {
      return {false, "OS-enabled AVX-512 requires complete ZMM XSTATE"};
    }
    if (feature_available(record.os_enabled, CpuFeatureV2::amx_tile) &&
        (record.os_xstate_mask & kAmxXstate) != kAmxXstate) {
      return {false, "OS-enabled AMX requires tile configuration and data XSTATE"};
    }
  }

  if (record.usable_vector_bits != derive_usable_vector_bits(record)) {
    return {false, "usable vector width disagrees with usable feature domains"};
  }
  return {true, {}};
}

CpuImplementationAvailabilityV2
unknown_cpu_implementation_availability_v2() noexcept {
  return {};
}

CpuFeatureDomainV2 cpu_compiler_feature_domain_v2(
    ArchitectureKindV1 architecture, CompilerFrontendV1 frontend,
    std::uint32_t compiler_major_version) noexcept {
  CpuFeatureDomainV2 result;
  if (!known(architecture) || !known(frontend) ||
      architecture == ArchitectureKindV1::unknown ||
      frontend == CompilerFrontendV1::unknown) {
    return result;
  }

  mark_available(result, CpuFeatureV2::portable_scalar_f32);
  if (architecture != ArchitectureKindV1::x86_64) return result;

  // The ISA-specific compiler vocabulary is known only for compiler families
  // whose isolated-function strategy is part of this build.  clang-cl is
  // intentionally normalized with Clang rather than falling through the
  // `_MSC_VER` ABI compatibility macro.
  switch (frontend) {
    case CompilerFrontendV1::clang_gnu:
    case CompilerFrontendV1::clang_cl:
      mark_known(result, kX86CpuFeatureBitsV2);
      if (compiler_major_version >= 14) {
        result.available |= kX86CpuFeatureBitsV2;
      } else if (compiler_major_version != 0) {
        result.available |= feature_bit(CpuFeatureV2::avx2) |
                            feature_bit(CpuFeatureV2::fma) |
                            feature_bit(CpuFeatureV2::avx512f) |
                            feature_bit(CpuFeatureV2::avx512dq) |
                            feature_bit(CpuFeatureV2::avx512bw) |
                            feature_bit(CpuFeatureV2::avx512vl) |
                            feature_bit(CpuFeatureV2::avx512vnni);
      }
      break;
    case CompilerFrontendV1::gcc:
      mark_known(result, kX86CpuFeatureBitsV2);
      if (compiler_major_version >= 12) {
        result.available |= kX86CpuFeatureBitsV2;
      } else if (compiler_major_version >= 8) {
        result.available |= feature_bit(CpuFeatureV2::avx2) |
                            feature_bit(CpuFeatureV2::fma) |
                            feature_bit(CpuFeatureV2::avx512f) |
                            feature_bit(CpuFeatureV2::avx512dq) |
                            feature_bit(CpuFeatureV2::avx512bw) |
                            feature_bit(CpuFeatureV2::avx512vl) |
                            feature_bit(CpuFeatureV2::avx512vnni);
      }
      break;
    case CompilerFrontendV1::msvc:
      // The MSVC ABI is supported through clang-cl.  Native MSVC has no
      // validated isolated ISA implementation in this milestone.
      mark_known(result, kX86CpuFeatureBitsV2);
      break;
    case CompilerFrontendV1::unknown:
      return {};
  }
  return result;
}

CpuCapabilitiesV2 discover_cpu_capabilities_v2(
    const CpuImplementationAvailabilityV2 &implementation) noexcept {
  CpuCapabilitiesV2 result;
#if defined(__x86_64__) || defined(_M_X64)
  result.architecture = ArchitectureKindV1::x86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
  result.architecture = ArchitectureKindV1::aarch64;
#endif

  if (result.architecture == ArchitectureKindV1::unknown) return result;

  mark_available(result.hardware, CpuFeatureV2::portable_scalar_f32);
  mark_available(result.os_enabled, CpuFeatureV2::portable_scalar_f32);
  result.compiler = cpu_compiler_feature_domain_v2(
      result.architecture, compile_frontend(),
      compile_frontend_major_version());

  if (implementation.version == kCpuImplementationAvailabilityVersionV2 &&
      domain_well_formed(implementation.compiled) &&
      domain_well_formed(implementation.runtime_validated)) {
    result.implementation = implementation.compiled;
    result.runtime_validation = implementation.runtime_validated;
  }

  if (result.architecture != ArchitectureKindV1::x86_64) {
    result.usable_vector_bits = derive_usable_vector_bits(result);
    return result;
  }

  CpuidResult leaf1;
  if (!read_cpuid(1, 0, &leaf1)) {
    result.usable_vector_bits = derive_usable_vector_bits(result);
    return result;
  }

  mark_known(result.hardware, kX86CpuFeatureBitsV2);
  const bool hardware_avx = (leaf1.ecx & (UINT32_C(1) << 28U)) != 0;
  const bool hardware_fma = (leaf1.ecx & (UINT32_C(1) << 12U)) != 0;
  const bool osxsave = (leaf1.ecx & (UINT32_C(1) << 27U)) != 0;
  if (hardware_avx && hardware_fma) {
    mark_available(result.hardware, CpuFeatureV2::fma);
  }

  CpuidResult leaf7_0;
  bool leaf7_available = read_cpuid(7, 0, &leaf7_0);
  std::uint32_t maximum_leaf7_subleaf = 0;
  if (leaf7_available) {
    maximum_leaf7_subleaf = leaf7_0.eax;
    if (hardware_avx && (leaf7_0.ebx & (UINT32_C(1) << 5U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::avx2);
    }
    if ((leaf7_0.ebx & (UINT32_C(1) << 16U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::avx512f);
    }
    if ((leaf7_0.ebx & (UINT32_C(1) << 17U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::avx512dq);
    }
    if ((leaf7_0.ebx & (UINT32_C(1) << 30U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::avx512bw);
    }
    if ((leaf7_0.ebx & (UINT32_C(1) << 31U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::avx512vl);
    }
    if ((leaf7_0.ecx & (UINT32_C(1) << 11U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::avx512vnni);
    }
    if ((leaf7_0.edx & (UINT32_C(1) << 24U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::amx_tile);
    }
    if ((leaf7_0.edx & (UINT32_C(1) << 22U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::amx_bf16);
    }
    if ((leaf7_0.edx & (UINT32_C(1) << 25U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::amx_int8);
    }
  }
  if (maximum_leaf7_subleaf >= 1) {
    CpuidResult leaf7_1;
    if (read_cpuid(7, 1, &leaf7_1) &&
        (leaf7_1.eax & (UINT32_C(1) << 5U)) != 0) {
      mark_available(result.hardware, CpuFeatureV2::avx512bf16);
    }
  }

  mark_known(result.os_enabled, kX86CpuFeatureBitsV2);
  std::uint64_t xcr0 = 0;
  if (osxsave && read_xcr0(&xcr0)) {
    result.os_xstate_mask = xcr0;
    result.os_xstate_mask_known = true;
    const bool avx_state = (xcr0 & kAvxXstate) == kAvxXstate;
    const bool avx512_state = (xcr0 & kAvx512Xstate) == kAvx512Xstate;
    if (avx_state &&
        feature_available(result.hardware, CpuFeatureV2::avx2)) {
      mark_available(result.os_enabled, CpuFeatureV2::avx2);
    }
    if (avx_state && feature_available(result.hardware, CpuFeatureV2::fma)) {
      mark_available(result.os_enabled, CpuFeatureV2::fma);
    }
    for (const CpuFeatureV2 feature :
         {CpuFeatureV2::avx512f, CpuFeatureV2::avx512dq,
          CpuFeatureV2::avx512bw, CpuFeatureV2::avx512vl,
          CpuFeatureV2::avx512vnni, CpuFeatureV2::avx512bf16}) {
      if (avx512_state && feature_available(result.hardware, feature)) {
        mark_available(result.os_enabled, feature);
      }
    }

    const bool any_amx_hardware =
        feature_available(result.hardware, CpuFeatureV2::amx_tile) ||
        feature_available(result.hardware, CpuFeatureV2::amx_bf16) ||
        feature_available(result.hardware, CpuFeatureV2::amx_int8);
    if (any_amx_hardware) {
      bool permission = false;
      result.amx_permission_known = read_amx_permission(&permission);
      result.amx_permission_granted =
          result.amx_permission_known && permission;
      if (!result.amx_permission_known) {
        const std::uint64_t amx_bits =
            feature_bit(CpuFeatureV2::amx_tile) |
            feature_bit(CpuFeatureV2::amx_bf16) |
            feature_bit(CpuFeatureV2::amx_int8);
        result.os_enabled.known &= ~amx_bits;
        result.os_enabled.available &= ~amx_bits;
      } else if (permission && (xcr0 & kAmxXstate) == kAmxXstate) {
        for (const CpuFeatureV2 feature :
             {CpuFeatureV2::amx_tile, CpuFeatureV2::amx_bf16,
              CpuFeatureV2::amx_int8}) {
          if (feature_available(result.hardware, feature)) {
            mark_available(result.os_enabled, feature);
          }
        }
      }
    } else {
      result.amx_permission_known = true;
      result.amx_permission_granted = false;
    }
  }

  result.usable_vector_bits = derive_usable_vector_bits(result);
  return result;
}

std::string_view to_string(CpuFeatureV2 feature) noexcept {
  switch (feature) {
    case CpuFeatureV2::portable_scalar_f32:
      return "portable-scalar-f32";
    case CpuFeatureV2::avx2:
      return "avx2";
    case CpuFeatureV2::fma:
      return "fma";
    case CpuFeatureV2::avx512f:
      return "avx512f";
    case CpuFeatureV2::avx512dq:
      return "avx512dq";
    case CpuFeatureV2::avx512bw:
      return "avx512bw";
    case CpuFeatureV2::avx512vl:
      return "avx512vl";
    case CpuFeatureV2::avx512vnni:
      return "avx512vnni";
    case CpuFeatureV2::avx512bf16:
      return "avx512bf16";
    case CpuFeatureV2::amx_tile:
      return "amx-tile";
    case CpuFeatureV2::amx_bf16:
      return "amx-bf16";
    case CpuFeatureV2::amx_int8:
      return "amx-int8";
  }
  return "invalid";
}

std::string format_cpu_capabilities_v2(const CpuCapabilitiesV2 &record) {
  std::ostringstream output;
  output << "cpu-capabilities-v2{version=" << record.version
         << ",architecture=" << to_string(record.architecture)
         << ",usable-vector-bits=" << record.usable_vector_bits
         << ",xstate=";
  if (record.os_xstate_mask_known) {
    output << "0x" << std::hex << record.os_xstate_mask << std::dec;
  } else {
    output << "unknown";
  }
  output << ",amx-permission=";
  if (!record.amx_permission_known) {
    output << "unknown";
  } else {
    output << (record.amx_permission_granted ? "granted" : "not-granted");
  }
  output << ",features=[";
  bool first = true;
  for (const CpuFeatureV2 feature : kCpuFeatureOrderV2) {
    if (!first) output << ',';
    first = false;
    output << to_string(feature) << ":hardware="
           << domain_state(record.hardware, feature)
           << "/os=" << domain_state(record.os_enabled, feature)
           << "/compiler=" << domain_state(record.compiler, feature)
           << "/implementation="
           << domain_state(record.implementation, feature)
           << "/runtime="
           << domain_state(record.runtime_validation, feature);
  }
  output << "]}";
  return output.str();
}

}  // namespace matcore::mdslc::platform
