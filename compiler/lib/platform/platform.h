#ifndef MATCORE_MDSLC_PLATFORM_PLATFORM_H
#define MATCORE_MDSLC_PLATFORM_PLATFORM_H

#include <cstdint>
#include <string>
#include <string_view>

namespace matcore::mdslc::platform {

inline constexpr std::uint32_t kPlatformRecordVersionV1 = 1;

enum class PlatformKindV1 : std::uint8_t {
  unknown = 0,
  linux = 1,
  windows = 2,
};

enum class ArchitectureKindV1 : std::uint8_t {
  unknown = 0,
  x86_64 = 1,
  aarch64 = 2,
};

enum class CompilerFrontendV1 : std::uint8_t {
  unknown = 0,
  clang_gnu = 1,
  clang_cl = 2,
  gcc = 3,
  msvc = 4,
};

enum class ObjectFormatV1 : std::uint8_t {
  unknown = 0,
  elf = 1,
  coff = 2,
};

enum class ExecutableFormatV1 : std::uint8_t {
  unknown = 0,
  elf = 1,
  portable_executable = 2,
};

enum class RuntimeLibraryModelV1 : std::uint8_t {
  unknown = 0,
  elf_shared_object = 1,
  windows_dll_import_library = 2,
};

enum class ProcessLaunchBackendV1 : std::uint8_t {
  unavailable = 0,
  posix_fork_exec = 1,
  windows_create_process = 2,
};

enum class DynamicLibraryModelV1 : std::uint8_t {
  unavailable = 0,
  posix_dlopen = 1,
  windows_load_library = 2,
};

enum class CapabilitySupportV1 : std::uint8_t {
  unavailable = 0,
  modeled = 1,
  implemented = 2,
};

enum class RuntimeValidationV1 : std::uint8_t {
  not_applicable = 0,
  not_validated = 1,
  runtime_validated = 2,
};

struct CapabilityStatusV1 {
  CapabilitySupportV1 support = CapabilitySupportV1::unavailable;
  bool discovery_complete = false;
  RuntimeValidationV1 runtime_validation =
      RuntimeValidationV1::not_applicable;
};

struct PlatformRecordV1 {
  std::uint32_t version = kPlatformRecordVersionV1;
  PlatformKindV1 platform = PlatformKindV1::unknown;
  ArchitectureKindV1 architecture = ArchitectureKindV1::unknown;
  CompilerFrontendV1 compiler_frontend = CompilerFrontendV1::unknown;
  ObjectFormatV1 object_format = ObjectFormatV1::unknown;
  ExecutableFormatV1 executable_format = ExecutableFormatV1::unknown;
  RuntimeLibraryModelV1 runtime_library_model =
      RuntimeLibraryModelV1::unknown;
  ProcessLaunchBackendV1 process_launch_backend =
      ProcessLaunchBackendV1::unavailable;
  DynamicLibraryModelV1 dynamic_library_model =
      DynamicLibraryModelV1::unavailable;
  bool discovery_complete = false;
  CapabilityStatusV1 toolchain;
  CapabilityStatusV1 artifact_model;
  CapabilityStatusV1 runtime_library;
  CapabilityStatusV1 process_launch;
  CapabilityStatusV1 dynamic_library;
};

struct PlatformRecordValidationV1 {
  bool valid = false;
  std::string_view reason;

  constexpr explicit operator bool() const noexcept { return valid; }
};

constexpr bool known(PlatformKindV1 value) noexcept {
  switch (value) {
    case PlatformKindV1::unknown:
    case PlatformKindV1::linux:
    case PlatformKindV1::windows:
      return true;
  }
  return false;
}

constexpr bool known(ArchitectureKindV1 value) noexcept {
  switch (value) {
    case ArchitectureKindV1::unknown:
    case ArchitectureKindV1::x86_64:
    case ArchitectureKindV1::aarch64:
      return true;
  }
  return false;
}

constexpr bool known(CompilerFrontendV1 value) noexcept {
  switch (value) {
    case CompilerFrontendV1::unknown:
    case CompilerFrontendV1::clang_gnu:
    case CompilerFrontendV1::clang_cl:
    case CompilerFrontendV1::gcc:
    case CompilerFrontendV1::msvc:
      return true;
  }
  return false;
}

constexpr bool known(ObjectFormatV1 value) noexcept {
  switch (value) {
    case ObjectFormatV1::unknown:
    case ObjectFormatV1::elf:
    case ObjectFormatV1::coff:
      return true;
  }
  return false;
}

constexpr bool known(ExecutableFormatV1 value) noexcept {
  switch (value) {
    case ExecutableFormatV1::unknown:
    case ExecutableFormatV1::elf:
    case ExecutableFormatV1::portable_executable:
      return true;
  }
  return false;
}

constexpr bool known(RuntimeLibraryModelV1 value) noexcept {
  switch (value) {
    case RuntimeLibraryModelV1::unknown:
    case RuntimeLibraryModelV1::elf_shared_object:
    case RuntimeLibraryModelV1::windows_dll_import_library:
      return true;
  }
  return false;
}

constexpr bool known(ProcessLaunchBackendV1 value) noexcept {
  switch (value) {
    case ProcessLaunchBackendV1::unavailable:
    case ProcessLaunchBackendV1::posix_fork_exec:
    case ProcessLaunchBackendV1::windows_create_process:
      return true;
  }
  return false;
}

constexpr bool known(DynamicLibraryModelV1 value) noexcept {
  switch (value) {
    case DynamicLibraryModelV1::unavailable:
    case DynamicLibraryModelV1::posix_dlopen:
    case DynamicLibraryModelV1::windows_load_library:
      return true;
  }
  return false;
}

constexpr bool known(CapabilitySupportV1 value) noexcept {
  switch (value) {
    case CapabilitySupportV1::unavailable:
    case CapabilitySupportV1::modeled:
    case CapabilitySupportV1::implemented:
      return true;
  }
  return false;
}

constexpr bool known(RuntimeValidationV1 value) noexcept {
  switch (value) {
    case RuntimeValidationV1::not_applicable:
    case RuntimeValidationV1::not_validated:
    case RuntimeValidationV1::runtime_validated:
      return true;
  }
  return false;
}

constexpr PlatformRecordValidationV1
validate_capability_status_v1(const CapabilityStatusV1 &status) noexcept {
  if (!known(status.support) || !known(status.runtime_validation)) {
    return {false, "capability status contains an unknown enum value"};
  }
  if (status.support == CapabilitySupportV1::unavailable) {
    if (status.runtime_validation != RuntimeValidationV1::not_applicable) {
      return {false,
              "unavailable capability must use not-applicable validation"};
    }
    return {true, {}};
  }
  if (status.runtime_validation == RuntimeValidationV1::not_applicable) {
    return {false,
            "modeled or implemented capability requires a validation state"};
  }
  if (status.runtime_validation == RuntimeValidationV1::runtime_validated &&
      (status.support != CapabilitySupportV1::implemented ||
       !status.discovery_complete)) {
    return {false,
            "runtime validation requires implemented, complete capability"};
  }
  return {true, {}};
}

constexpr PlatformRecordValidationV1
validate_platform_record_v1(const PlatformRecordV1 &record) noexcept {
  if (record.version != kPlatformRecordVersionV1) {
    return {false, "platform record version is unsupported"};
  }
  if (!known(record.platform) || !known(record.architecture) ||
      !known(record.compiler_frontend) || !known(record.object_format) ||
      !known(record.executable_format) ||
      !known(record.runtime_library_model) ||
      !known(record.process_launch_backend) ||
      !known(record.dynamic_library_model)) {
    return {false, "platform record contains an unknown enum value"};
  }
  for (const CapabilityStatusV1 status :
       {record.toolchain, record.artifact_model, record.runtime_library,
        record.process_launch, record.dynamic_library}) {
    const PlatformRecordValidationV1 validation =
        validate_capability_status_v1(status);
    if (!validation) return validation;
  }
  if (record.discovery_complete &&
      (record.platform == PlatformKindV1::unknown ||
       record.architecture == ArchitectureKindV1::unknown ||
       record.compiler_frontend == CompilerFrontendV1::unknown ||
       record.object_format == ObjectFormatV1::unknown ||
       record.executable_format == ExecutableFormatV1::unknown ||
       record.runtime_library_model == RuntimeLibraryModelV1::unknown ||
       record.process_launch_backend == ProcessLaunchBackendV1::unavailable ||
       record.dynamic_library_model == DynamicLibraryModelV1::unavailable)) {
    return {false, "complete discovery requires every modeled platform field"};
  }
  if (record.platform == PlatformKindV1::unknown) {
    if (record.object_format != ObjectFormatV1::unknown ||
        record.executable_format != ExecutableFormatV1::unknown ||
        record.runtime_library_model != RuntimeLibraryModelV1::unknown ||
        record.process_launch_backend != ProcessLaunchBackendV1::unavailable ||
        record.dynamic_library_model != DynamicLibraryModelV1::unavailable) {
      return {false, "unknown platform cannot claim a concrete platform model"};
    }
  } else if (record.platform == PlatformKindV1::linux) {
    if (record.object_format != ObjectFormatV1::elf ||
        record.executable_format != ExecutableFormatV1::elf ||
        record.runtime_library_model !=
            RuntimeLibraryModelV1::elf_shared_object ||
        record.process_launch_backend !=
            ProcessLaunchBackendV1::posix_fork_exec ||
        record.dynamic_library_model !=
            DynamicLibraryModelV1::posix_dlopen) {
      return {false, "Linux record requires the ELF and POSIX platform model"};
    }
  } else if (record.platform == PlatformKindV1::windows) {
    if (record.object_format != ObjectFormatV1::coff ||
        record.executable_format !=
            ExecutableFormatV1::portable_executable ||
        record.runtime_library_model !=
            RuntimeLibraryModelV1::windows_dll_import_library ||
        record.process_launch_backend !=
            ProcessLaunchBackendV1::windows_create_process ||
        record.dynamic_library_model !=
            DynamicLibraryModelV1::windows_load_library) {
      return {false, "Windows record requires the COFF and PE platform model"};
    }
  }
  if ((record.compiler_frontend == CompilerFrontendV1::clang_cl ||
       record.compiler_frontend == CompilerFrontendV1::msvc) &&
      record.platform != PlatformKindV1::windows) {
    return {false, "MSVC-style compiler frontend requires Windows"};
  }
  if (record.compiler_frontend == CompilerFrontendV1::gcc &&
      record.platform == PlatformKindV1::windows) {
    return {false, "GCC frontend is outside the Windows MSVC ABI model"};
  }
  if (record.toolchain.support == CapabilitySupportV1::implemented &&
      record.compiler_frontend == CompilerFrontendV1::unknown) {
    return {false, "implemented toolchain requires a known compiler frontend"};
  }
  if (record.artifact_model.support == CapabilitySupportV1::implemented &&
      (record.object_format == ObjectFormatV1::unknown ||
       record.executable_format == ExecutableFormatV1::unknown)) {
    return {false, "implemented artifact model requires known formats"};
  }
  if (record.runtime_library.support == CapabilitySupportV1::implemented &&
      record.runtime_library_model == RuntimeLibraryModelV1::unknown) {
    return {false, "implemented runtime library requires a known model"};
  }
  if (record.process_launch.support == CapabilitySupportV1::implemented &&
      record.process_launch_backend == ProcessLaunchBackendV1::unavailable) {
    return {false, "implemented process launch requires a backend"};
  }
  if (record.dynamic_library.support == CapabilitySupportV1::implemented &&
      record.dynamic_library_model == DynamicLibraryModelV1::unavailable) {
    return {false, "implemented dynamic loading requires a model"};
  }
  return {true, {}};
}

constexpr bool compiler_pipeline_supported_v1(
    const PlatformRecordV1 &record) noexcept {
  return validate_platform_record_v1(record).valid &&
         record.toolchain.support == CapabilitySupportV1::implemented &&
         record.artifact_model.support == CapabilitySupportV1::implemented &&
         record.runtime_library.support == CapabilitySupportV1::implemented &&
         record.process_launch.support == CapabilitySupportV1::implemented;
}

constexpr bool compiler_pipeline_runtime_validated_v1(
    const PlatformRecordV1 &record) noexcept {
  return compiler_pipeline_supported_v1(record) &&
         record.toolchain.runtime_validation ==
             RuntimeValidationV1::runtime_validated &&
         record.artifact_model.runtime_validation ==
             RuntimeValidationV1::runtime_validated &&
         record.runtime_library.runtime_validation ==
             RuntimeValidationV1::runtime_validated &&
         record.process_launch.runtime_validation ==
             RuntimeValidationV1::runtime_validated;
}

constexpr PlatformRecordV1 discover_compile_platform_v1() noexcept {
  PlatformRecordV1 record;

#if defined(_WIN32)
  record.platform = PlatformKindV1::windows;
  record.object_format = ObjectFormatV1::coff;
  record.executable_format = ExecutableFormatV1::portable_executable;
  record.runtime_library_model =
      RuntimeLibraryModelV1::windows_dll_import_library;
  record.process_launch_backend =
      ProcessLaunchBackendV1::windows_create_process;
  record.dynamic_library_model =
      DynamicLibraryModelV1::windows_load_library;
#elif defined(__linux__)
  record.platform = PlatformKindV1::linux;
  record.object_format = ObjectFormatV1::elf;
  record.executable_format = ExecutableFormatV1::elf;
  record.runtime_library_model = RuntimeLibraryModelV1::elf_shared_object;
  record.process_launch_backend = ProcessLaunchBackendV1::posix_fork_exec;
  record.dynamic_library_model = DynamicLibraryModelV1::posix_dlopen;
#endif

#if defined(__x86_64__) || defined(_M_X64)
  record.architecture = ArchitectureKindV1::x86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
  record.architecture = ArchitectureKindV1::aarch64;
#endif

#if defined(__clang__) && defined(_MSC_VER)
  record.compiler_frontend = CompilerFrontendV1::clang_cl;
#elif defined(__clang__)
  record.compiler_frontend = CompilerFrontendV1::clang_gnu;
#elif defined(_MSC_VER)
  record.compiler_frontend = CompilerFrontendV1::msvc;
#elif defined(__GNUC__)
  record.compiler_frontend = CompilerFrontendV1::gcc;
#endif

  record.discovery_complete =
      record.platform != PlatformKindV1::unknown &&
      record.architecture != ArchitectureKindV1::unknown &&
      record.compiler_frontend != CompilerFrontendV1::unknown &&
      record.object_format != ObjectFormatV1::unknown &&
      record.executable_format != ExecutableFormatV1::unknown &&
      record.runtime_library_model != RuntimeLibraryModelV1::unknown &&
      record.process_launch_backend != ProcessLaunchBackendV1::unavailable &&
      record.dynamic_library_model != DynamicLibraryModelV1::unavailable;

  if (record.platform == PlatformKindV1::linux) {
    const CapabilityStatusV1 implemented{
        CapabilitySupportV1::implemented, record.discovery_complete,
        RuntimeValidationV1::not_validated};
    record.toolchain = implemented;
    record.artifact_model = implemented;
    record.runtime_library = implemented;
    record.process_launch = implemented;
    record.dynamic_library = {CapabilitySupportV1::modeled,
                              record.discovery_complete,
                              RuntimeValidationV1::not_validated};
  } else if (record.platform == PlatformKindV1::windows) {
    const CapabilityStatusV1 modeled{
        CapabilitySupportV1::modeled, record.discovery_complete,
        RuntimeValidationV1::not_validated};
    record.toolchain = modeled;
    record.artifact_model = modeled;
    record.runtime_library = modeled;
    record.process_launch = modeled;
    record.dynamic_library = modeled;
  }
  return record;
}

std::string_view to_string(PlatformKindV1 value) noexcept;
std::string_view to_string(ArchitectureKindV1 value) noexcept;
std::string_view to_string(CompilerFrontendV1 value) noexcept;
std::string_view to_string(ObjectFormatV1 value) noexcept;
std::string_view to_string(ExecutableFormatV1 value) noexcept;
std::string_view to_string(RuntimeLibraryModelV1 value) noexcept;
std::string_view to_string(ProcessLaunchBackendV1 value) noexcept;
std::string_view to_string(DynamicLibraryModelV1 value) noexcept;
std::string_view to_string(CapabilitySupportV1 value) noexcept;
std::string_view to_string(RuntimeValidationV1 value) noexcept;

std::string format_platform_record_v1(const PlatformRecordV1 &record);

}  // namespace matcore::mdslc::platform

#endif
