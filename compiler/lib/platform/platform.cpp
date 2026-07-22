#include "platform.h"

#include <sstream>

namespace matcore::mdslc::platform {

std::string_view to_string(PlatformKindV1 value) noexcept {
  switch (value) {
    case PlatformKindV1::unknown:
      return "unknown";
    case PlatformKindV1::linux:
      return "linux";
    case PlatformKindV1::windows:
      return "windows";
  }
  return "invalid";
}

std::string_view to_string(ArchitectureKindV1 value) noexcept {
  switch (value) {
    case ArchitectureKindV1::unknown:
      return "unknown";
    case ArchitectureKindV1::x86_64:
      return "x86_64";
    case ArchitectureKindV1::aarch64:
      return "aarch64";
  }
  return "invalid";
}

std::string_view to_string(CompilerFrontendV1 value) noexcept {
  switch (value) {
    case CompilerFrontendV1::unknown:
      return "unknown";
    case CompilerFrontendV1::clang_gnu:
      return "clang-gnu";
    case CompilerFrontendV1::clang_cl:
      return "clang-cl";
    case CompilerFrontendV1::gcc:
      return "gcc";
    case CompilerFrontendV1::msvc:
      return "msvc";
  }
  return "invalid";
}

std::string_view to_string(ObjectFormatV1 value) noexcept {
  switch (value) {
    case ObjectFormatV1::unknown:
      return "unknown";
    case ObjectFormatV1::elf:
      return "elf";
    case ObjectFormatV1::coff:
      return "coff";
  }
  return "invalid";
}

std::string_view to_string(ExecutableFormatV1 value) noexcept {
  switch (value) {
    case ExecutableFormatV1::unknown:
      return "unknown";
    case ExecutableFormatV1::elf:
      return "elf";
    case ExecutableFormatV1::portable_executable:
      return "pe";
  }
  return "invalid";
}

std::string_view to_string(RuntimeLibraryModelV1 value) noexcept {
  switch (value) {
    case RuntimeLibraryModelV1::unknown:
      return "unknown";
    case RuntimeLibraryModelV1::elf_shared_object:
      return "elf-shared-object";
    case RuntimeLibraryModelV1::windows_dll_import_library:
      return "windows-dll-import-library";
  }
  return "invalid";
}

std::string_view to_string(ProcessLaunchBackendV1 value) noexcept {
  switch (value) {
    case ProcessLaunchBackendV1::unavailable:
      return "unavailable";
    case ProcessLaunchBackendV1::posix_fork_exec:
      return "posix-fork-exec";
    case ProcessLaunchBackendV1::windows_create_process:
      return "windows-create-process";
  }
  return "invalid";
}

std::string_view to_string(DynamicLibraryModelV1 value) noexcept {
  switch (value) {
    case DynamicLibraryModelV1::unavailable:
      return "unavailable";
    case DynamicLibraryModelV1::posix_dlopen:
      return "posix-dlopen";
    case DynamicLibraryModelV1::windows_load_library:
      return "windows-load-library";
  }
  return "invalid";
}

std::string_view to_string(CapabilitySupportV1 value) noexcept {
  switch (value) {
    case CapabilitySupportV1::unavailable:
      return "unavailable";
    case CapabilitySupportV1::modeled:
      return "modeled";
    case CapabilitySupportV1::implemented:
      return "implemented";
  }
  return "invalid";
}

std::string_view to_string(RuntimeValidationV1 value) noexcept {
  switch (value) {
    case RuntimeValidationV1::not_applicable:
      return "not-applicable";
    case RuntimeValidationV1::not_validated:
      return "not-validated";
    case RuntimeValidationV1::runtime_validated:
      return "runtime-validated";
  }
  return "invalid";
}

namespace {

void append_capability(std::ostringstream &output, std::string_view name,
                       const CapabilityStatusV1 &status) {
  output << name << ':' << to_string(status.support) << '/'
         << (status.discovery_complete ? "complete" : "incomplete") << '/'
         << to_string(status.runtime_validation);
}

}  // namespace

std::string format_platform_record_v1(const PlatformRecordV1 &record) {
  std::ostringstream output;
  output << "platform-record-v1{version=" << record.version
         << ",platform=" << to_string(record.platform)
         << ",architecture=" << to_string(record.architecture)
         << ",compiler=" << to_string(record.compiler_frontend)
         << ",object=" << to_string(record.object_format)
         << ",executable=" << to_string(record.executable_format)
         << ",runtime=" << to_string(record.runtime_library_model)
         << ",process=" << to_string(record.process_launch_backend)
         << ",dynamic-library=" << to_string(record.dynamic_library_model)
         << ",discovery="
         << (record.discovery_complete ? "complete" : "incomplete")
         << ",capabilities=[";
  append_capability(output, "toolchain", record.toolchain);
  output << ',';
  append_capability(output, "artifact", record.artifact_model);
  output << ',';
  append_capability(output, "runtime", record.runtime_library);
  output << ',';
  append_capability(output, "process", record.process_launch);
  output << ',';
  append_capability(output, "dynamic-library", record.dynamic_library);
  output << "]}";
  return output.str();
}

}  // namespace matcore::mdslc::platform
