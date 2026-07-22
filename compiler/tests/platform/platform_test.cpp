#include "platform.h"

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

constexpr platform::CapabilityStatusV1 modeled_status() {
  return {platform::CapabilitySupportV1::modeled, true,
          platform::RuntimeValidationV1::not_validated};
}

constexpr platform::CapabilityStatusV1 implemented_status() {
  return {platform::CapabilitySupportV1::implemented, true,
          platform::RuntimeValidationV1::not_validated};
}

constexpr platform::PlatformRecordV1 synthetic_windows_record() {
  platform::PlatformRecordV1 record;
  record.platform = platform::PlatformKindV1::windows;
  record.architecture = platform::ArchitectureKindV1::x86_64;
  record.compiler_frontend = platform::CompilerFrontendV1::clang_cl;
  record.object_format = platform::ObjectFormatV1::coff;
  record.executable_format =
      platform::ExecutableFormatV1::portable_executable;
  record.runtime_library_model =
      platform::RuntimeLibraryModelV1::windows_dll_import_library;
  record.process_launch_backend =
      platform::ProcessLaunchBackendV1::windows_create_process;
  record.dynamic_library_model =
      platform::DynamicLibraryModelV1::windows_load_library;
  record.discovery_complete = true;
  record.toolchain = modeled_status();
  record.artifact_model = modeled_status();
  record.runtime_library = modeled_status();
  record.process_launch = modeled_status();
  record.dynamic_library = modeled_status();
  return record;
}

constexpr platform::PlatformRecordV1 synthetic_linux_record() {
  platform::PlatformRecordV1 record;
  record.platform = platform::PlatformKindV1::linux;
  record.architecture = platform::ArchitectureKindV1::aarch64;
  record.compiler_frontend = platform::CompilerFrontendV1::clang_gnu;
  record.object_format = platform::ObjectFormatV1::elf;
  record.executable_format = platform::ExecutableFormatV1::elf;
  record.runtime_library_model =
      platform::RuntimeLibraryModelV1::elf_shared_object;
  record.process_launch_backend =
      platform::ProcessLaunchBackendV1::posix_fork_exec;
  record.dynamic_library_model =
      platform::DynamicLibraryModelV1::posix_dlopen;
  record.discovery_complete = true;
  record.toolchain = implemented_status();
  record.artifact_model = implemented_status();
  record.runtime_library = implemented_status();
  record.process_launch = implemented_status();
  record.dynamic_library = modeled_status();
  return record;
}

constexpr auto kCompileRecord = platform::discover_compile_platform_v1();
static_assert(platform::validate_platform_record_v1(kCompileRecord).valid);
static_assert(
    platform::validate_platform_record_v1(synthetic_windows_record()).valid);
static_assert(
    platform::validate_platform_record_v1(synthetic_linux_record()).valid);
static_assert(
    !platform::compiler_pipeline_supported_v1(synthetic_windows_record()));
static_assert(platform::compiler_pipeline_supported_v1(
    synthetic_linux_record()));

}  // namespace

int main() {
  const platform::PlatformRecordV1 current =
      platform::discover_compile_platform_v1();
  const platform::PlatformRecordValidationV1 current_validation =
      platform::validate_platform_record_v1(current);
  expect(current_validation.valid, "compile-platform record validates");

#if defined(__linux__)
  expect(current.platform == platform::PlatformKindV1::linux,
         "Linux compilation reports Linux");
  expect(current.object_format == platform::ObjectFormatV1::elf,
         "Linux compilation reports ELF objects");
  expect(current.executable_format == platform::ExecutableFormatV1::elf,
         "Linux compilation reports ELF executables");
  expect(current.process_launch_backend ==
             platform::ProcessLaunchBackendV1::posix_fork_exec,
         "Linux compilation reports the current POSIX process backend");
  expect(platform::compiler_pipeline_supported_v1(current),
         "current Linux compiler pipeline is marked implemented");
#endif
#if defined(__clang__) && !defined(_MSC_VER)
  expect(current.compiler_frontend ==
             platform::CompilerFrontendV1::clang_gnu,
         "GNU-mode Clang is distinguished from clang-cl");
#endif
  expect(!platform::compiler_pipeline_runtime_validated_v1(current),
         "compile-time discovery does not invent runtime validation");

  const platform::PlatformRecordV1 windows = synthetic_windows_record();
  expect(platform::validate_platform_record_v1(windows).valid,
         "synthetic Windows MSVC-ABI model validates");
  expect(!platform::compiler_pipeline_supported_v1(windows),
         "modeled Windows record is not advertised as implemented");
  expect(!platform::compiler_pipeline_runtime_validated_v1(windows),
         "synthetic Windows record is explicitly not runtime-validated");

  platform::PlatformRecordV1 false_windows_validation = windows;
  false_windows_validation.process_launch.runtime_validation =
      platform::RuntimeValidationV1::runtime_validated;
  expect(!platform::validate_platform_record_v1(false_windows_validation).valid,
         "modeled Windows backend cannot claim runtime validation");

  platform::PlatformRecordV1 wrong_windows_format = windows;
  wrong_windows_format.object_format = platform::ObjectFormatV1::elf;
  expect(!platform::validate_platform_record_v1(wrong_windows_format).valid,
         "Windows record rejects ELF object format");

  platform::PlatformRecordV1 wrong_linux_process = synthetic_linux_record();
  wrong_linux_process.process_launch_backend =
      platform::ProcessLaunchBackendV1::windows_create_process;
  expect(!platform::validate_platform_record_v1(wrong_linux_process).valid,
         "Linux record rejects Windows process backend");

  platform::PlatformRecordV1 incomplete = synthetic_linux_record();
  incomplete.architecture = platform::ArchitectureKindV1::unknown;
  expect(!platform::validate_platform_record_v1(incomplete).valid,
         "complete record requires a known architecture");

  platform::PlatformRecordV1 future_version = synthetic_linux_record();
  future_version.version = platform::kPlatformRecordVersionV1 + 1;
  expect(!platform::validate_platform_record_v1(future_version).valid,
         "unknown platform-record version fails closed");

  platform::PlatformRecordV1 unknown;
  expect(platform::validate_platform_record_v1(unknown).valid,
         "well-formed unknown platform remains representable");
  expect(!platform::compiler_pipeline_supported_v1(unknown),
         "unknown platform cannot be advertised as supported");

  const std::string first = platform::format_platform_record_v1(windows);
  const std::string second = platform::format_platform_record_v1(windows);
  expect(first == second, "platform diagnostics are deterministic");
  expect(first.find("platform=windows") != std::string::npos,
         "diagnostic names the Windows platform");
  expect(first.find("compiler=clang-cl") != std::string::npos,
         "diagnostic names the clang-cl frontend");
  expect(first.find("object=coff") != std::string::npos,
         "diagnostic names the COFF object format");
  expect(first.find("executable=pe") != std::string::npos,
         "diagnostic names the PE executable format");
  expect(first.find("modeled/complete/not-validated") != std::string::npos,
         "diagnostic exposes modeled, unvalidated capability status");

  if (failures != 0) return 1;
  std::cout << platform::format_platform_record_v1(current) << '\n';
  std::cout << "platform record v1 tests PASS\n";
  return 0;
}
