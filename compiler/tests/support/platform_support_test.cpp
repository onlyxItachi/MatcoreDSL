#include "platform_support.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace support = matcore::mdslc::support;

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

std::string path_utf8(const std::filesystem::path &path) {
  std::string error;
  const std::optional<std::string> value =
      support::path_to_utf8_v1(path, error);
  return value.value_or(std::string());
}

void test_unicode_boundaries() {
  std::string error;
  const std::string valid = "folder with spaces/ü/çığ.mdsl";
  const std::optional<std::filesystem::path> path =
      support::path_from_utf8_v1(valid, error);
  expect(path.has_value() && error.empty(),
         "well-formed UTF-8 path converts to the native path type");
  const std::optional<std::string> round_trip =
      path ? support::path_to_utf8_v1(*path, error) : std::nullopt;
  expect(round_trip == valid && error.empty(),
         "native path round-trips through strict UTF-8");

  const std::string malformed("bad-\xc0\xaf", 6);
  expect(!support::path_from_utf8_v1(malformed, error) && !error.empty(),
         "overlong UTF-8 path is rejected");
  const std::string surrogate("bad-\xed\xa0\x80", 7);
  expect(!support::path_from_utf8_v1(surrogate, error) && !error.empty(),
         "UTF-8 surrogate path is rejected");

  wchar_t first[] = L"tool.exe";
  wchar_t second[] = L"path ü çığ";
  wchar_t *arguments[] = {first, second};
  const std::optional<std::vector<std::string>> converted =
      support::wide_arguments_to_utf8_v1(2, arguments, error);
  expect(converted && converted->size() == 2 &&
             (*converted)[1] == "path ü çığ" && error.empty(),
         "wide process arguments convert strictly to UTF-8");

  wchar_t invalid[] = {static_cast<wchar_t>(0xd800), L'\0'};
  wchar_t *invalid_arguments[] = {invalid};
  expect(!support::wide_arguments_to_utf8_v1(1, invalid_arguments, error) &&
             !error.empty(),
         "unpaired wide surrogate is rejected");
}

int child_main(int argc, char **argv) {
#if defined(_WIN32)
  if (argc >= 3 &&
      std::string_view(argv[1]) == "--support-child-check-handle") {
    const std::uintptr_t encoded =
        static_cast<std::uintptr_t>(std::stoull(argv[2]));
    DWORD flags = 0;
    return ::GetHandleInformation(reinterpret_cast<HANDLE>(encoded), &flags)
               ? 88
               : 0;
  }
#endif
  if (argc >= 2 && std::string_view(argv[1]) == "--support-child-silent") {
    return 7;
  }
  if (argc >= 2 && std::string_view(argv[1]) == "--support-child-large") {
    const std::string stdout_payload(131072, 'O');
    const std::string stderr_payload(131072, 'E');
    std::cout.write(stdout_payload.data(),
                    static_cast<std::streamsize>(stdout_payload.size()));
    std::cerr.write(stderr_payload.data(),
                    static_cast<std::streamsize>(stderr_payload.size()));
    return 0;
  }

  std::string error;
  std::cout << "CWD:" << path_utf8(std::filesystem::current_path()) << '\n';
  const std::optional<std::string> environment =
      support::environment_utf8_v1("MDSLC_SUPPORT_TEST_ENV", error);
  if (!error.empty()) {
    std::cerr << error << '\n';
    return 90;
  }
  std::cout << "ENV:" << environment.value_or("<unset>") << '\n';
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    std::cout << "ARG:" << argument.size() << ':' << argument << '\n';
  }
  std::cerr << "child-stderr\n";
  return 23;
}

void test_windows_quoting() {
  expect(support::quote_windows_command_line_argument_v1("tool.exe") ==
             "tool.exe",
         "simple Windows argument remains unquoted");
  expect(support::quote_windows_command_line_argument_v1("") == "\"\"",
         "empty Windows argument is represented");
  expect(support::quote_windows_command_line_argument_v1("two words") ==
             "\"two words\"",
         "space-containing Windows argument is quoted");
  expect(support::quote_windows_command_line_argument_v1("a\"b") ==
             "\"a\\\"b\"",
         "embedded quote is escaped");
  expect(support::quote_windows_command_line_argument_v1(
             "C:\\Program Files\\LLVM\\") ==
             "\"C:\\Program Files\\LLVM\\\\\"",
         "trailing backslash is doubled before the closing quote");
  expect(support::quote_windows_command_line_argument_v1(
             "slashes\\\\\"quote space") ==
             "\"slashes\\\\\\\\\\\"quote space\"",
         "backslashes immediately before a quote are doubled and escaped");

  const std::vector<std::string> argv = {
      "C:\\Program Files\\LLVM\\clang-cl.exe", "/c", "source file.cpp",
      "/DNAME=çığ"};
  expect(support::build_windows_command_line_v1(argv) ==
             "\"C:\\Program Files\\LLVM\\clang-cl.exe\" /c "
             "\"source file.cpp\" /DNAME=çığ",
         "Windows command-line construction preserves UTF-8 and boundaries");
}

void test_compiler_argument_policy() {
  using Risk = support::CompilerArgumentRiskV1;
  expect(support::ascii_case_equal_v1("OuT", "out"),
         "ASCII option comparison is case-insensitive");
  expect(support::windows_option_equals_v1("-LiNk", "link"),
         "Windows option comparison accepts dash and mixed case");
  expect(support::windows_option_starts_with_v1("/Tcfixture.c", "tc"),
         "Windows joined option comparison is case-insensitive");
  expect(support::clang_cl_option_consumes_next_v1("/FI") &&
             support::clang_cl_option_consumes_next_v1("/imsvc") &&
             support::clang_cl_option_consumes_next_v1("/external:I") &&
             support::clang_cl_option_consumes_next_v1("/winsysroot") &&
             !support::clang_cl_option_consumes_next_v1("/fi") &&
             support::clang_cl_option_is_link_context_v1("/winsysroot") &&
             support::clang_cl_option_is_link_context_v1(
                 "/winsysrootC:\\Windows SDK") &&
             support::clang_cl_option_is_link_context_v1(
                 "--sysroot=C:/Windows SDK") &&
             !support::clang_cl_option_is_link_context_v1("/FI"),
         "audited clang-cl separated-value arity and propagation are exact");
  expect(support::compiler_consumed_value_is_safe_v1("ordinary value") &&
             !support::compiler_consumed_value_is_safe_v1("@nested.rsp"),
         "nested response expansion is forbidden in consumed values");

  expect(support::classify_untrusted_compiler_argument_v1(
             "/ClAnG:-serialize-diagnostics", true) ==
             Risk::opaque_forwarding,
         "opaque clang-cl forwarding is rejected as one policy class");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-Xclang=-load", true) == Risk::opaque_forwarding &&
             support::classify_untrusted_compiler_argument_v1(
                 "--driver-mode=g++", true) == Risk::opaque_forwarding &&
             support::classify_untrusted_compiler_argument_v1(
                 "-mllvm=untrusted", true) == Risk::opaque_forwarding,
         "joined cc1 forwarding and driver-mode overrides are rejected");
  expect(support::classify_untrusted_compiler_argument_v1(
             "/Fd:sentinel.mdsl", true) == Risk::output_producing,
         "clang-cl auxiliary output is rejected");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-Fo:sentinel.obj", true) == Risk::output_producing &&
             support::classify_untrusted_compiler_argument_v1(
                 "-Fd:sentinel.pdb", true) == Risk::output_producing,
         "dash-prefixed clang-cl auxiliary outputs are rejected");
  expect(support::classify_untrusted_compiler_argument_v1(
             "/favor:AMD64", true) == Risk::none,
         "clang-cl CPU tuning is not confused with /Fa output");
  expect(support::classify_untrusted_compiler_argument_v1(
             "/FItrusted-header.h", true) == Risk::none &&
             support::classify_untrusted_compiler_argument_v1(
                 "/fp:fast", true) == Risk::none &&
             support::classify_untrusted_compiler_argument_v1(
                 "/Fiuntrusted.i", true) == Risk::output_producing,
         "case-sensitive clang-cl /FI input and /Fi output remain distinct");
  expect(support::classify_untrusted_compiler_argument_v1(
             "/C", true) == Risk::unsafe_control &&
             support::classify_untrusted_compiler_argument_v1(
                 "/c", true) == Risk::unsafe_control &&
             support::classify_untrusted_compiler_argument_v1(
                 "/Tpfixture.cpp", true) == Risk::unsafe_control,
         "clang-cl compiler-mode policy rejects non-object actions and "
         "preserves exact option case");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-MD", true) == Risk::none &&
             support::classify_untrusted_compiler_argument_v1(
                 "-MT", true) == Risk::none &&
             support::classify_untrusted_compiler_argument_v1(
                 "-MTd", true) == Risk::none &&
             support::classify_untrusted_compiler_argument_v1(
                 "-openmp", true) == Risk::unsafe_control &&
             support::classify_untrusted_compiler_argument_v1(
                 "-openmp:experimental", true) == Risk::unsafe_control &&
             support::classify_untrusted_compiler_argument_v1(
                 "-openmp-", true) == Risk::unsafe_control &&
             support::classify_untrusted_compiler_argument_v1(
                 "-MMD", true) != Risk::none,
         "clang-cl CRT flags remain legal while unpackageable OpenMP modes "
         "and GNU dependency outputs fail closed");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-TC", true) == Risk::unsafe_control,
         "dash-prefixed clang-cl language override is rejected");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-ftime-trace=sentinel.mdsl", true) ==
             Risk::output_producing,
         "Clang time-trace output is rejected");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-serialize-diagnostics", false) == Risk::output_producing,
         "serialized diagnostic output is rejected in GNU mode");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-fmodule-output=sentinel.mdsl", false) ==
             Risk::output_producing,
         "module output is rejected");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-fsanitize=address", true) == Risk::none,
         "ordinary audited sanitizer selection remains allowed");
  expect(support::classify_untrusted_compiler_argument_v1(
                 "-fPIC", true) == Risk::none &&
             support::classify_untrusted_compiler_argument_v1(
                 "-fomit-frame-pointer", true) == Risk::none,
         "ordinary lowercase Clang -f options are not parsed as /F outputs");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-fmodules", true) == Risk::unsafe_control &&
             support::classify_untrusted_compiler_argument_v1(
                 "/Zi", true) == Risk::output_producing &&
             support::classify_untrusted_compiler_argument_v1(
                 "/Z7", true) == Risk::none &&
             support::classify_untrusted_compiler_argument_v1(
                 "/GL", true) == Risk::unsafe_control,
         "implicit module caches, external PDBs, and LTO modes fail closed");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-gsplit-dwarf", true) == Risk::output_producing,
         "split DWARF sidecar output fails closed");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-xc++-header", true) == Risk::unsafe_control &&
             support::classify_untrusted_compiler_argument_v1(
                 "--precompile", true) == Risk::unsafe_control,
         "joined language and precompile actions fail closed");
  expect(support::classify_untrusted_compiler_argument_v1(
             "/fd/absolute/include", false) == Risk::none,
         "GNU-mode absolute paths are not parsed as clang-cl options");
  expect(support::classify_untrusted_compiler_argument_v1(
             "-fpass-plugin=untrusted.dll", true) != Risk::none,
         "compiler pass plugins are rejected");
  expect(support::classify_untrusted_compiler_argument_v1(
             "--config-user-dir=untrusted", true) == Risk::unsafe_control,
         "compiler configuration directories are rejected");
  expect(support::classify_untrusted_compiler_argument_v1(
             "/Yuattacker.pch", true) == Risk::unsafe_control,
         "clang-cl precompiled-header controls are rejected");

  const std::vector<support::EnvironmentOverrideV1> sanitization =
      support::compiler_environment_sanitization_v1();
  const auto removes = [&](std::string_view name) {
    return std::any_of(sanitization.begin(), sanitization.end(),
                       [&](const auto &entry) {
                         return entry.name == name && !entry.value;
                       });
  };
  expect(removes("CL") && removes("_CL_") && removes("LINK") &&
             removes("_LINK_") && removes("CCC_OVERRIDE_OPTIONS") &&
             removes("CLANG_CONFIG_PATH") && removes("CC_PRINT_OPTIONS") &&
             removes("CC_PRINT_OPTIONS_FILE") && removes("LINK_REPRO") &&
             removes("LLD_REPRODUCE"),
         "compiler child environment removes hidden driver/linker inputs");
}

void test_prospective_path_identity(const std::filesystem::path &directory) {
  std::string error;
  const auto first = directory / "Prospective-Output.lib";
  const auto second = directory / "prospective-output.LIB";
  const bool same =
      support::paths_refer_to_same_location_v1(first, second, error);
  expect(error.empty(), "prospective output path comparison succeeds");
#if defined(_WIN32)
  expect(same, "Windows prospective output comparison is case-insensitive");
  error.clear();
  expect(support::paths_refer_to_same_location_v1(
             directory / "prospective-trailing.lib",
             directory / "prospective-trailing.lib.", error) &&
             error.empty(),
         "Windows prospective output comparison normalizes trailing dots");
  error.clear();
  expect(support::paths_refer_to_same_location_v1(
             directory / "prospective-space.lib",
             directory / "prospective-space.lib ", error) &&
             error.empty(),
         "Windows prospective output comparison normalizes trailing spaces");
  error.clear();
  expect(!support::prospective_output_path_supported_v1(
             directory / "NUL.lib", error) &&
             !error.empty(),
         "Windows reserved device output is rejected");
  error.clear();
  expect(!support::prospective_output_path_supported_v1(
             directory / "result.lib.", error) &&
             !error.empty(),
         "Windows trailing-dot output is rejected");
#else
  expect(!same, "case-sensitive hosts preserve distinct prospective paths");
#endif
  error.clear();
  expect(support::prospective_output_path_supported_v1(
             directory / "ordinary-output.bin", error) &&
             error.empty(),
         "ordinary prospective output is supported");
}

void test_response_files(const std::filesystem::path &directory) {
  const std::vector<std::string> arguments = {
      "-c", "source file.cpp", "quote\"value", "utf8-çığ", ""};
  std::string error;
  const std::string windows = support::encode_response_file_utf8_v1(
      arguments, support::ResponseFileSyntaxV1::windows, error);
  expect(error.empty(), "Windows response-file encoding succeeds");
  expect(windows ==
             "-c \"source file.cpp\" \"quote\\\"value\" "
             "utf8-çığ \"\"\n",
         "Windows response-file encoding preserves one logical argv line");

  const std::string gnu = support::encode_response_file_utf8_v1(
      arguments, support::ResponseFileSyntaxV1::gnu, error);
  expect(error.empty(), "GNU response-file encoding succeeds");
  expect(gnu ==
             "-c\n\"source file.cpp\"\n\"quote\\\"value\"\n"
             "utf8-çığ\n\"\"\n",
         "GNU response-file encoding is deterministic");

  const std::filesystem::path response_path =
      directory / std::filesystem::path(u8"args with spaces-çığ.rsp");
  expect(support::write_response_file_utf8_v1(
             response_path, arguments,
             support::ResponseFileSyntaxV1::windows, error),
         "UTF-8 response file is written through a space-containing path");
  std::ifstream input(response_path, std::ios::binary);
  const std::string actual((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  expect(actual == windows, "response file bytes match the encoded UTF-8");

  std::vector<std::string> invalid = {std::string("bad\0arg", 7)};
  (void)support::encode_response_file_utf8_v1(
      invalid, support::ResponseFileSyntaxV1::windows, error);
  expect(!error.empty(), "response-file NUL injection is rejected");
  (void)support::encode_response_file_utf8_v1(
      {"line\nbreak"}, support::ResponseFileSyntaxV1::windows, error);
  expect(!error.empty(), "response-file line-boundary injection is rejected");
}

void test_argument_files(const std::filesystem::path &directory) {
  const std::filesystem::path path = directory / u8"compiler arguments ü.v1";
  const std::vector<std::string> expected = {
      "clang-cl.exe", "/TP", "/DVALUE=space value", "path ü.mdsl", ""};
  std::string error;
  expect(support::write_argument_file_v1(path, expected, error) &&
             error.empty(),
         "bounded argument file is written");
  const std::optional<std::vector<std::string>> actual =
      support::read_argument_file_v1(path, error);
  expect(actual == expected && error.empty(),
         "bounded argument file round-trips exact UTF-8 argv");

  const std::filesystem::path malformed = directory / "malformed-args.v1";
  {
    std::ofstream output(malformed, std::ios::binary);
    output << "MDSLC-ARGV-V1\r\ntruncated";
  }
  expect(!support::read_argument_file_v1(malformed, error) && !error.empty(),
         "malformed argument file fails closed");
}

void test_file_snapshot(const std::filesystem::path &directory) {
  const std::filesystem::path fixture =
      directory / std::filesystem::path(u8"fixture UTF8 ç.txt");
  {
    std::ofstream output(fixture, std::ios::binary);
    output << "payload";
  }

  std::string error;
  const support::FileSnapshotV1 first =
      support::capture_file_snapshot_v1(fixture, error);
  expect(error.empty(), "file snapshot succeeds");
  expect(first.exists && first.regular_file && first.size_bytes == 7,
         "file snapshot records basic metadata");
  expect(static_cast<bool>(first.identity),
         "file snapshot records native identity");

  const std::filesystem::path hardlink = directory / "fixture-hardlink.txt";
  std::error_code filesystem_error;
  std::filesystem::create_hard_link(fixture, hardlink, filesystem_error);
  expect(!filesystem_error, "hard link fixture can be created");
  const support::FileSnapshotV1 second =
      support::capture_file_snapshot_v1(hardlink, error);
  expect(error.empty(), "hard-link snapshot succeeds");
  expect(support::same_file_identity_v1(first.identity, second.identity),
         "hard links share the same native file identity");

  const std::filesystem::path nested = directory / "nested";
  std::filesystem::create_directory(nested, filesystem_error);
  const std::filesystem::path normalized = support::normalize_path_v1(
      nested / ".." / fixture.filename(), true, error);
  expect(error.empty(), "path normalization succeeds");
  expect(normalized == first.normalized_path,
         "normalization resolves lexical parent components");

  const support::FileSnapshotV1 invalid =
      support::capture_file_snapshot_v1({}, error);
  expect(!error.empty() && !invalid.exists && !invalid.identity,
         "invalid snapshot input fails closed without an identity");
}

void test_atomic_replacement(const std::filesystem::path &directory) {
  const std::filesystem::path destination = directory / u8"published ü.txt";
  const std::filesystem::path temporary = directory / u8"replacement ü.tmp";
  {
    std::ofstream output(destination, std::ios::binary);
    output << "old";
  }
  {
    std::ofstream output(temporary, std::ios::binary);
    output << "new";
  }
  std::string error;
  expect(support::replace_file_atomically_v1(temporary, destination, error) &&
             error.empty(),
         "atomic publication replaces an existing destination");
  std::ifstream input(destination, std::ios::binary);
  const std::string contents((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  expect(contents == "new" && !std::filesystem::exists(temporary),
         "atomic publication exposes replacement bytes and consumes temp");
}

void test_process(const std::filesystem::path &directory,
                  const std::filesystem::path &self) {
  std::string error;
  const std::optional<std::filesystem::path> discovered =
      support::find_executable_v1(path_utf8(self), error);
  expect(discovered.has_value() && error.empty(),
         "explicit executable discovery succeeds");

  support::ProcessRequestV1 request;
  request.argv = {path_utf8(self), "--support-child", "space value",
                  "quote\"value", "utf8-çığ", "semi;&|$()"};
  request.working_directory = directory;
  request.environment = {{"MDSLC_SUPPORT_TEST_ENV", "value with spaces-çığ"}};
  const support::ProcessResultV1 result = support::run_process_v1(request);
  expect(result.launched, "vector-argv child process launches without a shell");
  expect(result.error.empty(), "child process has no launch/wait error");
  expect(result.exit_code == 23, "child exit status is propagated");
  expect(result.stdout_text.find("ENV:value with spaces-çığ\n") !=
             std::string::npos,
         "child environment override preserves UTF-8 and spaces");
  expect(result.stdout_text.find("ARG:11:space value\n") != std::string::npos,
         "space-containing argument remains one argument");
  expect(result.stdout_text.find("ARG:11:quote\"value\n") != std::string::npos,
         "quote-containing argument round-trips");
  expect(result.stdout_text.find("ARG:11:utf8-çığ\n") != std::string::npos,
         "UTF-8 argument round-trips");
  expect(result.stdout_text.find("ARG:10:semi;&|$()\n") != std::string::npos,
         "shell metacharacters remain inert argument bytes");
  expect(result.stdout_text.find("CWD:" + path_utf8(directory) + "\n") !=
             std::string::npos,
         "requested working directory is applied");
  expect(result.stderr_text == "child-stderr\n",
         "stderr is captured separately");

  support::ProcessRequestV1 large;
  large.argv = {path_utf8(self), "--support-child-large"};
  const support::ProcessResultV1 large_result = support::run_process_v1(large);
  expect(large_result.launched && large_result.exit_code == 0,
         "large-output child completes");
  expect(large_result.stdout_text.size() == 131072 &&
             large_result.stderr_text.size() == 131072,
         "stdout and stderr are drained concurrently without truncation");

  support::ProcessRequestV1 invalid;
  const support::ProcessResultV1 invalid_result =
      support::run_process_v1(invalid);
  expect(!invalid_result.launched && !invalid_result.error.empty(),
         "empty argv fails before launch");

  support::ProcessRequestV1 future = request;
  future.version = support::kPlatformSupportApiVersionV1 + 1;
  const support::ProcessResultV1 future_result =
      support::run_process_v1(future);
  expect(!future_result.launched && !future_result.error.empty(),
         "unknown process request version fails closed");

  support::ProcessRequestV1 missing_directory = request;
  missing_directory.working_directory = directory / "does-not-exist";
  const support::ProcessResultV1 missing_directory_result =
      support::run_process_v1(missing_directory);
  expect(!missing_directory_result.launched &&
             !missing_directory_result.error.empty(),
         "invalid working directory fails before reporting a launch");

  support::ProcessRequestV1 inherited;
  inherited.argv = {path_utf8(self), "--support-child-silent"};
  inherited.capture_stdout = false;
  inherited.capture_stderr = false;
  const support::ProcessResultV1 inherited_result =
      support::run_process_v1(inherited);
  expect(inherited_result.launched && inherited_result.exit_code == 7 &&
             inherited_result.stdout_text.empty() &&
             inherited_result.stderr_text.empty(),
         "uncaptured standard streams are inherited without synthetic data");

#if defined(_WIN32)
  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;
  const HANDLE unrelated = ::CreateEventW(&security, TRUE, FALSE, nullptr);
  expect(unrelated != nullptr,
         "inheritable handle fixture is created for leak test");
  if (unrelated != nullptr) {
    support::ProcessRequestV1 handle_check;
    handle_check.argv = {
        path_utf8(self), "--support-child-check-handle",
        std::to_string(reinterpret_cast<std::uintptr_t>(unrelated))};
    const support::ProcessResultV1 handle_result =
        support::run_process_v1(handle_check);
    expect(handle_result.launched && handle_result.exit_code == 0,
           "unrelated inheritable handles are excluded from child process");
    ::CloseHandle(unrelated);
  }
#endif
}

void test_windows_path_shadow(const std::filesystem::path &directory,
                              const std::filesystem::path &self) {
#if defined(_WIN32)
  const std::filesystem::path shadow = directory / "shadow-probe.exe";
  const std::filesystem::path empty_path = directory / "empty-path";
  std::error_code filesystem_error;
  std::filesystem::create_directory(empty_path, filesystem_error);
  expect(!filesystem_error, "empty PATH fixture directory is created");
  std::filesystem::copy_file(self, shadow,
                             std::filesystem::copy_options::overwrite_existing,
                             filesystem_error);
  expect(!filesystem_error, "current-directory shadow executable is created");
  const std::filesystem::path old_current = std::filesystem::current_path();
  std::string error;
  const std::optional<std::string> old_path =
      support::environment_utf8_v1("PATH", error);
  const std::optional<std::string> empty_path_utf8 =
      support::path_to_utf8_v1(empty_path, error);
  const std::wstring empty_path_wide = empty_path.native();
  if (!filesystem_error && empty_path_utf8) {
    std::filesystem::current_path(directory, filesystem_error);
    expect(!filesystem_error, "shadow fixture becomes current directory");
    expect(::SetEnvironmentVariableW(L"PATH", empty_path_wide.c_str()) != 0,
           "isolated PATH is installed for shadow test");
    const std::optional<std::filesystem::path> discovered =
        support::find_executable_v1("shadow-probe.exe", error);
    expect(!discovered && !error.empty(),
           "executable discovery rejects current-directory shadowing");
    std::filesystem::current_path(old_current, filesystem_error);
    if (old_path) {
      const int wide_count = ::MultiByteToWideChar(
          CP_UTF8, MB_ERR_INVALID_CHARS, old_path->data(),
          static_cast<int>(old_path->size()), nullptr, 0);
      std::wstring old_path_wide(static_cast<std::size_t>(wide_count), L'\0');
      ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, old_path->data(),
                            static_cast<int>(old_path->size()),
                            old_path_wide.data(), wide_count);
      ::SetEnvironmentVariableW(L"PATH", old_path_wide.c_str());
    } else {
      ::SetEnvironmentVariableW(L"PATH", nullptr);
    }
  }
#else
  (void)directory;
  (void)self;
#endif
}

}  // namespace

int support_test_main(int argc, char **argv) {
  if (argc >= 2 &&
      (std::string_view(argv[1]) == "--support-child" ||
       std::string_view(argv[1]) == "--support-child-large" ||
       std::string_view(argv[1]) == "--support-child-silent" ||
       std::string_view(argv[1]) == "--support-child-check-handle")) {
    return child_main(argc, argv);
  }

  std::string error;
  expect(support::process_launch_backend_v1() !=
             support::ProcessLaunchBackendV1::unsupported,
         "host process backend is available");
  const std::optional<std::filesystem::path> self =
      support::current_executable_path_v1(error);
  expect(self.has_value() && error.empty(),
         "current executable path is discoverable");
  if (!self) return 1;

  const std::optional<std::string> path =
      support::environment_utf8_v1("PATH", error);
  expect(path.has_value() && error.empty(), "PATH is readable as UTF-8");
  const std::optional<std::filesystem::path> self_by_path =
      support::find_executable_v1(path_utf8(*self), error);
  expect(self_by_path.has_value() && error.empty(),
         "current executable is discoverable by explicit path");

  std::filesystem::path removed_path;
  {
    std::optional<support::TempDirectoryV1> temporary =
        support::create_temp_directory_v1("mdslc support-çığ", error);
    expect(temporary.has_value() && error.empty(),
           "temporary directory supports spaces and UTF-8");
    if (!temporary) return 1;
    removed_path = temporary->path();
    expect(std::filesystem::is_directory(removed_path),
           "temporary directory exists while owned");
    test_unicode_boundaries();
    test_windows_quoting();
    test_compiler_argument_policy();
    test_prospective_path_identity(removed_path);
    test_response_files(removed_path);
    test_argument_files(removed_path);
    test_file_snapshot(removed_path);
    test_atomic_replacement(removed_path);
    std::filesystem::path copied_self =
        removed_path / std::filesystem::path(u8"child executable ç");
    copied_self += self->extension();
    std::error_code copy_error;
    std::filesystem::copy_file(*self, copied_self,
                               std::filesystem::copy_options::overwrite_existing,
                               copy_error);
    expect(!copy_error, "test executable copies into a space/UTF-8 path");
#if !defined(_WIN32)
    std::filesystem::permissions(
        copied_self,
        std::filesystem::perms::owner_exec |
            std::filesystem::perms::group_exec |
            std::filesystem::perms::others_exec,
        std::filesystem::perm_options::add, copy_error);
    expect(!copy_error, "copied test executable remains executable");
#endif
    if (!copy_error) test_process(removed_path, copied_self);
    test_windows_path_shadow(removed_path, *self);
  }
  expect(!std::filesystem::exists(removed_path),
         "temporary directory is removed by ownership scope");

  if (failures != 0) return 1;
  std::cout << "platform support v1 tests PASS\n";
  return 0;
}

#if defined(_WIN32)
int wmain(int argc, wchar_t **argv) {
  std::string error;
  const std::optional<std::vector<std::string>> utf8 =
      support::wide_arguments_to_utf8_v1(argc, argv, error);
  if (!utf8) {
    std::cerr << "FAIL: cannot decode test process arguments: " << error
              << '\n';
    return 1;
  }
  std::vector<char *> narrow;
  narrow.reserve(utf8->size());
  for (const std::string &argument : *utf8) {
    narrow.push_back(const_cast<char *>(argument.c_str()));
  }
  return support_test_main(static_cast<int>(narrow.size()), narrow.data());
}
#else
int main(int argc, char **argv) { return support_test_main(argc, argv); }
#endif
