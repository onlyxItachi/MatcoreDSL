#include "platform_support.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
  const std::u8string value = path.u8string();
  return std::string(reinterpret_cast<const char *>(value.data()), value.size());
}

int child_main(int argc, char **argv) {
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

void test_response_files(const std::filesystem::path &directory) {
  const std::vector<std::string> arguments = {
      "-c", "source file.cpp", "quote\"value", "utf8-çığ", ""};
  std::string error;
  const std::string windows = support::encode_response_file_utf8_v1(
      arguments, support::ResponseFileSyntaxV1::windows, error);
  expect(error.empty(), "Windows response-file encoding succeeds");
  expect(windows ==
             "-c\n\"source file.cpp\"\n\"quote\\\"value\"\n"
             "utf8-çığ\n\"\"\n",
         "Windows response-file encoding is deterministic");

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
}

}  // namespace

int main(int argc, char **argv) {
  if (argc >= 2 &&
      (std::string_view(argv[1]) == "--support-child" ||
       std::string_view(argv[1]) == "--support-child-large" ||
       std::string_view(argv[1]) == "--support-child-silent")) {
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
    test_windows_quoting();
    test_response_files(removed_path);
    test_file_snapshot(removed_path);
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
  }
  expect(!std::filesystem::exists(removed_path),
         "temporary directory is removed by ownership scope");

  if (failures != 0) return 1;
  std::cout << "platform support v1 tests PASS\n";
  return 0;
}
