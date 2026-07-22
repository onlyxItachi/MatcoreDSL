#ifndef MATCORE_MDSLC_SUPPORT_PLATFORM_SUPPORT_H
#define MATCORE_MDSLC_SUPPORT_PLATFORM_SUPPORT_H

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace matcore::mdslc::support {

inline constexpr std::uint32_t kPlatformSupportApiVersionV1 = 1;

enum class ProcessLaunchBackendV1 : std::uint8_t {
  unsupported = 0,
  posix_fork_exec = 1,
  windows_create_process_w = 2,
};

enum class FileIdentityKindV1 : std::uint8_t {
  unavailable = 0,
  posix_device_inode = 1,
  windows_volume_file_id = 2,
};

enum class ResponseFileSyntaxV1 : std::uint8_t {
  gnu = 1,
  windows = 2,
};

struct EnvironmentOverrideV1 {
  std::string name;
  // A missing value removes the variable from the child environment.
  std::optional<std::string> value;
};

struct ProcessRequestV1 {
  std::uint32_t version = kPlatformSupportApiVersionV1;
  std::vector<std::string> argv;
  std::filesystem::path working_directory;
  std::vector<EnvironmentOverrideV1> environment;
  bool capture_stdout = true;
  bool capture_stderr = true;
};

struct ProcessResultV1 {
  std::uint32_t version = kPlatformSupportApiVersionV1;
  bool launched = false;
  std::int64_t exit_code = -1;
  std::string stdout_text;
  std::string stderr_text;
  std::string error;
};

struct FileIdentityV1 {
  FileIdentityKindV1 kind = FileIdentityKindV1::unavailable;
  std::array<std::uint64_t, 2> words{};

  constexpr explicit operator bool() const noexcept {
    return kind != FileIdentityKindV1::unavailable;
  }

  friend constexpr bool operator==(const FileIdentityV1 &,
                                   const FileIdentityV1 &) = default;
};

struct PathComponentSnapshotV1 {
  FileIdentityV1 identity;
  std::array<std::uint64_t, 6> metadata_words{};
  std::string symbolic_link_target;

  friend bool operator==(const PathComponentSnapshotV1 &,
                         const PathComponentSnapshotV1 &) = default;
};

struct FileSnapshotV1 {
  std::uint32_t version = kPlatformSupportApiVersionV1;
  std::filesystem::path normalized_path;
  bool exists = false;
  bool regular_file = false;
  std::uint64_t size_bytes = 0;
  std::int64_t last_write_time_ticks = 0;
  FileIdentityV1 identity;
  // Identity of each path component as traversed, including symlink/reparse
  // components. This prevents a dependency path from being retargeted while
  // keeping identical final bytes.
  std::vector<PathComponentSnapshotV1> path_identity_chain;
};

class TempDirectoryV1 {
 public:
  TempDirectoryV1() = default;
  TempDirectoryV1(const TempDirectoryV1 &) = delete;
  TempDirectoryV1 &operator=(const TempDirectoryV1 &) = delete;
  TempDirectoryV1(TempDirectoryV1 &&other) noexcept;
  TempDirectoryV1 &operator=(TempDirectoryV1 &&other) noexcept;
  ~TempDirectoryV1();

  const std::filesystem::path &path() const noexcept { return path_; }
  explicit operator bool() const noexcept { return !path_.empty(); }

  // Relinquishes automatic removal and returns the directory path.
  std::filesystem::path release() noexcept;

 private:
  friend std::optional<TempDirectoryV1> create_temp_directory_v1(
      std::string_view, std::string &);
  explicit TempDirectoryV1(std::filesystem::path path)
      : path_(std::move(path)) {}

  void remove() noexcept;
  std::filesystem::path path_;
};

ProcessLaunchBackendV1 process_launch_backend_v1() noexcept;

std::optional<std::filesystem::path> current_executable_path_v1(
    std::string &error);

ProcessResultV1 run_process_v1(const ProcessRequestV1 &request);

std::optional<TempDirectoryV1> create_temp_directory_v1(
    std::string_view prefix, std::string &error);

std::optional<std::string> environment_utf8_v1(std::string_view name,
                                               std::string &error);

std::optional<std::filesystem::path> find_executable_v1(
    std::string_view name, std::string &error);

// These are the only supported narrow-string boundary for filesystem paths.
// UTF-8 is validated strictly before conversion; malformed UTF-8 and invalid
// UTF-16 process arguments fail closed.
std::optional<std::filesystem::path> path_from_utf8_v1(
    std::string_view value, std::string &error);

std::optional<std::string> path_to_utf8_v1(
    const std::filesystem::path &path, std::string &error);

std::optional<std::vector<std::string>> wide_arguments_to_utf8_v1(
    int argc, wchar_t *const *argv, std::string &error);

std::filesystem::path normalize_path_v1(const std::filesystem::path &path,
                                        bool resolve_existing,
                                        std::string &error);

FileSnapshotV1 capture_file_snapshot_v1(const std::filesystem::path &path,
                                        std::string &error);

bool same_file_identity_v1(const FileIdentityV1 &left,
                           const FileIdentityV1 &right) noexcept;

// Implements the CommandLineToArgvW-compatible quoting convention used for
// CreateProcessW command lines. It is pure and tested on every host.
std::string quote_windows_command_line_argument_v1(std::string_view argument);

std::string build_windows_command_line_v1(
    const std::vector<std::string> &argv);

std::string encode_response_file_utf8_v1(
    const std::vector<std::string> &arguments, ResponseFileSyntaxV1 syntax,
    std::string &error);

bool write_response_file_utf8_v1(
    const std::filesystem::path &path,
    const std::vector<std::string> &arguments, ResponseFileSyntaxV1 syntax,
    std::string &error);

// A bounded, binary, length-prefixed argv transport for launching helper
// tools without approaching Windows' CreateProcessW command-line limit.
bool write_argument_file_v1(const std::filesystem::path &path,
                            const std::vector<std::string> &arguments,
                            std::string &error);

std::optional<std::vector<std::string>> read_argument_file_v1(
    const std::filesystem::path &path, std::string &error);

}  // namespace matcore::mdslc::support

#endif
