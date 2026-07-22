#include "platform_support_backend.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace matcore::mdslc::support {

ProcessLaunchBackendV1 process_launch_backend_v1() noexcept {
  return ProcessLaunchBackendV1::posix_fork_exec;
}

namespace detail {
namespace {

class FileDescriptor {
 public:
  FileDescriptor() = default;
  explicit FileDescriptor(int value) : value_(value) {}
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  FileDescriptor(FileDescriptor &&other) noexcept : value_(other.release()) {}
  FileDescriptor &operator=(FileDescriptor &&other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
  }
  ~FileDescriptor() { reset(); }

  int get() const noexcept { return value_; }
  int release() noexcept {
    const int result = value_;
    value_ = -1;
    return result;
  }
  void reset(int value = -1) noexcept {
    if (value_ >= 0) ::close(value_);
    value_ = value;
  }

 private:
  int value_ = -1;
};

bool make_pipe(FileDescriptor &read_end, FileDescriptor &write_end,
               std::string &error) {
  int descriptors[2] = {-1, -1};
  if (::pipe(descriptors) != 0) {
    error = "pipe failed: " + std::string(std::strerror(errno));
    return false;
  }
  read_end.reset(descriptors[0]);
  write_end.reset(descriptors[1]);
  if (::fcntl(read_end.get(), F_SETFD, FD_CLOEXEC) != 0 ||
      ::fcntl(write_end.get(), F_SETFD, FD_CLOEXEC) != 0) {
    error = "cannot mark pipe close-on-exec: " +
            std::string(std::strerror(errno));
    return false;
  }
  return true;
}

void read_all(int descriptor, std::string &output) {
  char buffer[8192];
  for (;;) {
    const ssize_t count = ::read(descriptor, buffer, sizeof(buffer));
    if (count > 0) {
      output.append(buffer, static_cast<std::size_t>(count));
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    return;
  }
}

bool executable_file(const std::filesystem::path &path) {
  struct stat status {};
  return ::stat(path.c_str(), &status) == 0 && S_ISREG(status.st_mode) &&
         ::access(path.c_str(), X_OK) == 0;
}

std::optional<std::filesystem::path> resolve_process_executable(
    const ProcessRequestV1 &request, std::string &error) {
  const std::filesystem::path requested(request.argv.front());
  if (requested.has_parent_path() && requested.is_relative() &&
      !request.working_directory.empty()) {
    const std::filesystem::path candidate =
        request.working_directory / requested;
    if (!executable_file(candidate)) {
      error = "process executable is not executable: " + candidate.string();
      return std::nullopt;
    }
    return normalize_path_v1(candidate, true, error);
  }
  return find_executable_native_v1(request.argv.front(), error);
}

void report_child_failure(int descriptor, int child_errno) noexcept {
  const char *bytes = reinterpret_cast<const char *>(&child_errno);
  std::size_t written = 0;
  while (written < sizeof(child_errno)) {
    const ssize_t count =
        ::write(descriptor, bytes + written, sizeof(child_errno) - written);
    if (count > 0) {
      written += static_cast<std::size_t>(count);
    } else if (count < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
}

}  // namespace

std::optional<std::filesystem::path> current_executable_path_native_v1(
    std::string &error) {
  std::vector<char> buffer(256);
  for (;;) {
    const ssize_t count =
        ::readlink("/proc/self/exe", buffer.data(), buffer.size());
    if (count < 0) {
      error = "cannot resolve /proc/self/exe: " +
              std::string(std::strerror(errno));
      return std::nullopt;
    }
    if (static_cast<std::size_t>(count) < buffer.size()) {
      return std::filesystem::path(
          std::string(buffer.data(), static_cast<std::size_t>(count)));
    }
    if (buffer.size() > (1U << 20U)) {
      error = "current executable path exceeds one MiB";
      return std::nullopt;
    }
    buffer.resize(buffer.size() * 2);
  }
}

ProcessResultV1 run_process_native_v1(const ProcessRequestV1 &request) {
  ProcessResultV1 result;
  std::string discovery_error;
  const std::optional<std::filesystem::path> executable =
      resolve_process_executable(request, discovery_error);
  if (!executable) {
    result.error = std::move(discovery_error);
    return result;
  }

  FileDescriptor stdout_read;
  FileDescriptor stdout_write;
  FileDescriptor stderr_read;
  FileDescriptor stderr_write;
  FileDescriptor error_read;
  FileDescriptor error_write;
  if (request.capture_stdout &&
      !make_pipe(stdout_read, stdout_write, result.error)) {
    return result;
  }
  if (request.capture_stderr &&
      !make_pipe(stderr_read, stderr_write, result.error)) {
    return result;
  }
  if (!make_pipe(error_read, error_write, result.error)) return result;

  std::vector<char *> argv;
  argv.reserve(request.argv.size() + 1);
  for (const std::string &argument : request.argv) {
    argv.push_back(const_cast<char *>(argument.c_str()));
  }
  argv.push_back(nullptr);

  const pid_t child = ::fork();
  if (child < 0) {
    result.error = "fork failed: " + std::string(std::strerror(errno));
    return result;
  }
  if (child == 0) {
    error_read.reset();
    stdout_read.reset();
    stderr_read.reset();

    if (!request.working_directory.empty() &&
        ::chdir(request.working_directory.c_str()) != 0) {
      report_child_failure(error_write.get(), errno);
      ::_exit(126);
    }
    for (const EnvironmentOverrideV1 &entry : request.environment) {
      const int status = entry.value
          ? ::setenv(entry.name.c_str(), entry.value->c_str(), 1)
          : ::unsetenv(entry.name.c_str());
      if (status != 0) {
        report_child_failure(error_write.get(), errno);
        ::_exit(126);
      }
    }
    if (request.capture_stdout &&
        ::dup2(stdout_write.get(), STDOUT_FILENO) < 0) {
      report_child_failure(error_write.get(), errno);
      ::_exit(126);
    }
    if (request.capture_stderr &&
        ::dup2(stderr_write.get(), STDERR_FILENO) < 0) {
      report_child_failure(error_write.get(), errno);
      ::_exit(126);
    }
    stdout_write.reset();
    stderr_write.reset();

    ::execv(executable->c_str(), argv.data());
    report_child_failure(error_write.get(), errno);
    ::_exit(127);
  }

  error_write.reset();
  stdout_write.reset();
  stderr_write.reset();

  int child_errno = 0;
  std::size_t error_bytes = 0;
  while (error_bytes < sizeof(child_errno)) {
    const ssize_t count = ::read(
        error_read.get(), reinterpret_cast<char *>(&child_errno) + error_bytes,
        sizeof(child_errno) - error_bytes);
    if (count > 0) {
      error_bytes += static_cast<std::size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    break;
  }
  error_read.reset();

  std::thread stdout_reader;
  std::thread stderr_reader;
  if (request.capture_stdout) {
    stdout_reader = std::thread(read_all, stdout_read.get(),
                                std::ref(result.stdout_text));
  }
  if (request.capture_stderr) {
    stderr_reader = std::thread(read_all, stderr_read.get(),
                                std::ref(result.stderr_text));
  }

  int wait_status = 0;
  while (::waitpid(child, &wait_status, 0) < 0) {
    if (errno == EINTR) continue;
    result.error = "waitpid failed: " + std::string(std::strerror(errno));
    break;
  }
  if (stdout_reader.joinable()) stdout_reader.join();
  if (stderr_reader.joinable()) stderr_reader.join();
  stdout_read.reset();
  stderr_read.reset();

  if (error_bytes != 0) {
    if (error_bytes == sizeof(child_errno)) {
      result.error = "child setup or exec failed: " +
                     std::string(std::strerror(child_errno));
    } else {
      result.error = "child setup or exec failed with a truncated error";
    }
    return result;
  }
  if (!result.error.empty()) return result;

  result.launched = true;
  if (WIFEXITED(wait_status)) {
    result.exit_code = WEXITSTATUS(wait_status);
  } else if (WIFSIGNALED(wait_status)) {
    result.exit_code = 128 + WTERMSIG(wait_status);
  } else {
    result.exit_code = -1;
    result.error = "child ended with an unsupported wait status";
  }
  return result;
}

std::optional<std::filesystem::path> create_temp_directory_native_v1(
    std::string_view prefix, std::string &error) {
  std::error_code filesystem_error;
  const std::filesystem::path root =
      std::filesystem::temp_directory_path(filesystem_error);
  if (filesystem_error) {
    error = "cannot discover temporary directory: " +
            filesystem_error.message();
    return std::nullopt;
  }

  std::string pattern = (root / (std::string(prefix) + "-XXXXXX")).string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char *created = ::mkdtemp(writable.data());
  if (created == nullptr) {
    error = "mkdtemp failed: " + std::string(std::strerror(errno));
    return std::nullopt;
  }
  return std::filesystem::path(created);
}

std::optional<std::string> environment_utf8_native_v1(std::string_view name,
                                                      std::string &error) {
  (void)error;
  const std::string owned(name);
  const char *value = ::getenv(owned.c_str());
  if (value == nullptr) return std::nullopt;
  return std::string(value);
}

std::optional<std::filesystem::path> find_executable_native_v1(
    std::string_view name, std::string &error) {
  const std::filesystem::path requested{std::string(name)};
  if (requested.has_parent_path()) {
    if (!executable_file(requested)) {
      error = "executable is not a regular executable file: " +
              requested.string();
      return std::nullopt;
    }
    return normalize_path_v1(requested, true, error);
  }

  const char *path_value = ::getenv("PATH");
  const std::string search_path =
      path_value == nullptr ? "/bin:/usr/bin" : path_value;
  std::size_t begin = 0;
  while (begin <= search_path.size()) {
    const std::size_t end = search_path.find(':', begin);
    const std::string component = search_path.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    const std::filesystem::path directory =
        component.empty() ? std::filesystem::path(".")
                          : std::filesystem::path(component);
    const std::filesystem::path candidate = directory / requested;
    if (executable_file(candidate)) {
      return normalize_path_v1(candidate, true, error);
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }

  error = "executable was not found on PATH: " + std::string(name);
  return std::nullopt;
}

FileIdentityV1 file_identity_native_v1(const std::filesystem::path &path,
                                       std::string &error) {
  struct stat status {};
  if (::stat(path.c_str(), &status) != 0) {
    error = "stat failed: " + std::string(std::strerror(errno));
    return {};
  }
  FileIdentityV1 identity;
  identity.kind = FileIdentityKindV1::posix_device_inode;
  identity.words = {static_cast<std::uint64_t>(status.st_dev),
                    static_cast<std::uint64_t>(status.st_ino)};
  return identity;
}

std::vector<FileIdentityV1> path_identity_chain_native_v1(
    const std::filesystem::path &path, std::string &error) {
  std::vector<FileIdentityV1> chain;
  std::filesystem::path current = path.root_path();
  for (const std::filesystem::path &component : path.relative_path()) {
    if (component == ".") continue;
    current /= component;
    struct stat status {};
    if (::lstat(current.c_str(), &status) != 0) {
      error = "lstat failed while capturing path identity: " +
              std::string(std::strerror(errno));
      return {};
    }
    FileIdentityV1 identity;
    identity.kind = FileIdentityKindV1::posix_device_inode;
    identity.words = {static_cast<std::uint64_t>(status.st_dev),
                      static_cast<std::uint64_t>(status.st_ino)};
    chain.push_back(identity);
  }
  return chain;
}

}  // namespace detail
}  // namespace matcore::mdslc::support
