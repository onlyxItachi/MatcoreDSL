#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "platform_support_backend.h"

#include <atomic>
#include <cwchar>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace matcore::mdslc::support {

ProcessLaunchBackendV1 process_launch_backend_v1() noexcept {
  return ProcessLaunchBackendV1::windows_create_process_w;
}

namespace detail {
namespace {

class Handle {
 public:
  Handle() = default;
  explicit Handle(HANDLE value) : value_(value) {}
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  Handle(Handle &&other) noexcept : value_(other.release()) {}
  Handle &operator=(Handle &&other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
  }
  ~Handle() { reset(); }

  HANDLE get() const noexcept { return value_; }
  HANDLE release() noexcept {
    const HANDLE result = value_;
    value_ = INVALID_HANDLE_VALUE;
    return result;
  }
  void reset(HANDLE value = INVALID_HANDLE_VALUE) noexcept {
    if (value_ != nullptr && value_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(value_);
    }
    value_ = value;
  }

 private:
  HANDLE value_ = INVALID_HANDLE_VALUE;
};

std::string windows_error(DWORD code) {
  wchar_t *message = nullptr;
  const DWORD count = ::FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, code, 0, reinterpret_cast<wchar_t *>(&message), 0, nullptr);
  std::wstring wide;
  if (count != 0 && message != nullptr) wide.assign(message, count);
  if (message != nullptr) ::LocalFree(message);
  while (!wide.empty() && (wide.back() == L'\r' || wide.back() == L'\n')) {
    wide.pop_back();
  }

  if (wide.empty()) return "Windows error " + std::to_string(code);
  const int bytes = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                          wide.data(),
                                          static_cast<int>(wide.size()),
                                          nullptr, 0, nullptr, nullptr);
  if (bytes <= 0) return "Windows error " + std::to_string(code);
  std::string result(static_cast<std::size_t>(bytes), '\0');
  ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(),
                        static_cast<int>(wide.size()), result.data(), bytes,
                        nullptr, nullptr);
  return result;
}

std::optional<std::wstring> utf8_to_wide(std::string_view text,
                                         std::string &error) {
  if (text.empty()) return std::wstring();
  if (text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    error = "UTF-8 input exceeds Windows conversion limits";
    return std::nullopt;
  }
  const int chars = ::MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
      nullptr, 0);
  if (chars <= 0) {
    error = "invalid UTF-8: " + windows_error(::GetLastError());
    return std::nullopt;
  }
  std::wstring result(static_cast<std::size_t>(chars), L'\0');
  if (::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                            static_cast<int>(text.size()), result.data(),
                            chars) != chars) {
    error = "UTF-8 conversion failed: " + windows_error(::GetLastError());
    return std::nullopt;
  }
  return result;
}

std::optional<std::string> wide_to_utf8(std::wstring_view text,
                                        std::string &error) {
  if (text.empty()) return std::string();
  if (text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    error = "wide-character input exceeds Windows conversion limits";
    return std::nullopt;
  }
  const int bytes = ::WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
      nullptr, 0, nullptr, nullptr);
  if (bytes <= 0) {
    error = "wide-character conversion failed: " +
            windows_error(::GetLastError());
    return std::nullopt;
  }
  std::string result(static_cast<std::size_t>(bytes), '\0');
  if (::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
                            static_cast<int>(text.size()), result.data(), bytes,
                            nullptr, nullptr) != bytes) {
    error = "wide-character conversion failed: " +
            windows_error(::GetLastError());
    return std::nullopt;
  }
  return result;
}

bool executable_file(const std::filesystem::path &path) {
  const DWORD attributes = ::GetFileAttributesW(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES &&
         (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool make_pipe(Handle &read_end, Handle &write_end, std::string &error) {
  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;
  HANDLE raw_read = INVALID_HANDLE_VALUE;
  HANDLE raw_write = INVALID_HANDLE_VALUE;
  if (!::CreatePipe(&raw_read, &raw_write, &security, 0)) {
    error = "CreatePipe failed: " + windows_error(::GetLastError());
    return false;
  }
  read_end.reset(raw_read);
  write_end.reset(raw_write);
  if (!::SetHandleInformation(read_end.get(), HANDLE_FLAG_INHERIT, 0)) {
    error = "cannot make pipe read handle private: " +
            windows_error(::GetLastError());
    return false;
  }
  return true;
}

void read_all(HANDLE handle, std::string &output) {
  char buffer[8192];
  for (;;) {
    DWORD count = 0;
    if (!::ReadFile(handle, buffer, static_cast<DWORD>(sizeof(buffer)), &count,
                    nullptr) ||
        count == 0) {
      return;
    }
    output.append(buffer, count);
  }
}

struct EnvironmentNameLess {
  bool operator()(const std::wstring &left,
                  const std::wstring &right) const noexcept {
    const int result = ::CompareStringOrdinal(
        left.data(), static_cast<int>(left.size()), right.data(),
        static_cast<int>(right.size()), TRUE);
    return result == CSTR_LESS_THAN;
  }
};

std::optional<std::vector<wchar_t>> build_environment_block(
    const std::vector<EnvironmentOverrideV1> &overrides, std::string &error) {
  if (overrides.empty()) return std::nullopt;

  LPWCH raw = ::GetEnvironmentStringsW();
  if (raw == nullptr) {
    error = "GetEnvironmentStringsW failed: " +
            windows_error(::GetLastError());
    return std::nullopt;
  }

  std::vector<std::wstring> special;
  std::map<std::wstring, std::wstring, EnvironmentNameLess> environment;
  for (const wchar_t *entry = raw; *entry != L'\0';
       entry += std::wcslen(entry) + 1) {
    const std::wstring value(entry);
    const std::size_t separator =
        value.find(L'=', value.empty() || value.front() != L'=' ? 0 : 1);
    if (separator == std::wstring::npos) continue;
    if (!value.empty() && value.front() == L'=') {
      special.push_back(value);
    } else {
      environment[value.substr(0, separator)] = value.substr(separator + 1);
    }
  }
  ::FreeEnvironmentStringsW(raw);

  for (const EnvironmentOverrideV1 &entry : overrides) {
    const std::optional<std::wstring> name = utf8_to_wide(entry.name, error);
    if (!name) return std::nullopt;
    if (entry.value) {
      const std::optional<std::wstring> value =
          utf8_to_wide(*entry.value, error);
      if (!value) return std::nullopt;
      environment[*name] = *value;
    } else {
      environment.erase(*name);
    }
  }

  std::vector<wchar_t> block;
  for (const std::wstring &entry : special) {
    block.insert(block.end(), entry.begin(), entry.end());
    block.push_back(L'\0');
  }
  for (const auto &[name, value] : environment) {
    block.insert(block.end(), name.begin(), name.end());
    block.push_back(L'=');
    block.insert(block.end(), value.begin(), value.end());
    block.push_back(L'\0');
  }
  block.push_back(L'\0');
  return block;
}

std::optional<std::filesystem::path> resolve_process_executable(
    const ProcessRequestV1 &request, std::string &error) {
  const std::optional<std::wstring> requested_wide =
      utf8_to_wide(request.argv.front(), error);
  if (!requested_wide) return std::nullopt;
  const std::filesystem::path requested(*requested_wide);
  if (requested.has_parent_path() && requested.is_relative() &&
      !request.working_directory.empty()) {
    const std::filesystem::path candidate =
        request.working_directory / requested;
    if (!executable_file(candidate)) {
      error = "process executable is not a regular file";
      return std::nullopt;
    }
    return normalize_path_v1(candidate, true, error);
  }
  return find_executable_native_v1(request.argv.front(), error);
}

}  // namespace

std::optional<std::filesystem::path> current_executable_path_native_v1(
    std::string &error) {
  std::vector<wchar_t> buffer(260);
  for (;;) {
    ::SetLastError(ERROR_SUCCESS);
    const DWORD count = ::GetModuleFileNameW(
        nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (count == 0) {
      error = "GetModuleFileNameW failed: " +
              windows_error(::GetLastError());
      return std::nullopt;
    }
    if (count < buffer.size() - 1 || ::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
      return std::filesystem::path(std::wstring(buffer.data(), count));
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

  const std::string command_utf8 = build_windows_command_line_v1(request.argv);
  const std::optional<std::wstring> command =
      utf8_to_wide(command_utf8, result.error);
  if (!command) return result;
  if (command->size() >= 32767) {
    result.error = "CreateProcessW command line exceeds 32766 characters";
    return result;
  }
  std::vector<wchar_t> mutable_command(command->begin(), command->end());
  mutable_command.push_back(L'\0');

  Handle stdout_read;
  Handle stdout_write;
  Handle stderr_read;
  Handle stderr_write;
  if (request.capture_stdout &&
      !make_pipe(stdout_read, stdout_write, result.error)) {
    return result;
  }
  if (request.capture_stderr &&
      !make_pipe(stderr_read, stderr_write, result.error)) {
    return result;
  }

  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = ::GetStdHandle(STD_INPUT_HANDLE);
  startup.hStdOutput = request.capture_stdout
      ? stdout_write.get()
      : ::GetStdHandle(STD_OUTPUT_HANDLE);
  startup.hStdError = request.capture_stderr
      ? stderr_write.get()
      : ::GetStdHandle(STD_ERROR_HANDLE);

  std::optional<std::vector<wchar_t>> environment =
      build_environment_block(request.environment, result.error);
  if (!result.error.empty()) return result;

  const wchar_t *working_directory = request.working_directory.empty()
      ? nullptr
      : request.working_directory.c_str();
  PROCESS_INFORMATION process{};
  if (!::CreateProcessW(
          executable->c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
          CREATE_UNICODE_ENVIRONMENT,
          environment ? static_cast<void *>(environment->data()) : nullptr,
          working_directory, &startup, &process)) {
    result.error = "CreateProcessW failed: " +
                   windows_error(::GetLastError());
    return result;
  }

  result.launched = true;
  Handle process_handle(process.hProcess);
  Handle thread_handle(process.hThread);
  stdout_write.reset();
  stderr_write.reset();

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

  if (::WaitForSingleObject(process_handle.get(), INFINITE) != WAIT_OBJECT_0) {
    result.error = "WaitForSingleObject failed: " +
                   windows_error(::GetLastError());
  }
  if (stdout_reader.joinable()) stdout_reader.join();
  if (stderr_reader.joinable()) stderr_reader.join();

  DWORD exit_code = 0;
  if (!::GetExitCodeProcess(process_handle.get(), &exit_code)) {
    if (result.error.empty()) {
      result.error = "GetExitCodeProcess failed: " +
                     windows_error(::GetLastError());
    }
    return result;
  }
  result.exit_code = static_cast<std::int64_t>(exit_code);
  return result;
}

std::optional<std::filesystem::path> create_temp_directory_native_v1(
    std::string_view prefix, std::string &error) {
  const std::optional<std::wstring> wide_prefix = utf8_to_wide(prefix, error);
  if (!wide_prefix) return std::nullopt;

  std::vector<wchar_t> root_buffer(260);
  DWORD count = ::GetTempPathW(static_cast<DWORD>(root_buffer.size()),
                              root_buffer.data());
  if (count == 0) {
    error = "GetTempPathW failed: " + windows_error(::GetLastError());
    return std::nullopt;
  }
  if (count >= root_buffer.size()) {
    root_buffer.resize(static_cast<std::size_t>(count) + 1);
    count = ::GetTempPathW(static_cast<DWORD>(root_buffer.size()),
                          root_buffer.data());
    if (count == 0 || count >= root_buffer.size()) {
      error = "GetTempPathW returned an unstable path length";
      return std::nullopt;
    }
  }
  const std::filesystem::path root(std::wstring(root_buffer.data(), count));

  static std::atomic<std::uint64_t> sequence{0};
  for (std::uint64_t attempt = 0; attempt < 256; ++attempt) {
    const std::uint64_t nonce =
        (::GetTickCount64() << 16U) ^
        static_cast<std::uint64_t>(::GetCurrentProcessId()) ^
        sequence.fetch_add(1, std::memory_order_relaxed) ^ attempt;
    const std::filesystem::path candidate =
        root / (*wide_prefix + L"-" + std::to_wstring(nonce));
    if (::CreateDirectoryW(candidate.c_str(), nullptr)) return candidate;
    if (::GetLastError() != ERROR_ALREADY_EXISTS) {
      error = "CreateDirectoryW failed: " +
              windows_error(::GetLastError());
      return std::nullopt;
    }
  }
  error = "could not create a unique temporary directory";
  return std::nullopt;
}

std::optional<std::string> environment_utf8_native_v1(std::string_view name,
                                                      std::string &error) {
  const std::optional<std::wstring> wide_name = utf8_to_wide(name, error);
  if (!wide_name) return std::nullopt;
  ::SetLastError(ERROR_SUCCESS);
  DWORD count = ::GetEnvironmentVariableW(wide_name->c_str(), nullptr, 0);
  if (count == 0) {
    if (::GetLastError() == ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
    if (::GetLastError() == ERROR_SUCCESS) return std::string();
    error = "GetEnvironmentVariableW failed: " +
            windows_error(::GetLastError());
    return std::nullopt;
  }
  std::vector<wchar_t> value(count);
  count = ::GetEnvironmentVariableW(wide_name->c_str(), value.data(),
                                    static_cast<DWORD>(value.size()));
  if (count >= value.size()) {
    error = "environment variable changed while it was read";
    return std::nullopt;
  }
  return wide_to_utf8(std::wstring_view(value.data(), count), error);
}

std::optional<std::filesystem::path> find_executable_native_v1(
    std::string_view name, std::string &error) {
  const std::optional<std::wstring> wide_name = utf8_to_wide(name, error);
  if (!wide_name) return std::nullopt;

  const bool explicit_path = wide_name->find(L'\\') != std::wstring::npos ||
                             wide_name->find(L'/') != std::wstring::npos ||
                             wide_name->find(L':') != std::wstring::npos;
  if (explicit_path) {
    const std::filesystem::path requested(*wide_name);
    if (!executable_file(requested)) {
      error = "executable is not a regular file";
      return std::nullopt;
    }
    return normalize_path_v1(requested, true, error);
  }

  DWORD capacity = ::SearchPathW(nullptr, wide_name->c_str(), L".exe", 0,
                                 nullptr, nullptr);
  if (capacity == 0) {
    error = "executable was not found on PATH: " + std::string(name);
    return std::nullopt;
  }
  std::vector<wchar_t> buffer(static_cast<std::size_t>(capacity) + 1);
  const DWORD count = ::SearchPathW(nullptr, wide_name->c_str(), L".exe",
                                    static_cast<DWORD>(buffer.size()),
                                    buffer.data(), nullptr);
  if (count == 0 || count >= buffer.size()) {
    error = "SearchPathW failed: " + windows_error(::GetLastError());
    return std::nullopt;
  }
  const std::filesystem::path found(std::wstring(buffer.data(), count));
  if (!executable_file(found)) {
    error = "PATH result is not a regular executable file";
    return std::nullopt;
  }
  return normalize_path_v1(found, true, error);
}

FileIdentityV1 file_identity_native_v1(const std::filesystem::path &path,
                                       std::string &error) {
  Handle file(::CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                            FILE_SHARE_READ | FILE_SHARE_WRITE |
                                FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING,
                            FILE_FLAG_BACKUP_SEMANTICS, nullptr));
  if (file.get() == INVALID_HANDLE_VALUE) {
    error = "CreateFileW failed: " + windows_error(::GetLastError());
    return {};
  }
  BY_HANDLE_FILE_INFORMATION info{};
  if (!::GetFileInformationByHandle(file.get(), &info)) {
    error = "GetFileInformationByHandle failed: " +
            windows_error(::GetLastError());
    return {};
  }

  FileIdentityV1 identity;
  identity.kind = FileIdentityKindV1::windows_volume_file_id;
  identity.words = {
      static_cast<std::uint64_t>(info.dwVolumeSerialNumber),
      (static_cast<std::uint64_t>(info.nFileIndexHigh) << 32U) |
          static_cast<std::uint64_t>(info.nFileIndexLow)};
  return identity;
}

}  // namespace detail
}  // namespace matcore::mdslc::support
