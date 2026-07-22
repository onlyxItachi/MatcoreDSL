#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "platform_support_backend.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
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

std::string windows_error(DWORD code);

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

class ProcessAttributeList {
 public:
  ProcessAttributeList() = default;
  ProcessAttributeList(const ProcessAttributeList &) = delete;
  ProcessAttributeList &operator=(const ProcessAttributeList &) = delete;
  ~ProcessAttributeList() {
    if (list_ != nullptr) ::DeleteProcThreadAttributeList(list_);
  }

  bool initialize(std::string &error) {
    SIZE_T bytes = 0;
    ::InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    if (bytes == 0) {
      error = "cannot size process attribute list: " +
              windows_error(::GetLastError());
      return false;
    }
    const std::size_t words =
        (static_cast<std::size_t>(bytes) + sizeof(std::uintptr_t) - 1) /
        sizeof(std::uintptr_t);
    storage_.resize(words);
    list_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage_.data());
    if (!::InitializeProcThreadAttributeList(list_, 1, 0, &bytes)) {
      error = "cannot initialize process attribute list: " +
              windows_error(::GetLastError());
      list_ = nullptr;
      return false;
    }
    return true;
  }

  LPPROC_THREAD_ATTRIBUTE_LIST get() const noexcept { return list_; }

 private:
  std::vector<std::uintptr_t> storage_;
  LPPROC_THREAD_ATTRIBUTE_LIST list_ = nullptr;
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

bool make_inheritable_standard_handle(DWORD standard_handle,
                                      DWORD null_access, Handle &owned,
                                      std::string &error) {
  const HANDLE source = ::GetStdHandle(standard_handle);
  HANDLE duplicate = INVALID_HANDLE_VALUE;
  if (source != nullptr && source != INVALID_HANDLE_VALUE) {
    if (!::DuplicateHandle(::GetCurrentProcess(), source,
                           ::GetCurrentProcess(), &duplicate, 0, TRUE,
                           DUPLICATE_SAME_ACCESS)) {
      error = "cannot duplicate standard handle: " +
              windows_error(::GetLastError());
      return false;
    }
  } else {
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    duplicate = ::CreateFileW(L"NUL", null_access,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (duplicate == INVALID_HANDLE_VALUE) {
      error = "cannot create fallback standard handle: " +
              windows_error(::GetLastError());
      return false;
    }
  }
  owned.reset(duplicate);
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

  Handle child_stdin;
  Handle child_stdout;
  Handle child_stderr;
  if (!make_inheritable_standard_handle(STD_INPUT_HANDLE, GENERIC_READ,
                                        child_stdin, result.error) ||
      (!request.capture_stdout &&
       !make_inheritable_standard_handle(STD_OUTPUT_HANDLE, GENERIC_WRITE,
                                         child_stdout, result.error)) ||
      (!request.capture_stderr &&
       !make_inheritable_standard_handle(STD_ERROR_HANDLE, GENERIC_WRITE,
                                         child_stderr, result.error))) {
    return result;
  }

  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = child_stdin.get();
  startup.StartupInfo.hStdOutput = request.capture_stdout
      ? stdout_write.get()
      : child_stdout.get();
  startup.StartupInfo.hStdError = request.capture_stderr
      ? stderr_write.get()
      : child_stderr.get();

  std::vector<HANDLE> inherited_handles{startup.StartupInfo.hStdInput,
                                        startup.StartupInfo.hStdOutput,
                                        startup.StartupInfo.hStdError};
  std::sort(inherited_handles.begin(), inherited_handles.end());
  inherited_handles.erase(
      std::unique(inherited_handles.begin(), inherited_handles.end()),
      inherited_handles.end());
  ProcessAttributeList attributes;
  if (!attributes.initialize(result.error)) return result;
  if (!::UpdateProcThreadAttribute(
          attributes.get(), 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
          inherited_handles.data(), inherited_handles.size() * sizeof(HANDLE),
          nullptr, nullptr)) {
    result.error = "cannot restrict inherited process handles: " +
                   windows_error(::GetLastError());
    return result;
  }
  startup.lpAttributeList = attributes.get();

  std::optional<std::vector<wchar_t>> environment =
      build_environment_block(request.environment, result.error);
  if (!result.error.empty()) return result;

  const wchar_t *working_directory = request.working_directory.empty()
      ? nullptr
      : request.working_directory.c_str();
  PROCESS_INFORMATION process{};
  if (!::CreateProcessW(
          executable->c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
          CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
          environment ? static_cast<void *>(environment->data()) : nullptr,
          working_directory, &startup.StartupInfo, &process)) {
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

bool path_names_equal_native_v1(const std::filesystem::path &left,
                                const std::filesystem::path &right,
                                std::string &error) {
  error.clear();
  struct ProspectivePath {
    std::filesystem::path existing_prefix;
    std::vector<std::wstring> missing_components;
  };
  const auto split_at_existing_prefix = [&](std::filesystem::path path) {
    ProspectivePath split;
    std::error_code status_error;
    while (!path.empty() && !std::filesystem::exists(path, status_error)) {
      if (status_error) {
        error = "cannot inspect prospective path prefix: " +
                status_error.message();
        return split;
      }
      std::wstring component = path.filename().native();
      // Win32 normalizes trailing spaces and periods in ordinary path
      // components.  Model that before comparing not-yet-created outputs so
      // `result.lib`, `result.lib.`, and `result.lib ` cannot bypass alias
      // guards.
      while (!component.empty() &&
             (component.back() == L'.' || component.back() == L' ')) {
        component.pop_back();
      }
      split.missing_components.push_back(std::move(component));
      const std::filesystem::path parent = path.parent_path();
      if (parent == path) break;
      path = parent;
      status_error.clear();
    }
    split.existing_prefix = path;
    std::reverse(split.missing_components.begin(),
                 split.missing_components.end());
    return split;
  };

  const ProspectivePath left_split = split_at_existing_prefix(left);
  if (!error.empty()) return false;
  const ProspectivePath right_split = split_at_existing_prefix(right);
  if (!error.empty()) return false;
  if (left_split.missing_components.size() !=
      right_split.missing_components.size()) {
    return false;
  }

  std::error_code equivalent_error;
  const bool same_prefix = std::filesystem::equivalent(
      left_split.existing_prefix, right_split.existing_prefix,
      equivalent_error);
  if (equivalent_error) {
    error = "cannot authenticate prospective path parent identity: " +
            equivalent_error.message();
    return false;
  }
  if (!same_prefix) return false;

  for (std::size_t index = 0;
       index < left_split.missing_components.size(); ++index) {
    const std::wstring &left_component = left_split.missing_components[index];
    const std::wstring &right_component = right_split.missing_components[index];
    if (left_component.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        right_component.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      error = "normalized path component is too long for ordinal comparison";
      return false;
    }
    const int comparison = ::CompareStringOrdinal(
        left_component.data(), static_cast<int>(left_component.size()),
        right_component.data(), static_cast<int>(right_component.size()),
        TRUE);
    if (comparison == 0) {
      error = "CompareStringOrdinal failed: " +
              windows_error(::GetLastError());
      return false;
    }
    if (comparison != CSTR_EQUAL) return false;
  }
  return true;
}

bool prospective_output_path_supported_native_v1(
    const std::filesystem::path &path, std::string &error) {
  error.clear();
  const std::wstring name = path.filename().native();
  if (name.empty()) {
    error = "output path must name a file";
    return false;
  }
  if (name.back() == L'.' || name.back() == L' ') {
    error = "Windows output names cannot end in a period or space";
    return false;
  }
  if (name.find_first_of(L"<>:\"|?*") != std::wstring::npos ||
      std::any_of(name.begin(), name.end(),
                  [](wchar_t value) { return value >= 0 && value < 32; })) {
    error = "Windows output name contains a reserved character";
    return false;
  }
  const std::size_t extension = name.find(L'.');
  const std::wstring_view stem(name.data(), extension == std::wstring::npos
                                                ? name.size()
                                                : extension);
  const auto reserved = [&](std::wstring_view candidate) {
    if (stem.size() != candidate.size()) return false;
    const int result = ::CompareStringOrdinal(
        stem.data(), static_cast<int>(stem.size()), candidate.data(),
        static_cast<int>(candidate.size()), TRUE);
    return result == CSTR_EQUAL;
  };
  if (reserved(L"CON") || reserved(L"PRN") || reserved(L"AUX") ||
      reserved(L"NUL") || reserved(L"CLOCK$") || reserved(L"CONIN$") ||
      reserved(L"CONOUT$")) {
    error = "Windows output name uses a reserved DOS device basename";
    return false;
  }
  const bool com_device =
      stem.size() == 4 && (stem[0] == L'C' || stem[0] == L'c') &&
      (stem[1] == L'O' || stem[1] == L'o') &&
      (stem[2] == L'M' || stem[2] == L'm');
  const bool lpt_device =
      stem.size() == 4 && (stem[0] == L'L' || stem[0] == L'l') &&
      (stem[1] == L'P' || stem[1] == L'p') &&
      (stem[2] == L'T' || stem[2] == L't');
  // Win32 also recognizes the ISO-8859-1 superscript digits 1, 2, and 3 in
  // COM/LPT device names.  Reject them before any compiler or archiver can
  // open the output path, rather than relying on the post-process regular-file
  // check to notice that no file was created.
  const bool reserved_device_digit =
      stem.size() == 4 &&
      ((stem[3] >= L'1' && stem[3] <= L'9') || stem[3] == L'\u00b9' ||
       stem[3] == L'\u00b2' || stem[3] == L'\u00b3');
  if ((com_device || lpt_device) && reserved_device_digit) {
    error = "Windows output name uses a reserved DOS device basename";
    return false;
  }
  return true;
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

  const DWORD required = ::GetEnvironmentVariableW(L"PATH", nullptr, 0);
  if (required == 0) {
    error = "PATH is unavailable for executable discovery";
    return std::nullopt;
  }
  std::vector<wchar_t> buffer(required);
  const DWORD count = ::GetEnvironmentVariableW(
      L"PATH", buffer.data(), static_cast<DWORD>(buffer.size()));
  if (count == 0 || count >= buffer.size()) {
    error = "cannot read PATH for executable discovery: " +
            windows_error(::GetLastError());
    return std::nullopt;
  }
  const std::wstring path_value(buffer.data(), count);
  std::filesystem::path filename(*wide_name);
  if (!filename.has_extension()) filename += L".exe";
  std::size_t begin = 0;
  while (begin <= path_value.size()) {
    const std::size_t end = path_value.find(L';', begin);
    std::wstring component = path_value.substr(
        begin, end == std::wstring::npos ? std::wstring::npos : end - begin);
    if (component.size() >= 2 && component.front() == L'"' &&
        component.back() == L'"') {
      component = component.substr(1, component.size() - 2);
    }
    // Empty PATH entries are deliberately skipped: unlike SearchPathW, this
    // contract never consults the current directory or App Paths.
    if (!component.empty()) {
      const std::filesystem::path candidate =
          std::filesystem::path(component) / filename;
      if (executable_file(candidate)) {
        return normalize_path_v1(candidate, true, error);
      }
    }
    if (end == std::wstring::npos) break;
    begin = end + 1;
  }

  error = "executable was not found in explicit PATH directories: " +
          std::string(name);
  return std::nullopt;
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

std::vector<PathComponentSnapshotV1> path_identity_chain_native_v1(
    const std::filesystem::path &path, std::string &error) {
  std::vector<PathComponentSnapshotV1> chain;
  std::filesystem::path current = path.root_path();
  for (const std::filesystem::path &component : path.relative_path()) {
    if (component == L".") continue;
    current /= component;
    PathComponentSnapshotV1 snapshot;
    snapshot.identity = file_identity_native_v1(current, error);
    if (!error.empty()) return {};
    WIN32_FILE_ATTRIBUTE_DATA attributes{};
    if (!::GetFileAttributesExW(current.c_str(), GetFileExInfoStandard,
                                &attributes)) {
      error = "GetFileAttributesExW failed while capturing path identity: " +
              windows_error(::GetLastError());
      return {};
    }
    const DWORD stable_type_attributes =
        attributes.dwFileAttributes &
        (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT);
    snapshot.metadata_words[0] =
        static_cast<std::uint64_t>(stable_type_attributes);
    // Directory write times legitimately change while generated artifacts are
    // emitted.  Preserve richer metadata only for reparse points, whose
    // retargeting must invalidate an authenticated path traversal.
    if ((attributes.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
      snapshot.metadata_words[1] =
          (static_cast<std::uint64_t>(attributes.nFileSizeHigh) << 32U) |
          attributes.nFileSizeLow;
      snapshot.metadata_words[2] =
          (static_cast<std::uint64_t>(
               attributes.ftLastWriteTime.dwHighDateTime)
           << 32U) |
          attributes.ftLastWriteTime.dwLowDateTime;
    }
    chain.push_back(std::move(snapshot));
  }
  return chain;
}

bool replace_file_atomically_native_v1(
    const std::filesystem::path &temporary,
    const std::filesystem::path &destination, std::string &error) {
  if (!::MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    error = "MoveFileExW failed while publishing file: " +
            windows_error(::GetLastError());
    return false;
  }
  return true;
}

}  // namespace detail
}  // namespace matcore::mdslc::support
