#include "platform_support.h"

#include "platform_support_backend.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <system_error>
#include <utility>

namespace matcore::mdslc::support {
namespace {

bool contains_nul(std::string_view text) {
  return text.find('\0') != std::string_view::npos;
}

bool decode_utf8_code_point(std::string_view value, std::size_t &offset,
                            std::uint32_t &code_point) {
  const auto byte = [&](std::size_t index) {
    return static_cast<unsigned char>(value[index]);
  };
  const unsigned char first = byte(offset);
  if (first <= 0x7f) {
    code_point = first;
    ++offset;
    return true;
  }

  std::size_t length = 0;
  std::uint32_t minimum = 0;
  if (first >= 0xc2 && first <= 0xdf) {
    length = 2;
    minimum = 0x80;
    code_point = first & 0x1f;
  } else if (first >= 0xe0 && first <= 0xef) {
    length = 3;
    minimum = 0x800;
    code_point = first & 0x0f;
  } else if (first >= 0xf0 && first <= 0xf4) {
    length = 4;
    minimum = 0x10000;
    code_point = first & 0x07;
  } else {
    return false;
  }
  if (offset + length > value.size()) return false;
  for (std::size_t index = 1; index < length; ++index) {
    const unsigned char continuation = byte(offset + index);
    if ((continuation & 0xc0) != 0x80) return false;
    code_point = (code_point << 6U) | (continuation & 0x3f);
  }
  if (code_point < minimum || code_point > 0x10ffff ||
      (code_point >= 0xd800 && code_point <= 0xdfff)) {
    return false;
  }
  offset += length;
  return true;
}

bool valid_utf8(std::string_view value) {
  std::size_t offset = 0;
  while (offset < value.size()) {
    std::uint32_t code_point = 0;
    if (!decode_utf8_code_point(value, offset, code_point)) return false;
  }
  return true;
}

void append_utf8(std::uint32_t code_point, std::string &output) {
  if (code_point <= 0x7f) {
    output.push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7ff) {
    output.push_back(static_cast<char>(0xc0 | (code_point >> 6U)));
    output.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
  } else if (code_point <= 0xffff) {
    output.push_back(static_cast<char>(0xe0 | (code_point >> 12U)));
    output.push_back(
        static_cast<char>(0x80 | ((code_point >> 6U) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
  } else {
    output.push_back(static_cast<char>(0xf0 | (code_point >> 18U)));
    output.push_back(
        static_cast<char>(0x80 | ((code_point >> 12U) & 0x3f)));
    output.push_back(
        static_cast<char>(0x80 | ((code_point >> 6U) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
  }
}

bool response_argument_valid(std::string_view argument, std::string &error) {
  if (contains_nul(argument)) {
    error = "response-file arguments cannot contain NUL bytes";
    return false;
  }
  return true;
}

constexpr std::string_view kArgumentFileMagic = "MDSLC-ARGV-V1\r\n";
constexpr std::uint64_t kMaximumArgumentFileBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaximumArgumentCount = 1024ULL * 1024ULL;

void append_u64(std::uint64_t value, std::string &output) {
  for (unsigned shift = 0; shift != 64; shift += 8) {
    output.push_back(static_cast<char>((value >> shift) & 0xffU));
  }
}

bool take_u64(std::string_view input, std::size_t &offset,
              std::uint64_t &value) {
  if (input.size() - std::min(input.size(), offset) < 8) return false;
  value = 0;
  for (unsigned shift = 0; shift != 64; shift += 8) {
    value |= static_cast<std::uint64_t>(
                 static_cast<unsigned char>(input[offset++]))
             << shift;
  }
  return true;
}

std::string quote_gnu_response_argument(std::string_view argument) {
  if (argument.empty()) return "\"\"";

  const bool needs_quotes = std::any_of(
      argument.begin(), argument.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0 || ch == '\\' || ch == '"' || ch == '\'';
      });
  if (!needs_quotes) return std::string(argument);

  std::string output;
  output.reserve(argument.size() + 2);
  output.push_back('"');
  for (const char ch : argument) {
    if (ch == '\\' || ch == '"') output.push_back('\\');
    output.push_back(ch);
  }
  output.push_back('"');
  return output;
}

}  // namespace

TempDirectoryV1::TempDirectoryV1(TempDirectoryV1 &&other) noexcept
    : path_(std::move(other.path_)) {
  other.path_.clear();
}

TempDirectoryV1 &TempDirectoryV1::operator=(TempDirectoryV1 &&other) noexcept {
  if (this == &other) return *this;
  remove();
  path_ = std::move(other.path_);
  other.path_.clear();
  return *this;
}

TempDirectoryV1::~TempDirectoryV1() { remove(); }

std::filesystem::path TempDirectoryV1::release() noexcept {
  std::filesystem::path result = std::move(path_);
  path_.clear();
  return result;
}

void TempDirectoryV1::remove() noexcept {
  if (path_.empty()) return;
  std::error_code ignored;
  std::filesystem::remove_all(path_, ignored);
  path_.clear();
}

std::optional<std::filesystem::path> current_executable_path_v1(
    std::string &error) {
  error.clear();
  return detail::current_executable_path_native_v1(error);
}

ProcessResultV1 run_process_v1(const ProcessRequestV1 &request) {
  ProcessResultV1 result;
  if (request.version != kPlatformSupportApiVersionV1) {
    result.error = "unsupported process request version";
    return result;
  }
  if (request.argv.empty() || request.argv.front().empty()) {
    result.error = "process argv must contain a non-empty executable";
    return result;
  }
  for (const std::string &argument : request.argv) {
    if (contains_nul(argument)) {
      result.error = "process arguments cannot contain NUL bytes";
      return result;
    }
  }
  for (const EnvironmentOverrideV1 &entry : request.environment) {
    if (entry.name.empty() || contains_nul(entry.name) ||
        entry.name.find('=') != std::string::npos ||
        (entry.value && contains_nul(*entry.value))) {
      result.error = "child environment override is malformed";
      return result;
    }
  }
  return detail::run_process_native_v1(request);
}

std::optional<TempDirectoryV1> create_temp_directory_v1(
    std::string_view prefix, std::string &error) {
  error.clear();
  if (prefix.empty()) {
    error = "temporary-directory prefix must not be empty";
    return std::nullopt;
  }
  if (contains_nul(prefix) || prefix.find('/') != std::string_view::npos ||
      prefix.find('\\') != std::string_view::npos) {
    error = "temporary-directory prefix must be a single path component";
    return std::nullopt;
  }
  std::optional<std::filesystem::path> path =
      detail::create_temp_directory_native_v1(prefix, error);
  if (!path) return std::nullopt;
  return TempDirectoryV1(std::move(*path));
}

std::optional<std::string> environment_utf8_v1(std::string_view name,
                                               std::string &error) {
  error.clear();
  if (name.empty() || contains_nul(name) ||
      name.find('=') != std::string_view::npos) {
    error = "environment variable name is malformed";
    return std::nullopt;
  }
  return detail::environment_utf8_native_v1(name, error);
}

std::optional<std::filesystem::path> find_executable_v1(
    std::string_view name, std::string &error) {
  error.clear();
  if (name.empty() || contains_nul(name)) {
    error = "executable name is empty or contains NUL";
    return std::nullopt;
  }
  return detail::find_executable_native_v1(name, error);
}

std::optional<std::filesystem::path> path_from_utf8_v1(
    std::string_view value, std::string &error) {
  error.clear();
  if (contains_nul(value)) {
    error = "UTF-8 path contains NUL";
    return std::nullopt;
  }
  if (!valid_utf8(value)) {
    error = "path is not well-formed UTF-8";
    return std::nullopt;
  }
  try {
    const std::u8string encoded(
        reinterpret_cast<const char8_t *>(value.data()), value.size());
    return std::filesystem::path(encoded);
  } catch (const std::filesystem::filesystem_error &exception) {
    error = "cannot convert UTF-8 path: " + std::string(exception.what());
    return std::nullopt;
  }
}

std::optional<std::string> path_to_utf8_v1(
    const std::filesystem::path &path, std::string &error) {
  error.clear();
  try {
    const std::u8string encoded = path.u8string();
    const std::string result(reinterpret_cast<const char *>(encoded.data()),
                             encoded.size());
    if (!valid_utf8(result)) {
      error = "filesystem path cannot be represented as well-formed UTF-8";
      return std::nullopt;
    }
    return result;
  } catch (const std::filesystem::filesystem_error &exception) {
    error = "cannot convert filesystem path to UTF-8: " +
            std::string(exception.what());
    return std::nullopt;
  }
}

std::optional<std::vector<std::string>> wide_arguments_to_utf8_v1(
    int argc, wchar_t *const *argv, std::string &error) {
  error.clear();
  if (argc < 0 || (argc != 0 && argv == nullptr)) {
    error = "wide process argument vector is malformed";
    return std::nullopt;
  }
  std::vector<std::string> result;
  result.reserve(static_cast<std::size_t>(argc));
  for (int argument_index = 0; argument_index < argc; ++argument_index) {
    if (argv[argument_index] == nullptr) {
      error = "wide process argument is null";
      return std::nullopt;
    }
    std::string encoded;
    for (std::size_t index = 0; argv[argument_index][index] != L'\0';
         ++index) {
      std::uint32_t code_point =
          static_cast<std::uint32_t>(argv[argument_index][index]);
      if constexpr (sizeof(wchar_t) == 2) {
        if (code_point >= 0xd800 && code_point <= 0xdbff) {
          const std::uint32_t trailing = static_cast<std::uint32_t>(
              argv[argument_index][index + 1]);
          if (trailing < 0xdc00 || trailing > 0xdfff) {
            error = "process argument contains an unpaired UTF-16 surrogate";
            return std::nullopt;
          }
          code_point = 0x10000 + ((code_point - 0xd800) << 10U) +
                       (trailing - 0xdc00);
          ++index;
        } else if (code_point >= 0xdc00 && code_point <= 0xdfff) {
          error = "process argument contains an unpaired UTF-16 surrogate";
          return std::nullopt;
        }
      } else {
        if (code_point > 0x10ffff ||
            (code_point >= 0xd800 && code_point <= 0xdfff)) {
          error = "process argument contains an invalid Unicode code point";
          return std::nullopt;
        }
      }
      append_utf8(code_point, encoded);
    }
    result.emplace_back(std::move(encoded));
  }
  return result;
}

std::filesystem::path normalize_path_v1(const std::filesystem::path &path,
                                        bool resolve_existing,
                                        std::string &error) {
  error.clear();
  if (path.empty()) {
    error = "cannot normalize an empty path";
    return {};
  }

  std::error_code ec;
  std::filesystem::path absolute = std::filesystem::absolute(path, ec);
  if (ec) {
    error = "cannot make path absolute: " + ec.message();
    return {};
  }
  absolute = absolute.lexically_normal();
  if (!resolve_existing) return absolute;

  std::filesystem::path resolved =
      std::filesystem::weakly_canonical(absolute, ec);
  if (ec) {
    error = "cannot resolve path: " + ec.message();
    return {};
  }
  return resolved;
}

FileSnapshotV1 capture_file_snapshot_v1(const std::filesystem::path &path,
                                        std::string &error) {
  FileSnapshotV1 snapshot;
  error.clear();
  std::error_code ec;
  snapshot.normalized_path = std::filesystem::absolute(path, ec);
  if (ec) {
    error = "cannot make snapshot path absolute: " + ec.message();
    return snapshot;
  }

  snapshot.exists = std::filesystem::exists(snapshot.normalized_path, ec);
  if (ec) {
    error = "cannot inspect path existence: " + ec.message();
    return snapshot;
  }
  if (!snapshot.exists) return snapshot;

  snapshot.regular_file =
      std::filesystem::is_regular_file(snapshot.normalized_path, ec);
  if (ec) {
    error = "cannot inspect file type: " + ec.message();
    return snapshot;
  }

  if (snapshot.regular_file) {
    const std::uintmax_t size =
        std::filesystem::file_size(snapshot.normalized_path, ec);
    if (ec) {
      error = "cannot inspect file size: " + ec.message();
      return snapshot;
    }
    if (size > std::numeric_limits<std::uint64_t>::max()) {
      error = "file size exceeds snapshot representation";
      return snapshot;
    }
    snapshot.size_bytes = static_cast<std::uint64_t>(size);
  }

  const std::filesystem::file_time_type write_time =
      std::filesystem::last_write_time(snapshot.normalized_path, ec);
  if (ec) {
    error = "cannot inspect file modification time: " + ec.message();
    return snapshot;
  }
  const auto ticks = write_time.time_since_epoch().count();
  if (ticks < std::numeric_limits<std::int64_t>::min() ||
      ticks > std::numeric_limits<std::int64_t>::max()) {
    error = "file modification time exceeds snapshot representation";
    return snapshot;
  }
  snapshot.last_write_time_ticks = static_cast<std::int64_t>(ticks);
  snapshot.identity =
      detail::file_identity_native_v1(snapshot.normalized_path, error);
  if (!error.empty()) return snapshot;
  snapshot.path_identity_chain =
      detail::path_identity_chain_native_v1(snapshot.normalized_path, error);
  return snapshot;
}

bool same_file_identity_v1(const FileIdentityV1 &left,
                           const FileIdentityV1 &right) noexcept {
  return left && right && left.kind == right.kind && left.words == right.words;
}

bool replace_file_atomically_v1(const std::filesystem::path &temporary,
                                const std::filesystem::path &destination,
                                std::string &error) {
  error.clear();
  if (temporary.empty() || destination.empty()) {
    error = "atomic file replacement requires two non-empty paths";
    return false;
  }
  return detail::replace_file_atomically_native_v1(temporary, destination,
                                                    error);
}

std::string quote_windows_command_line_argument_v1(
    std::string_view argument) {
  const bool requires_quotes = argument.empty() ||
      std::any_of(argument.begin(), argument.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0 || ch == '"';
      });
  if (!requires_quotes) return std::string(argument);

  std::string output;
  output.reserve(argument.size() + 2);
  output.push_back('"');
  std::size_t backslashes = 0;
  for (const char ch : argument) {
    if (ch == '\\') {
      ++backslashes;
      continue;
    }
    if (ch == '"') {
      output.append(backslashes * 2 + 1, '\\');
      output.push_back('"');
    } else {
      output.append(backslashes, '\\');
      output.push_back(ch);
    }
    backslashes = 0;
  }
  output.append(backslashes * 2, '\\');
  output.push_back('"');
  return output;
}

std::string build_windows_command_line_v1(
    const std::vector<std::string> &argv) {
  std::string output;
  for (std::size_t index = 0; index < argv.size(); ++index) {
    if (index != 0) output.push_back(' ');
    output += quote_windows_command_line_argument_v1(argv[index]);
  }
  return output;
}

std::string encode_response_file_utf8_v1(
    const std::vector<std::string> &arguments, ResponseFileSyntaxV1 syntax,
    std::string &error) {
  error.clear();
  if (syntax != ResponseFileSyntaxV1::gnu &&
      syntax != ResponseFileSyntaxV1::windows) {
    error = "response-file syntax is unsupported";
    return {};
  }

  std::string output;
  for (const std::string &argument : arguments) {
    if (!response_argument_valid(argument, error)) return {};
    if (syntax == ResponseFileSyntaxV1::windows) {
      output += quote_windows_command_line_argument_v1(argument);
    } else {
      output += quote_gnu_response_argument(argument);
    }
    output.push_back('\n');
  }
  return output;
}

bool write_response_file_utf8_v1(
    const std::filesystem::path &path,
    const std::vector<std::string> &arguments, ResponseFileSyntaxV1 syntax,
    std::string &error) {
  const std::string contents =
      encode_response_file_utf8_v1(arguments, syntax, error);
  if (!error.empty()) return false;

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    error = "cannot open response file for writing";
    return false;
  }
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  output.close();
  if (!output) {
    error = "cannot write response file";
    return false;
  }
  return true;
}

bool write_argument_file_v1(const std::filesystem::path &path,
                            const std::vector<std::string> &arguments,
                            std::string &error) {
  error.clear();
  if (arguments.size() > kMaximumArgumentCount) {
    error = "argument-file count exceeds the v1 limit";
    return false;
  }
  std::string contents(kArgumentFileMagic);
  append_u64(static_cast<std::uint64_t>(arguments.size()), contents);
  for (const std::string &argument : arguments) {
    if (contains_nul(argument) || !valid_utf8(argument)) {
      error = "argument-file values must be NUL-free well-formed UTF-8";
      return false;
    }
    if (contents.size() > kMaximumArgumentFileBytes - 8 ||
        argument.size() >
            kMaximumArgumentFileBytes - 8 - contents.size()) {
      error = "argument-file payload exceeds the v1 byte limit";
      return false;
    }
    append_u64(static_cast<std::uint64_t>(argument.size()), contents);
    contents.append(argument);
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    error = "cannot open argument file for writing";
    return false;
  }
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  output.close();
  if (!output) {
    error = "cannot write argument file";
    return false;
  }
  return true;
}

std::optional<std::vector<std::string>> read_argument_file_v1(
    const std::filesystem::path &path, std::string &error) {
  error.clear();
  const FileSnapshotV1 before = capture_file_snapshot_v1(path, error);
  if (!error.empty() || !before.exists || !before.regular_file ||
      !before.identity) {
    if (error.empty()) error = "argument file is not a regular file";
    return std::nullopt;
  }
  if (before.size_bytes > kMaximumArgumentFileBytes) {
    error = "argument-file payload exceeds the v1 byte limit";
    return std::nullopt;
  }
  std::ifstream input(before.normalized_path, std::ios::binary);
  if (!input) {
    error = "cannot open argument file";
    return std::nullopt;
  }
  const std::string contents((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
  if (!input.eof() && input.fail()) {
    error = "cannot read argument file";
    return std::nullopt;
  }
  const FileSnapshotV1 after = capture_file_snapshot_v1(path, error);
  if (!error.empty() || before.identity != after.identity ||
      before.path_identity_chain != after.path_identity_chain ||
      before.size_bytes != after.size_bytes ||
      before.last_write_time_ticks != after.last_write_time_ticks ||
      before.normalized_path != after.normalized_path) {
    if (error.empty()) error = "argument file changed while it was read";
    return std::nullopt;
  }
  if (!std::string_view(contents).starts_with(kArgumentFileMagic)) {
    error = "argument file has an invalid v1 signature";
    return std::nullopt;
  }
  std::size_t offset = kArgumentFileMagic.size();
  std::uint64_t count = 0;
  if (!take_u64(contents, offset, count) || count > kMaximumArgumentCount) {
    error = "argument file has an invalid count";
    return std::nullopt;
  }
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    std::uint64_t size = 0;
    if (!take_u64(contents, offset, size) ||
        size > contents.size() - std::min(contents.size(), offset)) {
      error = "argument file has a truncated value";
      return std::nullopt;
    }
    const std::string argument = contents.substr(
        offset, static_cast<std::size_t>(size));
    offset += static_cast<std::size_t>(size);
    if (contains_nul(argument) || !valid_utf8(argument)) {
      error = "argument file contains malformed UTF-8 or NUL";
      return std::nullopt;
    }
    arguments.push_back(argument);
  }
  if (offset != contents.size()) {
    error = "argument file contains trailing bytes";
    return std::nullopt;
  }
  return arguments;
}

}  // namespace matcore::mdslc::support
