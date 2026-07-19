#include "mdslc_config.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

enum class FrontendMode { Native, AstJsonBootstrap };

struct WrapperArguments {
  bool verbose = false;
  bool save_temps = false;
  bool cpu_pipeline = false;
  bool frontend_was_explicit = false;
  FrontendMode frontend = FrontendMode::Native;
  std::optional<fs::path> tool_prefix_for_testing;
  std::vector<std::string> compiler_arguments;
};

struct CpuInvocation {
  bool compile_only = false;
  bool has_link_only_arguments = false;
  std::string input;
  std::string output;
  std::string dependency_mode;
  std::string dependency_file;
  std::vector<std::string> compile_arguments;
  std::vector<std::string> link_context_arguments;
  std::vector<std::string> link_arguments;
};

struct ToolLayout {
  fs::path extractor;
  fs::path include_directory;
  fs::path runtime_directory;
  fs::path runtime_library;
};

struct GeneratedArtifacts {
  fs::path host_source;
  fs::path host_overlay;
  fs::path ir;
  fs::path sites_header;
  fs::path stubs_source;
  fs::path backend_source;
  fs::path host_object;
  fs::path stubs_object;
  fs::path backend_object;
};

struct SourceSnapshot {
  dev_t device = 0;
  ino_t inode = 0;
  off_t size = 0;
  timespec modified{};
  timespec changed{};
  std::string contents;
};

struct PathComponentSnapshot {
  fs::path path;
  dev_t device = 0;
  ino_t inode = 0;
  mode_t type = 0;
  off_t size = 0;
  timespec modified{};
  timespec changed{};
  std::string symlink_target;
};

struct DependencySnapshot {
  fs::path path;
  SourceSnapshot snapshot;
  std::vector<PathComponentSnapshot> path_identity;
};

class ScopedDirectoryCleanup {
public:
  ScopedDirectoryCleanup(fs::path directory, bool enabled)
      : directory_(std::move(directory)), enabled_(enabled) {}

  ScopedDirectoryCleanup(const ScopedDirectoryCleanup &) = delete;
  ScopedDirectoryCleanup &operator=(const ScopedDirectoryCleanup &) = delete;

  ~ScopedDirectoryCleanup() {
    if (enabled_) {
      std::error_code ignored;
      fs::remove_all(directory_, ignored);
    }
  }

private:
  fs::path directory_;
  bool enabled_ = false;
};

fs::path NormalizedPath(const fs::path &path) {
  std::error_code error;
  fs::path absolute = fs::absolute(path, error);
  if (error) {
    absolute = path;
  }
  error.clear();
  const fs::path normalized = fs::weakly_canonical(absolute, error);
  return error ? absolute.lexically_normal() : normalized;
}

bool PathsReferToSameLocation(const fs::path &left, const fs::path &right) {
  std::error_code error;
  if (fs::equivalent(left, right, error) && !error) {
    return true;
  }
  return NormalizedPath(left) == NormalizedPath(right);
}

std::string_view FrontendName(FrontendMode frontend) {
  switch (frontend) {
  case FrontendMode::Native:
    return "native";
  case FrontendMode::AstJsonBootstrap:
    return "ast-json-bootstrap";
  }
  return "native";
}

std::optional<FrontendMode> ParseFrontend(std::string_view name) {
  if (name == "native") {
    return FrontendMode::Native;
  }
  if (name == "ast-json-bootstrap") {
    return FrontendMode::AstJsonBootstrap;
  }
  std::cerr << "mdslc++: unsupported frontend '" << name
            << "'; expected native or ast-json-bootstrap (no fallback is "
               "performed)\n";
  return std::nullopt;
}

bool SameTimespec(const timespec &left, const timespec &right) {
  return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

bool SameFileMetadata(const struct stat &left, const struct stat &right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         left.st_size == right.st_size &&
         SameTimespec(left.st_mtim, right.st_mtim) &&
         SameTimespec(left.st_ctim, right.st_ctim);
}

std::optional<std::string> ReadSymlinkTarget(const fs::path &path,
                                             off_t expected_size,
                                             std::string &error_message) {
  std::size_t capacity = expected_size > 0
                             ? static_cast<std::size_t>(expected_size) + 1
                             : 256;
  constexpr std::size_t maximum_capacity = 1024 * 1024;
  while (capacity <= maximum_capacity) {
    std::string target(capacity, '\0');
    const ssize_t count = ::readlink(path.c_str(), target.data(), target.size());
    if (count < 0) {
      error_message = "cannot read symlink identity: " +
                      std::string(std::strerror(errno));
      return std::nullopt;
    }
    if (static_cast<std::size_t>(count) < target.size()) {
      target.resize(static_cast<std::size_t>(count));
      return target;
    }
    capacity *= 2;
  }
  error_message = "symlink target is too large to snapshot safely";
  return std::nullopt;
}

std::optional<std::vector<PathComponentSnapshot>>
ReadPathIdentity(const fs::path &absolute_path, std::string &error_message) {
  std::vector<PathComponentSnapshot> identity;
  fs::path current = absolute_path.root_path();
  for (const fs::path &component : absolute_path.relative_path()) {
    if (component == ".") {
      continue;
    }
    current /= component;
    struct stat before {};
    if (::lstat(current.c_str(), &before) != 0) {
      error_message = "cannot inspect dependency path identity: " +
                      std::string(std::strerror(errno));
      return std::nullopt;
    }
    PathComponentSnapshot snapshot{
        .path = current,
        .device = before.st_dev,
        .inode = before.st_ino,
        .type = static_cast<mode_t>(before.st_mode & S_IFMT),
        .size = before.st_size,
        .modified = before.st_mtim,
        .changed = before.st_ctim,
        .symlink_target = {}};
    if (S_ISLNK(before.st_mode)) {
      std::optional<std::string> target =
          ReadSymlinkTarget(current, before.st_size, error_message);
      if (!target) {
        return std::nullopt;
      }
      struct stat after {};
      if (::lstat(current.c_str(), &after) != 0 ||
          !SameFileMetadata(before, after)) {
        error_message = "dependency symlink changed while it was being read";
        return std::nullopt;
      }
      snapshot.symlink_target = std::move(*target);
    }
    identity.emplace_back(std::move(snapshot));
  }
  return identity;
}

bool PathIdentityMatches(const std::vector<PathComponentSnapshot> &expected,
                         std::string &error_message) {
  for (const PathComponentSnapshot &component : expected) {
    struct stat current {};
    if (::lstat(component.path.c_str(), &current) != 0) {
      error_message = "cannot inspect dependency path identity: " +
                      std::string(std::strerror(errno));
      return false;
    }
    if (current.st_dev != component.device ||
        current.st_ino != component.inode ||
        (current.st_mode & S_IFMT) != component.type) {
      error_message = "dependency path identity changed";
      return false;
    }
    if (S_ISLNK(current.st_mode)) {
      if (current.st_size != component.size ||
          !SameTimespec(current.st_mtim, component.modified) ||
          !SameTimespec(current.st_ctim, component.changed)) {
        error_message = "dependency symlink identity changed";
        return false;
      }
      std::optional<std::string> target =
          ReadSymlinkTarget(component.path, current.st_size, error_message);
      if (!target || *target != component.symlink_target) {
        if (target) {
          error_message = "dependency symlink target changed";
        }
        return false;
      }
    }
  }
  return true;
}

std::optional<SourceSnapshot> ReadSourceSnapshot(const fs::path &source,
                                                 std::string &error_message) {
  const int descriptor = ::open(source.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    error_message = "cannot open source: " + std::string(std::strerror(errno));
    return std::nullopt;
  }

  struct stat before {};
  if (::fstat(descriptor, &before) != 0) {
    error_message = "cannot inspect source: " +
                    std::string(std::strerror(errno));
    ::close(descriptor);
    return std::nullopt;
  }
  if (!S_ISREG(before.st_mode)) {
    error_message = "source is not a regular file";
    ::close(descriptor);
    return std::nullopt;
  }

  std::string contents;
  if (before.st_size > 0) {
    if (static_cast<std::uintmax_t>(before.st_size) > contents.max_size()) {
      error_message = "source is too large to snapshot safely";
      ::close(descriptor);
      return std::nullopt;
    }
    contents.reserve(static_cast<std::size_t>(before.st_size));
  }
  char buffer[16384];
  for (;;) {
    const ssize_t count = ::read(descriptor, buffer, sizeof(buffer));
    if (count > 0) {
      contents.append(buffer, static_cast<std::size_t>(count));
      continue;
    }
    if (count == 0) {
      break;
    }
    if (errno == EINTR) {
      continue;
    }
    error_message = "cannot read source: " + std::string(std::strerror(errno));
    ::close(descriptor);
    return std::nullopt;
  }

  struct stat after_descriptor {};
  const bool descriptor_stat_ok = ::fstat(descriptor, &after_descriptor) == 0;
  const int close_result = ::close(descriptor);
  struct stat after_path {};
  const bool path_stat_ok = ::stat(source.c_str(), &after_path) == 0;
  if (!descriptor_stat_ok || close_result != 0 || !path_stat_ok) {
    error_message = "cannot verify source after reading: " +
                    std::string(std::strerror(errno));
    return std::nullopt;
  }
  if (!SameFileMetadata(before, after_descriptor) ||
      !SameFileMetadata(after_descriptor, after_path) ||
      static_cast<off_t>(contents.size()) != after_descriptor.st_size) {
    error_message = "source changed while it was being read";
    return std::nullopt;
  }

  return SourceSnapshot{.device = after_descriptor.st_dev,
                        .inode = after_descriptor.st_ino,
                        .size = after_descriptor.st_size,
                        .modified = after_descriptor.st_mtim,
                        .changed = after_descriptor.st_ctim,
                        .contents = std::move(contents)};
}

bool SourceMatchesSnapshot(const fs::path &source,
                           const SourceSnapshot &expected,
                           std::string_view phase) {
  std::string error_message;
  const std::optional<SourceSnapshot> current =
      ReadSourceSnapshot(source, error_message);
  if (!current) {
    std::cerr << "mdslc++: source consistency check failed after " << phase
              << ": " << source << ": " << error_message << '\n';
    return false;
  }
  if (current->device != expected.device || current->inode != expected.inode ||
      current->size != expected.size ||
      !SameTimespec(current->modified, expected.modified) ||
      !SameTimespec(current->changed, expected.changed) ||
      current->contents != expected.contents) {
    std::cerr << "mdslc++: source changed during compilation after " << phase
              << ": " << source
              << "; refusing to emit an object from stale generated code\n";
    return false;
  }
  return true;
}

std::optional<std::vector<fs::path>>
ReadMakeDependencyPaths(const fs::path &dependency_file,
                        std::string &error_message) {
  std::ifstream input(dependency_file, std::ios::binary);
  if (!input) {
    error_message = "cannot open dependency scan output";
    return std::nullopt;
  }
  std::string contents((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
  if (!input.eof() && input.fail()) {
    error_message = "cannot read dependency scan output";
    return std::nullopt;
  }

  std::size_t separator = std::string::npos;
  bool escaped = false;
  for (std::size_t index = 0; index < contents.size(); ++index) {
    const char character = contents[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (character == '\\') {
      escaped = true;
      continue;
    }
    if (character == ':' &&
        (index + 1 == contents.size() || contents[index + 1] == ' ' ||
         contents[index + 1] == '\t' || contents[index + 1] == '\n' ||
         contents[index + 1] == '\r')) {
      separator = index;
      break;
    }
    if (character == '\n' || character == '\r') {
      break;
    }
  }
  if (separator == std::string::npos) {
    error_message = "dependency scan output has no Make rule separator";
    return std::nullopt;
  }

  std::vector<fs::path> paths;
  std::string token;
  const auto finish_token = [&]() {
    if (!token.empty()) {
      paths.emplace_back(std::move(token));
      token.clear();
    }
  };
  for (std::size_t index = separator + 1; index < contents.size(); ++index) {
    const char character = contents[index];
    if (character == '\\') {
      if (index + 1 == contents.size()) {
        error_message = "dependency scan output ends in an escape";
        return std::nullopt;
      }
      const char next = contents[index + 1];
      if (next == '\n') {
        ++index;
        continue;
      }
      if (next == '\r' && index + 2 < contents.size() &&
          contents[index + 2] == '\n') {
        index += 2;
        continue;
      }
      token += next;
      ++index;
      continue;
    }
    if (character == '$' && index + 1 < contents.size() &&
        contents[index + 1] == '$') {
      token += '$';
      ++index;
      continue;
    }
    if (character == ' ' || character == '\t') {
      finish_token();
      continue;
    }
    if (character == '\n' || character == '\r' || character == '#') {
      finish_token();
      break;
    }
    token += character;
  }
  finish_token();
  if (paths.empty()) {
    error_message = "dependency scan output contains no input files";
    return std::nullopt;
  }
  return paths;
}

std::optional<std::vector<DependencySnapshot>>
CaptureDependencyClosure(const fs::path &dependency_file,
                         std::string &error_message) {
  const std::optional<std::vector<fs::path>> parsed =
      ReadMakeDependencyPaths(dependency_file, error_message);
  if (!parsed) {
    return std::nullopt;
  }

  std::vector<DependencySnapshot> closure;
  std::vector<fs::path> captured_paths;
  for (const fs::path &dependency : *parsed) {
    std::error_code path_error;
    fs::path absolute = dependency;
    if (absolute.is_relative()) {
      absolute = fs::absolute(absolute, path_error);
      if (path_error) {
        error_message = "cannot resolve dependency path " +
                        dependency.string() + ": " + path_error.message();
        return std::nullopt;
      }
    }
    // Preserve the dependency path exactly as the compiler reported it after
    // making it absolute. Canonicalization and lexical normalization are both
    // unsafe here: `link/../header` is resolved by the kernel relative to the
    // symlink target, while lexically_normal() incorrectly collapses it before
    // the symlink is traversed.
    if (std::find(captured_paths.begin(), captured_paths.end(), absolute) !=
        captured_paths.end()) {
      continue;
    }
    std::string identity_error;
    std::optional<std::vector<PathComponentSnapshot>> path_identity =
        ReadPathIdentity(absolute, identity_error);
    if (!path_identity) {
      error_message = "cannot snapshot dependency path " + absolute.string() +
                      ": " + identity_error;
      return std::nullopt;
    }
    std::string snapshot_error;
    std::optional<SourceSnapshot> snapshot =
        ReadSourceSnapshot(absolute, snapshot_error);
    if (!snapshot) {
      error_message = "cannot snapshot dependency " + absolute.string() +
                      ": " + snapshot_error;
      return std::nullopt;
    }
    captured_paths.emplace_back(absolute);
    closure.push_back(DependencySnapshot{absolute, std::move(*snapshot),
                                         std::move(*path_identity)});
  }
  return closure;
}

std::optional<std::vector<fs::path>>
ReadDependencyPathSet(const fs::path &dependency_file,
                      const std::vector<fs::path> &excluded_paths,
                      std::string &error_message) {
  const std::optional<std::vector<fs::path>> parsed =
      ReadMakeDependencyPaths(dependency_file, error_message);
  if (!parsed) {
    return std::nullopt;
  }

  std::vector<fs::path> excluded_absolute;
  for (const fs::path &excluded : excluded_paths) {
    std::error_code path_error;
    fs::path absolute = excluded;
    if (absolute.is_relative()) {
      absolute = fs::absolute(absolute, path_error);
    }
    if (path_error) {
      error_message = "cannot resolve generated dependency path " +
                      excluded.string() + ": " + path_error.message();
      return std::nullopt;
    }
    excluded_absolute.emplace_back(std::move(absolute));
  }

  std::vector<fs::path> paths;
  for (const fs::path &dependency : *parsed) {
    std::error_code path_error;
    fs::path absolute = dependency;
    if (absolute.is_relative()) {
      absolute = fs::absolute(absolute, path_error);
    }
    if (path_error) {
      error_message = "cannot resolve dependency path " +
                      dependency.string() + ": " + path_error.message();
      return std::nullopt;
    }
    if (std::find(excluded_absolute.begin(), excluded_absolute.end(),
                  absolute) != excluded_absolute.end() ||
        std::find(paths.begin(), paths.end(), absolute) != paths.end()) {
      continue;
    }
    paths.emplace_back(std::move(absolute));
  }
  std::sort(paths.begin(), paths.end(),
            [](const fs::path &left, const fs::path &right) {
              return left.native() < right.native();
            });
  return paths;
}

bool RequireSameDependencyResolution(
    const fs::path &original_dependency_file,
    const fs::path &rewritten_dependency_file,
    const std::vector<fs::path> &excluded_paths) {
  std::string original_error;
  const std::optional<std::vector<fs::path>> original =
      ReadDependencyPathSet(original_dependency_file, {}, original_error);
  if (!original) {
    std::cerr << "mdslc++: cannot compare original source dependency "
                 "resolution: "
              << original_error << '\n';
    return false;
  }
  std::string rewritten_error;
  const std::optional<std::vector<fs::path>> rewritten =
      ReadDependencyPathSet(rewritten_dependency_file, excluded_paths,
                            rewritten_error);
  if (!rewritten) {
    std::cerr << "mdslc++: cannot compare rewritten host dependency "
                 "resolution: "
              << rewritten_error << '\n';
    return false;
  }
  if (*original == *rewritten) {
    return true;
  }

  const auto first_difference = [](const std::vector<fs::path> &left,
                                   const std::vector<fs::path> &right) {
    return std::find_if(left.begin(), left.end(), [&](const fs::path &path) {
      return std::find(right.begin(), right.end(), path) == right.end();
    });
  };
  const auto removed = first_difference(*original, *rewritten);
  const auto added = first_difference(*rewritten, *original);
  std::cerr << "mdslc++: dependency resolution changed between the original "
               "source and rewritten host";
  if (removed != original->end()) {
    std::cerr << "; no longer resolved: " << *removed;
  }
  if (added != rewritten->end()) {
    std::cerr << "; newly resolved: " << *added;
  }
  std::cerr << "; refusing to compile against a different header closure\n";
  return false;
}

void AppendDependencyClosure(
    std::vector<DependencySnapshot> &destination,
    std::vector<DependencySnapshot> additional) {
  for (DependencySnapshot &dependency : additional) {
    const bool already_captured =
        std::any_of(destination.begin(), destination.end(),
                    [&](const DependencySnapshot &existing) {
                      return existing.path == dependency.path;
                    });
    if (!already_captured) {
      destination.emplace_back(std::move(dependency));
    }
  }
}

bool DependencyClosureMatches(
    const std::vector<DependencySnapshot> &expected,
    std::string_view phase) {
  for (const DependencySnapshot &dependency : expected) {
    std::string identity_error;
    if (!PathIdentityMatches(dependency.path_identity, identity_error)) {
      std::cerr << "mdslc++: dependency changed during compilation after "
                << phase << ": " << dependency.path << ": "
                << identity_error
                << "; refusing to emit an object from stale generated code\n";
      return false;
    }
    std::string error_message;
    const std::optional<SourceSnapshot> current =
        ReadSourceSnapshot(dependency.path, error_message);
    if (!current) {
      std::cerr << "mdslc++: dependency consistency check failed after "
                << phase << ": " << dependency.path << ": " << error_message
                << '\n';
      return false;
    }
    const SourceSnapshot &snapshot = dependency.snapshot;
    if (current->device != snapshot.device ||
        current->inode != snapshot.inode || current->size != snapshot.size ||
        !SameTimespec(current->modified, snapshot.modified) ||
        !SameTimespec(current->changed, snapshot.changed) ||
        current->contents != snapshot.contents) {
      std::cerr << "mdslc++: dependency changed during compilation after "
                << phase << ": " << dependency.path
                << "; refusing to emit an object from stale generated code\n";
      return false;
    }
  }
  return true;
}

bool CompilationInputsMatch(
    const fs::path &source, const SourceSnapshot &source_snapshot,
    const std::vector<DependencySnapshot> &dependencies,
    std::string_view phase) {
  return SourceMatchesSnapshot(source, source_snapshot, phase) &&
         DependencyClosureMatches(dependencies, phase);
}

bool HasMdslExtension(std::string_view argument) {
  constexpr std::string_view extension = ".mdsl";
  return argument.size() >= extension.size() &&
         argument.substr(argument.size() - extension.size()) == extension;
}

bool OptionConsumesNextArgument(std::string_view argument) {
  return argument == "-o" || argument == "-I" || argument == "-L" ||
         argument == "-B" || argument == "-D" || argument == "-U" ||
         argument == "-l" || argument == "-x" || argument == "-arch" ||
         argument == "-target" || argument == "--target" ||
         argument == "--sysroot" || argument == "-isysroot" ||
         argument == "--gcc-toolchain" || argument == "-gcc-toolchain" ||
         argument == "-isystem" || argument == "-iquote" ||
         argument == "-idirafter" || argument == "-include" ||
         argument == "-imacros" || argument == "-iprefix" ||
         argument == "-iwithprefix" || argument == "-iwithprefixbefore" ||
         argument == "-MF" || argument == "-MT" || argument == "-MQ" ||
         argument == "-Xclang" || argument == "-Xlinker" ||
         argument == "-Xassembler" || argument == "-Xpreprocessor" ||
         argument == "-mllvm";
}

bool IsCompileAndLinkOptionWithValue(std::string_view argument) {
  return argument == "-B" || argument == "-arch" ||
         argument == "-target" || argument == "--target" ||
         argument == "--sysroot" || argument == "-isysroot" ||
         argument == "--gcc-toolchain" || argument == "-gcc-toolchain";
}

bool IsAttachedCompileAndLinkOption(std::string_view argument) {
  return argument == "-m32" || argument == "-m64" ||
         argument.starts_with("--target=") ||
         argument.starts_with("-target=") ||
         argument.starts_with("--sysroot=") ||
         (argument.starts_with("-isysroot") &&
          argument.size() > std::string_view("-isysroot").size()) ||
         argument.starts_with("-stdlib=") ||
         argument.starts_with("--gcc-toolchain=") ||
         argument.starts_with("-gcc-toolchain=") ||
         argument.starts_with("-rtlib=") ||
         argument.starts_with("-unwindlib=") ||
         (argument.starts_with("-B") && argument.size() > 2) ||
         argument.starts_with("-mabi=");
}

bool IsUnsupportedLtoMode(std::string_view argument) {
  return argument == "-flto" || argument.starts_with("-flto=") ||
         argument.starts_with("-flto-jobs=");
}

bool IsExtractionIncompatibleArgument(std::string_view argument) {
  return argument == "-E" || argument == "-S" ||
         argument == "-emit-llvm" || argument == "-fsyntax-only" ||
         argument == "-M" || argument == "-MM" || argument == "-MD" ||
         argument == "-MMD" || argument == "-MJ" || argument == "-MF" ||
         argument == "-MT" || argument == "-MQ" || argument == "-MP" ||
         argument == "-###" ||
         argument == "-Xclang" || argument == "-load" ||
         argument == "-fplugin" || argument == "--config" ||
         argument.starts_with("--config=") || argument.starts_with('@') ||
         argument == "-ivfsoverlay" || argument.starts_with("-ivfsoverlay") ||
         argument == "-vfsoverlay" || argument.starts_with("-vfsoverlay") ||
         argument.starts_with("-include-pch") ||
         argument.starts_with("-include-pth") ||
         argument.starts_with("-fmodule-file") ||
         argument.starts_with("-fprebuilt-module-path") ||
         argument.starts_with("-fplugin=");
}

bool IsUnsupportedFinalLinkMode(std::string_view argument) {
  return argument == "-shared" || argument == "--shared" ||
         argument == "-static" || argument == "--static" ||
         argument == "-static-pie" || argument == "--static-pie" ||
         argument == "-pie" || argument == "--pie" ||
         argument == "-no-pie" || argument == "--no-pie" ||
         argument == "-r" ||
         argument == "--relocatable" || argument == "-nostdlib" ||
         argument == "-nodefaultlibs" || argument.starts_with('@') ||
         argument.starts_with("-Xlinker=@") ||
         argument.starts_with("-Wl,");
}

bool IsLinkOptionWithValue(std::string_view argument) {
  return argument == "-L" || argument == "-l" ||
         argument == "-Xlinker";
}

bool IsAttachedLinkOption(std::string_view argument) {
  return (argument.starts_with("-L") && argument.size() > 2) ||
         (argument.starts_with("-l") && argument.size() > 2) ||
         argument.starts_with("-Wl,") || argument.starts_with("-fuse-ld=");
}

std::string QuoteForDisplay(std::string_view argument) {
  std::string quoted{"'"};
  for (char character : argument) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  quoted += '\'';
  return quoted;
}

int RunCommand(std::vector<std::string> command, bool verbose) {
  if (command.empty() || command.front().empty()) {
    std::cerr << "mdslc++: refusing to execute an empty command\n";
    return 1;
  }
  if (verbose) {
    std::cerr << "mdslc++:";
    for (const std::string &argument : command) {
      std::cerr << ' ' << QuoteForDisplay(argument);
    }
    std::cerr << '\n';
    std::cerr.flush();
  }

  std::vector<char *> arguments;
  arguments.reserve(command.size() + 1);
  for (std::string &argument : command) {
    arguments.push_back(argument.data());
  }
  arguments.push_back(nullptr);

  const pid_t child = fork();
  if (child < 0) {
    std::cerr << "mdslc++: failed to start " << command.front() << ": "
              << std::strerror(errno) << '\n';
    return 1;
  }

  if (child == 0) {
    execv(arguments.front(), arguments.data());
    std::cerr << "mdslc++: failed to execute " << command.front() << ": "
              << std::strerror(errno) << '\n';
    _exit(127);
  }

  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    std::cerr << "mdslc++: failed while waiting for " << command.front()
              << ": " << std::strerror(errno) << '\n';
    return 1;
  }

  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }

  std::cerr << "mdslc++: child process terminated with an unknown status\n";
  return 1;
}

std::optional<WrapperArguments> ParseWrapperArguments(int argc, char **argv) {
  WrapperArguments parsed;
  bool after_option_terminator = false;
  bool previous_option_consumes_argument = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (after_option_terminator) {
      parsed.compiler_arguments.emplace_back(argument);
      continue;
    }
    if (!previous_option_consumes_argument && argument == "--") {
      parsed.compiler_arguments.emplace_back(argument);
      after_option_terminator = true;
      continue;
    }
    if (!previous_option_consumes_argument && argument == "--verbose") {
      parsed.verbose = true;
      continue;
    }
    if (!previous_option_consumes_argument && argument == "--save-temps") {
      parsed.save_temps = true;
      continue;
    }
    if (!previous_option_consumes_argument &&
        argument.starts_with("--frontend=")) {
      if (parsed.frontend_was_explicit) {
        std::cerr << "mdslc++: --frontend may be specified only once\n";
        return std::nullopt;
      }
      const std::string_view name =
          argument.substr(std::string_view("--frontend=").size());
      const std::optional<FrontendMode> frontend = ParseFrontend(name);
      if (!frontend) {
        return std::nullopt;
      }
      parsed.frontend = *frontend;
      parsed.frontend_was_explicit = true;
      continue;
    }
    if (!previous_option_consumes_argument && argument == "--frontend") {
      std::cerr << "mdslc++: use --frontend=native or "
                   "--frontend=ast-json-bootstrap\n";
      return std::nullopt;
    }
    if (!previous_option_consumes_argument &&
        argument.starts_with("--tool-prefix-for-testing=")) {
#if MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE
      if (parsed.tool_prefix_for_testing) {
        std::cerr << "mdslc++: --tool-prefix-for-testing may be specified only "
                     "once\n";
        return std::nullopt;
      }
      const fs::path prefix(argument.substr(
          std::string_view("--tool-prefix-for-testing=").size()));
      if (prefix.empty() || !prefix.is_absolute()) {
        std::cerr << "mdslc++: --tool-prefix-for-testing requires an absolute "
                     "test fixture prefix\n";
        return std::nullopt;
      }
      parsed.tool_prefix_for_testing = prefix;
      continue;
#else
      std::cerr << "mdslc++: --tool-prefix-for-testing is unavailable in this "
                   "production driver build\n";
      return std::nullopt;
#endif
    }
    if (!previous_option_consumes_argument &&
        argument == "--tool-prefix-for-testing") {
#if MDSLC_ENABLE_TEST_TOOL_PREFIX_OVERRIDE
      std::cerr << "mdslc++: use --tool-prefix-for-testing=/absolute/path "
                   "only in deliberate driver tests\n";
#else
      std::cerr << "mdslc++: --tool-prefix-for-testing is unavailable in this "
                   "production driver build\n";
#endif
      return std::nullopt;
    }
    if (!previous_option_consumes_argument &&
        argument.starts_with("--matcore-target=")) {
      const std::string_view target =
          argument.substr(std::string_view("--matcore-target=").size());
      if (target != "cpu") {
        std::cerr << "mdslc++: unsupported Matcore target '" << target
                  << "'; bootstrap v0 supports only cpu and never falls back\n";
        return std::nullopt;
      }
      parsed.cpu_pipeline = true;
      continue;
    }
    if (!previous_option_consumes_argument &&
        argument == "--matcore-target") {
      std::cerr << "mdslc++: use --matcore-target=cpu\n";
      return std::nullopt;
    }

    parsed.compiler_arguments.emplace_back(argument);
    if (previous_option_consumes_argument) {
      previous_option_consumes_argument = false;
    } else {
      previous_option_consumes_argument = OptionConsumesNextArgument(argument);
    }
  }

  if (previous_option_consumes_argument) {
    std::cerr << "mdslc++: compiler option requires a value\n";
    return std::nullopt;
  }
  if (!parsed.cpu_pipeline && parsed.frontend_was_explicit) {
    std::cerr << "mdslc++: --frontend selects the Matcore extraction pipeline "
                 "and requires --matcore-target=cpu\n";
    return std::nullopt;
  }
  if (!parsed.cpu_pipeline && parsed.tool_prefix_for_testing) {
    std::cerr << "mdslc++: --tool-prefix-for-testing requires "
                 "--matcore-target=cpu\n";
    return std::nullopt;
  }
  return parsed;
}

std::vector<std::string> BuildDirectCommand(const WrapperArguments &arguments) {
  std::vector<std::string> command{MDSLC_DEFAULT_CLANGXX};
  if (arguments.save_temps) {
    command.emplace_back("-save-temps=obj");
  }

  bool after_option_terminator = false;
  bool previous_option_consumes_argument = false;
  bool injected_cpp_language_is_active = false;
  for (std::size_t index = 0; index < arguments.compiler_arguments.size();
       ++index) {
    const std::string_view argument = arguments.compiler_arguments[index];
    if (after_option_terminator) {
      command.emplace_back(argument);
      continue;
    }
    if (!previous_option_consumes_argument && argument == "--") {
      bool has_mdsl_input_after_terminator = false;
      for (std::size_t candidate = index + 1;
           candidate < arguments.compiler_arguments.size(); ++candidate) {
        if (HasMdslExtension(arguments.compiler_arguments[candidate])) {
          has_mdsl_input_after_terminator = true;
          break;
        }
      }
      if (has_mdsl_input_after_terminator) {
        command.emplace_back("-x");
        command.emplace_back("c++");
      }
      command.emplace_back(argument);
      after_option_terminator = true;
      continue;
    }

    const bool is_positional_argument =
        !previous_option_consumes_argument && !argument.starts_with('-');
    if (is_positional_argument && HasMdslExtension(argument)) {
      if (!injected_cpp_language_is_active) {
        command.emplace_back("-x");
        command.emplace_back("c++");
        injected_cpp_language_is_active = true;
      }
      command.emplace_back(argument);
    } else {
      if (is_positional_argument && injected_cpp_language_is_active) {
        command.emplace_back("-x");
        command.emplace_back("none");
        injected_cpp_language_is_active = false;
      }
      command.emplace_back(argument);
    }

    if (!previous_option_consumes_argument && argument == "-x") {
      injected_cpp_language_is_active = false;
    }
    if (previous_option_consumes_argument) {
      previous_option_consumes_argument = false;
    } else {
      previous_option_consumes_argument = OptionConsumesNextArgument(argument);
    }
  }
  return command;
}

bool RecordMdslInput(CpuInvocation &invocation, std::string_view input) {
  if (!HasMdslExtension(input)) {
    std::cerr << "mdslc++: CPU pipeline v0 accepts one .mdsl input and no "
                 "additional input files: "
              << input << '\n';
    return false;
  }
  if (!invocation.input.empty()) {
    std::cerr << "mdslc++: CPU pipeline v0 accepts exactly one .mdsl input\n";
    return false;
  }
  invocation.input = input;
  return true;
}

std::optional<CpuInvocation>
ParseCpuInvocation(const WrapperArguments &arguments) {
  CpuInvocation invocation;
  bool after_option_terminator = false;

  for (std::size_t index = 0; index < arguments.compiler_arguments.size();
       ++index) {
    const std::string &argument = arguments.compiler_arguments[index];
    if (after_option_terminator) {
      if (!RecordMdslInput(invocation, argument)) {
        return std::nullopt;
      }
      continue;
    }
    if (argument == "--") {
      after_option_terminator = true;
      continue;
    }
    if (argument == "-c") {
      invocation.compile_only = true;
      continue;
    }
    if (argument == "-o") {
      if (++index == arguments.compiler_arguments.size()) {
        std::cerr << "mdslc++: -o requires an output path\n";
        return std::nullopt;
      }
      if (!invocation.output.empty()) {
        std::cerr << "mdslc++: CPU pipeline v0 accepts one -o option\n";
        return std::nullopt;
      }
      invocation.output = arguments.compiler_arguments[index];
      continue;
    }
    if (argument.starts_with("-o") && argument.size() > 2) {
      if (!invocation.output.empty()) {
        std::cerr << "mdslc++: CPU pipeline v0 accepts one -o option\n";
        return std::nullopt;
      }
      invocation.output = argument.substr(2);
      continue;
    }
    if (argument == "-MD" || argument == "-MMD") {
      invocation.dependency_mode = argument;
      continue;
    }
    if (argument == "-MF") {
      if (++index == arguments.compiler_arguments.size()) {
        std::cerr << "mdslc++: -MF requires a dependency-file path\n";
        return std::nullopt;
      }
      invocation.dependency_file = arguments.compiler_arguments[index];
      continue;
    }
    if (argument.starts_with("-MF") && argument.size() > 3) {
      invocation.dependency_file = argument.substr(3);
      continue;
    }
    if (IsUnsupportedLtoMode(argument)) {
      std::cerr << "mdslc++: LTO mode " << argument
                << " is not supported by the CPU pipeline because the "
                   "generated relocatable-object boundary requires ordinary "
                   "native objects\n";
      return std::nullopt;
    }
    if (IsExtractionIncompatibleArgument(argument)) {
      std::cerr << "mdslc++: compiler argument is incompatible with the CPU "
                   "extraction pipeline: "
                << argument << '\n';
      return std::nullopt;
    }
    if (IsUnsupportedFinalLinkMode(argument)) {
      std::cerr << "mdslc++: final link mode " << argument
                << " is not implemented by the CPU bootstrap; use -c and "
                   "perform that link explicitly with clang++\n";
      return std::nullopt;
    }
    if (argument == "-x") {
      if (++index == arguments.compiler_arguments.size()) {
        std::cerr << "mdslc++: -x requires a language argument\n";
        return std::nullopt;
      }
      if (arguments.compiler_arguments[index] != "c++") {
        std::cerr << "mdslc++: .mdsl inputs must use -x c++\n";
        return std::nullopt;
      }
      continue;
    }
    if (OptionConsumesNextArgument(argument)) {
      if (++index == arguments.compiler_arguments.size()) {
        std::cerr << "mdslc++: compiler option requires a value: " << argument
                  << '\n';
        return std::nullopt;
      }
      const std::string &value = arguments.compiler_arguments[index];
      if (argument == "-Xlinker") {
        std::cerr << "mdslc++: opaque linker forwarding with -Xlinker is not "
                     "implemented by the CPU bootstrap; use -c and perform "
                     "that link explicitly with clang++\n";
        return std::nullopt;
      }
      if (IsLinkOptionWithValue(argument)) {
        invocation.has_link_only_arguments = true;
        invocation.link_arguments.emplace_back(argument);
        invocation.link_arguments.emplace_back(value);
      } else if (IsCompileAndLinkOptionWithValue(argument)) {
        invocation.compile_arguments.emplace_back(argument);
        invocation.compile_arguments.emplace_back(value);
        invocation.link_context_arguments.emplace_back(argument);
        invocation.link_context_arguments.emplace_back(value);
      } else {
        invocation.compile_arguments.emplace_back(argument);
        invocation.compile_arguments.emplace_back(value);
      }
      continue;
    }
    if (argument.starts_with('-')) {
      if (IsAttachedLinkOption(argument)) {
        invocation.has_link_only_arguments = true;
        invocation.link_arguments.emplace_back(argument);
      } else {
        invocation.compile_arguments.emplace_back(argument);
        if (IsAttachedCompileAndLinkOption(argument)) {
          invocation.link_context_arguments.emplace_back(argument);
        } else if (argument == "-pthread" ||
                   argument.starts_with("-fsanitize=")) {
          invocation.link_arguments.emplace_back(argument);
        }
      }
      continue;
    }
    if (!RecordMdslInput(invocation, argument)) {
      return std::nullopt;
    }
  }

  if (invocation.input.empty()) {
    std::cerr << "mdslc++: --matcore-target=cpu requires one .mdsl input\n";
    return std::nullopt;
  }
  if (invocation.output.empty()) {
    std::cerr << "mdslc++: CPU pipeline v0 requires an explicit -o output\n";
    return std::nullopt;
  }
  if (!invocation.dependency_file.empty() &&
      invocation.dependency_mode.empty()) {
    std::cerr << "mdslc++: -MF requires -MD or -MMD in the CPU pipeline\n";
    return std::nullopt;
  }
  if (!invocation.dependency_mode.empty() &&
      invocation.dependency_file.empty()) {
    fs::path dependency(invocation.output);
    dependency.replace_extension(".d");
    invocation.dependency_file = dependency.string();
  }
  if (invocation.dependency_file == "-") {
    std::cerr << "mdslc++: CPU pipeline dependency output must be a file, not "
                 "standard output\n";
    return std::nullopt;
  }
  if (invocation.compile_only && invocation.has_link_only_arguments) {
    std::cerr << "mdslc++: link-only arguments are invalid with the CPU "
                 "pipeline's -c mode\n";
    return std::nullopt;
  }
  return invocation;
}

std::optional<fs::path> StripRelativeSuffix(const fs::path &path,
                                            const fs::path &suffix) {
  if (suffix.empty() || suffix.is_absolute()) {
    return std::nullopt;
  }
  std::vector<fs::path> components;
  for (const fs::path &component : suffix.lexically_normal()) {
    if (component == ".") {
      continue;
    }
    if (component == ".." || component == "/") {
      return std::nullopt;
    }
    components.emplace_back(component);
  }
  if (components.empty()) {
    return std::nullopt;
  }

  fs::path prefix = path.lexically_normal();
  for (auto component = components.rbegin(); component != components.rend();
       ++component) {
    if (prefix.filename() != *component) {
      return std::nullopt;
    }
    prefix = prefix.parent_path();
  }
  return prefix;
}

std::optional<ToolLayout> LayoutUnderPrefix(const fs::path &prefix,
                                            const fs::path &bindir,
                                            const fs::path &includedir,
                                            const fs::path &libdir) {
  const auto is_safe_relative_path = [](const fs::path &path) {
    if (path.empty() || path.is_absolute()) {
      return false;
    }
    for (const fs::path &component : path.lexically_normal()) {
      if (component == ".." || component == "/") {
        return false;
      }
    }
    return true;
  };
  if (prefix.empty() || bindir.empty() || includedir.empty() ||
      libdir.empty() || !is_safe_relative_path(bindir) ||
      !is_safe_relative_path(includedir) ||
      !is_safe_relative_path(libdir)) {
    return std::nullopt;
  }
  const fs::path normalized_prefix = NormalizedPath(prefix);
  const fs::path binary_directory =
      (normalized_prefix / bindir).lexically_normal();
  const fs::path include_directory =
      (normalized_prefix / includedir).lexically_normal();
  const fs::path runtime_directory =
      (normalized_prefix / libdir).lexically_normal();
  return ToolLayout{.extractor = binary_directory / "matcore-extract",
                    .include_directory = include_directory,
                    .runtime_directory = runtime_directory,
                    .runtime_library =
                        runtime_directory / "libmatcore_runtime.so"};
}

bool SameLayout(const ToolLayout &left, const ToolLayout &right) {
  return NormalizedPath(left.extractor) == NormalizedPath(right.extractor) &&
         NormalizedPath(left.include_directory) ==
             NormalizedPath(right.include_directory) &&
         NormalizedPath(left.runtime_directory) ==
             NormalizedPath(right.runtime_directory);
}

bool IsUsableLayout(const ToolLayout &layout) {
  return ::access(layout.extractor.c_str(), X_OK) == 0 &&
         fs::is_regular_file(layout.include_directory / "matcore" / "mdsl.h") &&
         fs::is_regular_file(layout.include_directory / "matcore" /
                             "runtime_c.h") &&
         fs::is_regular_file(layout.runtime_library);
}

void AppendUniqueLayout(std::vector<ToolLayout> &layouts,
                        std::optional<ToolLayout> candidate) {
  if (!candidate) {
    return;
  }
  for (const ToolLayout &layout : layouts) {
    if (SameLayout(layout, *candidate)) {
      return;
    }
  }
  layouts.emplace_back(std::move(*candidate));
}

std::optional<ToolLayout>
DiscoverToolLayout(const std::optional<fs::path> &tool_prefix_for_testing) {
  std::error_code error;
  const fs::path executable = fs::canonical("/proc/self/exe", error);
  if (error) {
    std::cerr << "mdslc++: cannot resolve the running driver: "
              << error.message() << '\n';
    return std::nullopt;
  }

  const fs::path configured_bindir(MDSLC_INSTALL_BINDIR);
  const fs::path configured_includedir(MDSLC_INSTALL_INCLUDEDIR);
  const fs::path configured_libdir(MDSLC_INSTALL_LIBDIR);
  std::vector<ToolLayout> candidates;

  if (tool_prefix_for_testing) {
    const fs::path test_prefix = NormalizedPath(*tool_prefix_for_testing);
    if (!fs::is_directory(test_prefix)) {
      std::cerr << "mdslc++: test-only tool prefix is not a directory: "
                << test_prefix << '\n';
      return std::nullopt;
    }
    AppendUniqueLayout(candidates,
                       LayoutUnderPrefix(test_prefix, configured_bindir,
                                         configured_includedir,
                                         configured_libdir));
    AppendUniqueLayout(candidates,
                       LayoutUnderPrefix(test_prefix, "bin", "include", "lib"));
  } else {
    const std::optional<fs::path> configured_prefix = StripRelativeSuffix(
        executable.parent_path(), configured_bindir);
    if (configured_prefix) {
      AppendUniqueLayout(candidates,
                         LayoutUnderPrefix(*configured_prefix,
                                           configured_bindir,
                                           configured_includedir,
                                           configured_libdir));
    }
    AppendUniqueLayout(
        candidates,
        LayoutUnderPrefix(executable.parent_path().parent_path(), "bin",
                          "include", "lib"));
  }

  for (const ToolLayout &candidate : candidates) {
    if (IsUsableLayout(candidate)) {
      return candidate;
    }
  }

  std::cerr << "mdslc++: no coherent trusted Matcore tool layout was found "
               "relative to "
            << (tool_prefix_for_testing ? NormalizedPath(*tool_prefix_for_testing)
                                        : executable)
            << "; expected an executable matcore-extract, public headers, and "
               "libmatcore_runtime.so in one build or install prefix\n";
  return std::nullopt;
}

std::optional<fs::path> CreateTemporaryDirectory() {
  std::error_code error;
  const fs::path base = fs::temp_directory_path(error);
  if (error) {
    std::cerr << "mdslc++: cannot locate a temporary directory: "
              << error.message() << '\n';
    return std::nullopt;
  }
  std::string pattern = (base / "mdslc-XXXXXX").string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char *created = ::mkdtemp(writable.data());
  if (created == nullptr) {
    std::cerr << "mdslc++: mkdtemp failed: " << std::strerror(errno) << '\n';
    return std::nullopt;
  }
  return fs::path(created);
}

GeneratedArtifacts MakeArtifactPaths(const fs::path &directory,
                                     std::string_view stem) {
  const std::string prefix(stem);
  return {.host_source = directory / (prefix + ".host.cpp"),
          .host_overlay = directory / (prefix + ".host-overlay.yaml"),
          .ir = directory / (prefix + ".matcore.json"),
          .sites_header = directory / (prefix + ".sites.h"),
          .stubs_source = directory / (prefix + ".stubs.cpp"),
          .backend_source = directory / (prefix + ".backend.cpp"),
          .host_object = directory / (prefix + ".host.o"),
          .stubs_object = directory / (prefix + ".stubs.o"),
          .backend_object = directory / (prefix + ".backend.o")};
}

std::string JsonString(std::string_view value) {
  constexpr char hexadecimal[] = "0123456789abcdef";
  std::string encoded{"\""};
  for (const unsigned char character : value) {
    switch (character) {
    case '\"':
      encoded += "\\\"";
      break;
    case '\\':
      encoded += "\\\\";
      break;
    case '\b':
      encoded += "\\b";
      break;
    case '\f':
      encoded += "\\f";
      break;
    case '\n':
      encoded += "\\n";
      break;
    case '\r':
      encoded += "\\r";
      break;
    case '\t':
      encoded += "\\t";
      break;
    default:
      if (character < 0x20) {
        encoded += "\\u00";
        encoded += hexadecimal[(character >> 4) & 0x0f];
        encoded += hexadecimal[character & 0x0f];
      } else {
        encoded += static_cast<char>(character);
      }
      break;
    }
  }
  encoded += '\"';
  return encoded;
}

bool WriteHostOverlay(const fs::path &overlay,
                      const fs::path &virtual_source,
                      const fs::path &rewritten_source) {
  std::ofstream output(overlay, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "mdslc++: cannot create host VFS overlay: " << overlay
              << '\n';
    return false;
  }
  output << "{\n"
            "  \"version\": 0,\n"
            "  \"case-sensitive\": true,\n"
            "  \"use-external-names\": false,\n"
            "  \"roots\": [\n"
            "    {\n"
            "      \"type\": \"file\",\n"
            "      \"name\": "
         << JsonString(virtual_source.string())
         << ",\n"
            "      \"use-external-name\": false,\n"
            "      \"external-contents\": "
         << JsonString(rewritten_source.string())
         << "\n"
            "    }\n"
            "  ]\n"
            "}\n";
  if (!output) {
    std::cerr << "mdslc++: failed to write host VFS overlay: " << overlay
              << '\n';
    return false;
  }
  return true;
}

void AppendCompileEnvironment(std::vector<std::string> &command,
                              const CpuInvocation &invocation,
                              const ToolLayout &layout,
                              const fs::path &source_directory) {
  bool has_language_standard = false;
  for (const std::string &argument : invocation.compile_arguments) {
    if (argument.starts_with("-std=")) {
      has_language_standard = true;
      break;
    }
  }
  if (!has_language_standard) {
    command.emplace_back("-std=c++20");
  }
  command.emplace_back("-I" + layout.include_directory.string());
  command.emplace_back("-iquote");
  command.emplace_back(source_directory.string());
  command.insert(command.end(), invocation.compile_arguments.begin(),
                 invocation.compile_arguments.end());
}

int CompileGeneratedSource(const fs::path &source, const fs::path &object,
                           const CpuInvocation &invocation,
                           const ToolLayout &layout,
                           const fs::path &source_directory, bool verbose) {
  std::vector<std::string> command{MDSLC_DEFAULT_CLANGXX};
  AppendCompileEnvironment(command, invocation, layout, source_directory);
  command.insert(command.end(), {"-x", "c++", "-c", source.string(), "-o",
                                 object.string()});
  return RunCommand(std::move(command), verbose);
}

int CompileRewrittenHost(const fs::path &virtual_source,
                         const fs::path &overlay,
                         const fs::path &object,
                         const CpuInvocation &invocation,
                         const ToolLayout &layout,
                         const fs::path &source_directory, bool verbose) {
  std::vector<std::string> command{MDSLC_DEFAULT_CLANGXX};
  AppendCompileEnvironment(command, invocation, layout, source_directory);
  command.insert(command.end(),
                 {"-ivfsoverlay", overlay.string(), "-x", "c++", "-c",
                  virtual_source.string(), "-o", object.string()});
  return RunCommand(std::move(command), verbose);
}

int GenerateDependencyFile(const CpuInvocation &invocation,
                           std::string_view dependency_mode,
                           const ToolLayout &layout,
                           const fs::path &source_directory,
                           const fs::path &source,
                           const fs::path &object_target,
                           const fs::path &dependency_output, bool verbose) {
  std::vector<std::string> command{MDSLC_DEFAULT_CLANGXX};
  AppendCompileEnvironment(command, invocation, layout, source_directory);
  command.insert(command.end(),
                 {"-x", "c++", std::string(dependency_mode), "-MF",
                  dependency_output.string(), "-MQ", object_target.string(),
                  "-fsyntax-only", source.string()});
  return RunCommand(std::move(command), verbose);
}

int GenerateRewrittenHostDependencyFile(
    const CpuInvocation &invocation, std::string_view dependency_mode,
    const ToolLayout &layout, const fs::path &source_directory,
    const fs::path &virtual_source, const fs::path &overlay,
    const fs::path &object_target, const fs::path &dependency_output,
    bool verbose) {
  std::vector<std::string> command{MDSLC_DEFAULT_CLANGXX};
  AppendCompileEnvironment(command, invocation, layout, source_directory);
  command.insert(command.end(),
                 {"-ivfsoverlay", overlay.string(), "-x", "c++",
                  std::string(dependency_mode), "-MF",
                  dependency_output.string(), "-MQ", object_target.string(),
                  "-fsyntax-only", virtual_source.string()});
  return RunCommand(std::move(command), verbose);
}

std::string EscapeMakePath(std::string_view path) {
  std::string escaped;
  escaped.reserve(path.size());
  for (const char character : path) {
    if (character == '$') {
      escaped += "$$";
      continue;
    }
    if (character == ' ' || character == '\t' || character == '#' ||
        character == '\\') {
      escaped += '\\';
    }
    escaped += character;
  }
  return escaped;
}

bool MergeDependencyFiles(const std::vector<fs::path> &dependency_files,
                          const fs::path &object_target,
                          const std::vector<fs::path> &excluded_paths,
                          const fs::path &output) {
  std::vector<fs::path> dependencies;
  std::vector<fs::path> dependency_identities;
  for (const fs::path &dependency_file : dependency_files) {
    std::string parse_error;
    const std::optional<std::vector<fs::path>> parsed =
        ReadMakeDependencyPaths(dependency_file, parse_error);
    if (!parsed) {
      std::cerr << "mdslc++: cannot merge dependency output "
                << dependency_file << ": " << parse_error << '\n';
      return false;
    }
    for (const fs::path &dependency : *parsed) {
      std::error_code path_error;
      fs::path identity = dependency;
      if (identity.is_relative()) {
        identity = fs::absolute(identity, path_error);
      }
      if (path_error) {
        std::cerr << "mdslc++: cannot resolve published dependency "
                  << dependency << ": " << path_error.message() << '\n';
        return false;
      }
      const bool generated =
          std::any_of(excluded_paths.begin(), excluded_paths.end(),
                      [&](const fs::path &excluded) {
                        std::error_code excluded_error;
                        fs::path excluded_identity = excluded;
                        if (excluded_identity.is_relative()) {
                          excluded_identity =
                              fs::absolute(excluded_identity, excluded_error);
                        }
                        return !excluded_error && excluded_identity == identity;
                      });
      if (generated ||
          std::find(dependency_identities.begin(), dependency_identities.end(),
                    identity) != dependency_identities.end()) {
        continue;
      }
      dependency_identities.emplace_back(std::move(identity));
      dependencies.emplace_back(dependency);
    }
  }

  std::ofstream merged(output, std::ios::binary | std::ios::trunc);
  if (!merged) {
    std::cerr << "mdslc++: cannot create merged dependency output: " << output
              << '\n';
    return false;
  }
  merged << EscapeMakePath(object_target.string()) << ':';
  for (const fs::path &dependency : dependencies) {
    merged << " \\\n  " << EscapeMakePath(dependency.string());
  }
  merged << '\n';
  if (!merged) {
    std::cerr << "mdslc++: failed to write merged dependency output: "
              << output << '\n';
    return false;
  }
  return true;
}

bool PublishFileAtomically(const fs::path &source,
                           const fs::path &destination) {
  std::ifstream input(source, std::ios::binary);
  if (!input) {
    std::cerr << "mdslc++: dependency scan did not produce " << source << '\n';
    return false;
  }
  const fs::path temporary =
      destination.string() + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output << input.rdbuf();
    if (!output) {
      std::cerr << "mdslc++: failed to write dependency file " << temporary
                << '\n';
      std::error_code ignored;
      fs::remove(temporary, ignored);
      return false;
    }
  }
  std::error_code error;
  fs::rename(temporary, destination, error);
  if (error) {
    std::cerr << "mdslc++: failed to publish dependency file " << destination
              << ": " << error.message() << '\n';
    fs::remove(temporary, error);
    return false;
  }
  return true;
}

bool RequireGeneratedFile(const fs::path &path) {
  if (fs::is_regular_file(path)) {
    return true;
  }
  std::cerr << "mdslc++: extractor did not produce required artifact: " << path
            << '\n';
  return false;
}

int RunCpuPipeline(const WrapperArguments &wrapper,
                   const CpuInvocation &invocation) {
  const std::optional<ToolLayout> layout =
      DiscoverToolLayout(wrapper.tool_prefix_for_testing);
  if (!layout) {
    return 1;
  }

  std::error_code error;
  const fs::path output = fs::absolute(invocation.output, error);
  if (error || output.filename().empty()) {
    std::cerr << "mdslc++: invalid output path: " << invocation.output << '\n';
    return 2;
  }
  if (!fs::is_directory(output.parent_path())) {
    std::cerr << "mdslc++: output directory does not exist: "
              << output.parent_path() << '\n';
    return 2;
  }

  fs::path workspace;
  if (wrapper.save_temps) {
    workspace = output.parent_path();
  } else {
    const std::optional<fs::path> temporary = CreateTemporaryDirectory();
    if (!temporary) {
      return 1;
    }
    workspace = *temporary;
  }
  ScopedDirectoryCleanup cleanup(workspace, !wrapper.save_temps);
  const std::optional<fs::path> dependency_workspace =
      CreateTemporaryDirectory();
  if (!dependency_workspace) {
    return 1;
  }
  ScopedDirectoryCleanup dependency_cleanup(*dependency_workspace, true);

  std::string stem = output.stem().string();
  if (stem.empty()) {
    stem = output.filename().string();
  }
  const GeneratedArtifacts artifacts = MakeArtifactPaths(workspace, stem);
  const fs::path source_absolute = fs::absolute(invocation.input, error);
  if (error) {
    std::cerr << "mdslc++: invalid input path: " << invocation.input << '\n';
    return 2;
  }
  if (PathsReferToSameLocation(source_absolute, output)) {
    std::cerr << "mdslc++: output path must not overwrite the input .mdsl "
                 "source\n";
    return 2;
  }
  std::string source_snapshot_error;
  const std::optional<SourceSnapshot> source_snapshot =
      ReadSourceSnapshot(source_absolute, source_snapshot_error);
  if (!source_snapshot) {
    std::cerr << "mdslc++: cannot establish source consistency for "
              << source_absolute << ": " << source_snapshot_error << '\n';
    return 2;
  }
  const fs::path source_directory = source_absolute.parent_path();
  fs::path dependency_output;
  const fs::path private_source_dependency =
      *dependency_workspace / "private-source-closure.d";
  const fs::path public_source_dependency =
      *dependency_workspace / "public-source.d";
  const fs::path merged_public_dependency =
      *dependency_workspace / "public-combined.d";
  if (!invocation.dependency_mode.empty()) {
    dependency_output = fs::absolute(invocation.dependency_file, error);
    if (error || dependency_output.filename().empty() ||
        !fs::is_directory(dependency_output.parent_path())) {
      std::cerr << "mdslc++: invalid dependency-file path: "
                << invocation.dependency_file << '\n';
      return 2;
    }
    if (PathsReferToSameLocation(source_absolute, dependency_output) ||
        PathsReferToSameLocation(output, dependency_output)) {
      std::cerr << "mdslc++: dependency file must not overwrite or alias the "
                   "input or output\n";
      return 2;
    }
    for (const fs::path *generated :
         {&artifacts.host_source, &artifacts.host_overlay, &artifacts.ir,
          &artifacts.sites_header, &artifacts.stubs_source,
          &artifacts.backend_source, &artifacts.host_object,
          &artifacts.stubs_object, &artifacts.backend_object}) {
      if (PathsReferToSameLocation(*generated, dependency_output)) {
        std::cerr << "mdslc++: dependency file must not overwrite or alias a "
                     "generated --save-temps artifact: "
                  << *generated << '\n';
        return 2;
      }
    }
  }

  // The private consistency closure is intentionally independent of the
  // user's public depfile mode. In particular, -MD is required here so an
  // -isystem header cannot change between extraction and compilation.
  int result = GenerateDependencyFile(
      invocation, "-MD", *layout, source_directory, source_absolute, output,
      private_source_dependency, wrapper.verbose);
  if (!SourceMatchesSnapshot(source_absolute, *source_snapshot,
                             "dependency scanning")) {
    return 1;
  }
  if (result != 0) {
    return result;
  }
  std::string closure_error;
  std::optional<std::vector<DependencySnapshot>> captured_source_closure =
      CaptureDependencyClosure(private_source_dependency, closure_error);
  if (!captured_source_closure) {
    std::cerr << "mdslc++: cannot establish dependency consistency from "
              << private_source_dependency << ": " << closure_error << '\n';
    return 1;
  }
  std::vector<DependencySnapshot> dependency_closure =
      std::move(*captured_source_closure);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "dependency snapshotting")) {
    return 1;
  }

  std::vector<fs::path> public_dependency_files;
  if (!invocation.dependency_mode.empty()) {
    result = GenerateDependencyFile(
        invocation, invocation.dependency_mode, *layout, source_directory,
        source_absolute, output, public_source_dependency, wrapper.verbose);
    if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                                dependency_closure,
                                "public dependency scanning")) {
      return 1;
    }
    if (result != 0) {
      return result;
    }
    public_dependency_files.emplace_back(public_source_dependency);
  }

  std::vector<std::string> extract_command{
      layout->extractor.string(),
      "--frontend=" + std::string(FrontendName(wrapper.frontend)),
      "--input",
      invocation.input,
      "--ir-out",
      artifacts.ir.string(),
      "--rewrite-out",
      artifacts.host_source.string(),
      "--sites-out",
      artifacts.sites_header.string(),
      "--stubs-out",
      artifacts.stubs_source.string(),
      "--backend-out",
      artifacts.backend_source.string()};
  if (wrapper.verbose) {
    extract_command.emplace_back("--verbose");
  }
  extract_command.emplace_back("--");
  extract_command.emplace_back(MDSLC_DEFAULT_CLANGXX);
  AppendCompileEnvironment(extract_command, invocation, *layout,
                           source_directory);
  extract_command.emplace_back(invocation.input);

  result = RunCommand(std::move(extract_command), wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure, "frontend extraction")) {
    return 1;
  }
  if (result != 0) {
    return result;
  }
  if (!RequireGeneratedFile(artifacts.host_source) ||
      !RequireGeneratedFile(artifacts.ir) ||
      !RequireGeneratedFile(artifacts.sites_header) ||
      !RequireGeneratedFile(artifacts.stubs_source) ||
      !RequireGeneratedFile(artifacts.backend_source)) {
    return 1;
  }
  if (!WriteHostOverlay(artifacts.host_overlay, source_absolute,
                        fs::absolute(artifacts.host_source))) {
    return 1;
  }

  const fs::path private_host_dependency =
      *dependency_workspace / "private-host-closure.d";
  const fs::path private_stubs_dependency =
      *dependency_workspace / "private-stubs-closure.d";
  const fs::path private_backend_dependency =
      *dependency_workspace / "private-backend-closure.d";
  const fs::path public_host_dependency =
      *dependency_workspace / "public-host.d";
  const fs::path public_stubs_dependency =
      *dependency_workspace / "public-stubs.d";
  const fs::path public_backend_dependency =
      *dependency_workspace / "public-backend.d";

  const auto capture_generated_closure = [&](const fs::path &depfile,
                                             std::string_view phase) {
    if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                                dependency_closure, phase)) {
      return false;
    }
    std::string generated_closure_error;
    std::optional<std::vector<DependencySnapshot>> generated_closure =
        CaptureDependencyClosure(depfile, generated_closure_error);
    if (!generated_closure) {
      std::cerr << "mdslc++: cannot establish generated dependency "
                   "consistency from "
                << depfile << ": " << generated_closure_error << '\n';
      return false;
    }
    AppendDependencyClosure(dependency_closure,
                            std::move(*generated_closure));
    return CompilationInputsMatch(source_absolute, *source_snapshot,
                                  dependency_closure, phase);
  };

  result = GenerateRewrittenHostDependencyFile(
      invocation, "-MD", *layout, source_directory, source_absolute,
      artifacts.host_overlay, output, private_host_dependency,
      wrapper.verbose);
  if (result != 0) {
    return result;
  }
  const std::vector<fs::path> generated_host_paths{
      artifacts.host_source, artifacts.host_overlay, artifacts.ir,
      artifacts.sites_header, artifacts.stubs_source,
      artifacts.backend_source, artifacts.host_object,
      artifacts.stubs_object, artifacts.backend_object};
  if (!RequireSameDependencyResolution(private_source_dependency,
                                       private_host_dependency,
                                       generated_host_paths) ||
      !capture_generated_closure(private_host_dependency,
                                 "generated host dependency scanning")) {
    return 1;
  }
  result = GenerateDependencyFile(
      invocation, "-MD", *layout, source_directory, artifacts.stubs_source,
      output, private_stubs_dependency, wrapper.verbose);
  if (result != 0 ||
      !capture_generated_closure(private_stubs_dependency,
                                 "generated stub dependency scanning")) {
    return result != 0 ? result : 1;
  }
  result = GenerateDependencyFile(
      invocation, "-MD", *layout, source_directory, artifacts.backend_source,
      output, private_backend_dependency, wrapper.verbose);
  if (result != 0 ||
      !capture_generated_closure(private_backend_dependency,
                                 "generated backend dependency scanning")) {
    return result != 0 ? result : 1;
  }

  if (!invocation.dependency_mode.empty()) {
    const auto scan_public_generated = [&](const fs::path &source,
                                           const fs::path &depfile,
                                           bool rewritten_host) {
      const int scan_result =
          rewritten_host
              ? GenerateRewrittenHostDependencyFile(
                    invocation, invocation.dependency_mode, *layout,
                    source_directory, source_absolute, artifacts.host_overlay,
                    output, depfile, wrapper.verbose)
              : GenerateDependencyFile(
                    invocation, invocation.dependency_mode, *layout,
                    source_directory, source, output, depfile,
                    wrapper.verbose);
      if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                                  dependency_closure,
                                  "public generated dependency scanning")) {
        return 1;
      }
      return scan_result;
    };
    result = scan_public_generated(artifacts.host_source,
                                   public_host_dependency, true);
    if (result != 0) {
      return result;
    }
    public_dependency_files.emplace_back(public_host_dependency);
    result = scan_public_generated(artifacts.stubs_source,
                                   public_stubs_dependency, false);
    if (result != 0) {
      return result;
    }
    public_dependency_files.emplace_back(public_stubs_dependency);
    result = scan_public_generated(artifacts.backend_source,
                                   public_backend_dependency, false);
    if (result != 0) {
      return result;
    }
    public_dependency_files.emplace_back(public_backend_dependency);
  }

  result = CompileRewrittenHost(source_absolute, artifacts.host_overlay,
                                artifacts.host_object, invocation, *layout,
                                source_directory, wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "generated host compilation")) {
    return 1;
  }
  if (result != 0) {
    return result;
  }
  result = CompileGeneratedSource(artifacts.stubs_source,
                                  artifacts.stubs_object, invocation, *layout,
                                  source_directory, wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "generated stub compilation")) {
    return 1;
  }
  if (result != 0) {
    return result;
  }
  result = CompileGeneratedSource(artifacts.backend_source,
                                  artifacts.backend_object, invocation,
                                  *layout, source_directory, wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "generated backend compilation")) {
    return 1;
  }
  if (result != 0) {
    return result;
  }

  std::vector<std::string> link_command{MDSLC_DEFAULT_CLANGXX};
  link_command.insert(link_command.end(),
                      invocation.link_context_arguments.begin(),
                      invocation.link_context_arguments.end());
  link_command.insert(link_command.end(),
                      {artifacts.host_object.string(),
                       artifacts.stubs_object.string(),
                       artifacts.backend_object.string()});
  if (invocation.compile_only) {
    link_command.insert(link_command.end(),
                        {"-r", "-o", output.string()});
  } else {
    link_command.insert(link_command.end(), invocation.link_arguments.begin(),
                        invocation.link_arguments.end());
    link_command.emplace_back("-L" + layout->runtime_directory.string());
    link_command.emplace_back("-Xlinker");
    link_command.emplace_back("-rpath");
    link_command.emplace_back("-Xlinker");
    link_command.emplace_back(layout->runtime_directory.string());
    link_command.emplace_back("-lmatcore_runtime");
    link_command.emplace_back("-o");
    link_command.emplace_back(output.string());
  }
  result = RunCommand(std::move(link_command), wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure, "linking")) {
    fs::remove(output, error);
    return 1;
  }
  if (result != 0) {
    fs::remove(output, error);
    return result;
  }
  if (!invocation.dependency_mode.empty()) {
    const std::vector<fs::path> generated_dependencies{
        artifacts.host_source,   artifacts.host_overlay,
        artifacts.ir,            artifacts.sites_header,
        artifacts.stubs_source,  artifacts.backend_source,
        artifacts.host_object,   artifacts.stubs_object,
        artifacts.backend_object};
    if (!MergeDependencyFiles(public_dependency_files, output,
                              generated_dependencies,
                              merged_public_dependency) ||
        !PublishFileAtomically(merged_public_dependency,
                               dependency_output)) {
      fs::remove(output, error);
      return 1;
    }
  }
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "dependency publication")) {
    fs::remove(output, error);
    if (!invocation.dependency_mode.empty()) {
      fs::remove(dependency_output, error);
    }
    return 1;
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const std::optional<WrapperArguments> wrapper =
      ParseWrapperArguments(argc, argv);
  if (!wrapper) {
    return 2;
  }
  if (!wrapper->cpu_pipeline) {
    return RunCommand(BuildDirectCommand(*wrapper), wrapper->verbose);
  }

  const std::optional<CpuInvocation> invocation =
      ParseCpuInvocation(*wrapper);
  if (!invocation) {
    return 2;
  }
  return RunCpuPipeline(*wrapper, *invocation);
}
