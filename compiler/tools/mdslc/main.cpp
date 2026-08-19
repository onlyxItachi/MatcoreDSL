#include "mdslc_config.h"
#include "platform_support.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace support = matcore::mdslc::support;

bool IsWindowsHost();

bool WindowsCompilerOptionEquals(std::string_view argument,
                                 std::string_view name) noexcept {
  return support::windows_option_body_v1(argument) == name;
}

bool WindowsCompilerOptionStartsWith(std::string_view argument,
                                     std::string_view prefix) noexcept {
  return support::windows_option_body_v1(argument).starts_with(prefix);
}

bool CompilerOptionConsumesNextArgument(std::string_view argument) noexcept;

bool UnsafeConsumedCompilerValue(std::string_view value) noexcept {
  return !support::compiler_consumed_value_is_safe_v1(value);
}

enum class FrontendMode { Native, AstJsonBootstrap };
enum class SemanticPipelineMode { CaptureV0, MatcoreMlir };
enum class CompilerFlavor { ClangGnu, ClangCl };

constexpr std::string_view kConfiguredDefaultSemanticPipeline =
    MDSLC_DEFAULT_SEMANTIC_PIPELINE;
static_assert(kConfiguredDefaultSemanticPipeline == "capture-v0" ||
              kConfiguredDefaultSemanticPipeline == "matcore-mlir");
constexpr SemanticPipelineMode kDefaultSemanticPipeline =
    kConfiguredDefaultSemanticPipeline == "matcore-mlir"
        ? SemanticPipelineMode::MatcoreMlir
        : SemanticPipelineMode::CaptureV0;

struct CompilerToolchain {
  CompilerFlavor flavor = CompilerFlavor::ClangGnu;
  fs::path compiler;
  fs::path archiver;
  fs::path resource_directory;
};

struct WrapperArguments {
  bool verbose = false;
  bool save_temps = false;
  bool cpu_pipeline = false;
  bool frontend_was_explicit = false;
  bool semantic_pipeline_was_explicit = false;
  FrontendMode frontend = FrontendMode::Native;
  SemanticPipelineMode semantic_pipeline = kDefaultSemanticPipeline;
  std::optional<fs::path> tool_prefix_for_testing;
  std::optional<std::string> dependency_file;
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
  fs::path runtime_binary;
};

struct GeneratedArtifacts {
  fs::path host_source;
  fs::path host_overlay;
  fs::path ir;
  fs::path semantic_ir;
  fs::path sites_header;
  fs::path stubs_source;
  fs::path backend_source;
  fs::path host_object;
  fs::path stubs_object;
  fs::path backend_object;
};

struct SourceSnapshot {
  support::FileSnapshotV1 metadata;
  std::string contents;
};

struct DependencySnapshot {
  fs::path path;
  SourceSnapshot snapshot;
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
  std::string error;
  const fs::path normalized = support::normalize_path_v1(path, true, error);
  return error.empty() ? normalized : path.lexically_normal();
}

bool PathsReferToSameLocation(const fs::path &left, const fs::path &right) {
  std::string error;
  const bool same =
      support::paths_refer_to_same_location_v1(left, right, error);
  if (!error.empty()) {
    std::cerr << "mdslc++: cannot authenticate prospective path identity: "
              << error << '\n';
    return true;
  }
  return same;
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

std::string_view SemanticPipelineName(SemanticPipelineMode pipeline) {
  switch (pipeline) {
  case SemanticPipelineMode::CaptureV0:
    return "capture-v0";
  case SemanticPipelineMode::MatcoreMlir:
    return "matcore-mlir";
  }
  return "capture-v0";
}

std::optional<SemanticPipelineMode>
ParseSemanticPipeline(std::string_view name) {
  if (name == "capture-v0") return SemanticPipelineMode::CaptureV0;
  if (name == "matcore-mlir") return SemanticPipelineMode::MatcoreMlir;
  std::cerr << "mdslc++: unsupported semantic pipeline '" << name
            << "'; expected capture-v0 or matcore-mlir (no fallback is "
               "performed)\n";
  return std::nullopt;
}

bool SameFileSnapshot(const support::FileSnapshotV1 &left,
                      const support::FileSnapshotV1 &right) {
  return left.version == right.version && left.exists == right.exists &&
         left.regular_file == right.regular_file &&
         left.size_bytes == right.size_bytes &&
         left.last_write_time_ticks == right.last_write_time_ticks &&
         left.normalized_path == right.normalized_path &&
         support::same_file_identity_v1(left.identity, right.identity) &&
         left.path_identity_chain == right.path_identity_chain;
}

std::optional<SourceSnapshot> ReadSourceSnapshot(const fs::path &source,
                                                 std::string &error_message) {
  const support::FileSnapshotV1 before =
      support::capture_file_snapshot_v1(source, error_message);
  if (!error_message.empty()) {
    return std::nullopt;
  }
  if (!before.exists || !before.regular_file || !before.identity) {
    error_message = "source is not a regular file";
    return std::nullopt;
  }

  std::ifstream input(before.normalized_path, std::ios::binary);
  if (!input) {
    error_message = "cannot open source";
    return std::nullopt;
  }
  std::string contents((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
  if (!input.eof() && input.fail()) {
    error_message = "cannot read source";
    return std::nullopt;
  }
  if (static_cast<std::uint64_t>(contents.size()) != before.size_bytes) {
    error_message = "source size changed while it was being read";
    return std::nullopt;
  }
  std::string after_error;
  const support::FileSnapshotV1 after =
      support::capture_file_snapshot_v1(source, after_error);
  if (!after_error.empty()) {
    error_message = "cannot verify source after reading: " + after_error;
    return std::nullopt;
  }
  if (!SameFileSnapshot(before, after)) {
    error_message = "source changed while it was being read";
    return std::nullopt;
  }

  return SourceSnapshot{.metadata = after,
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
  if (!SameFileSnapshot(current->metadata, expected.metadata) ||
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
  bool token_failed = false;
  const auto finish_token = [&]() {
    if (!token.empty()) {
      std::string conversion_error;
      std::optional<fs::path> path =
          support::path_from_utf8_v1(token, conversion_error);
      if (!path) {
        error_message = "dependency path is not valid UTF-8: " +
                        conversion_error;
        token_failed = true;
      } else {
        paths.emplace_back(std::move(*path));
      }
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
      if (token_failed) return std::nullopt;
      break;
    }
    token += character;
  }
  finish_token();
  if (token_failed) return std::nullopt;
  if (paths.empty()) {
    error_message = "dependency scan output contains no input files";
    return std::nullopt;
  }
  return paths;
}

std::optional<std::vector<DependencySnapshot>>
CaptureDependencyPaths(const std::vector<fs::path> &dependencies,
                       std::string &error_message) {
  std::vector<DependencySnapshot> closure;
  std::vector<fs::path> captured_paths;
  for (const fs::path &dependency : dependencies) {
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
    std::string snapshot_error;
    std::optional<SourceSnapshot> snapshot =
        ReadSourceSnapshot(absolute, snapshot_error);
    if (!snapshot) {
      error_message = "cannot snapshot dependency " + absolute.string() +
                      ": " + snapshot_error;
      return std::nullopt;
    }
    captured_paths.emplace_back(absolute);
    closure.push_back(DependencySnapshot{absolute, std::move(*snapshot)});
  }
  return closure;
}

std::optional<std::vector<DependencySnapshot>>
CaptureDependencyClosure(const fs::path &dependency_file,
                         std::string &error_message) {
  const std::optional<std::vector<fs::path>> parsed =
      ReadMakeDependencyPaths(dependency_file, error_message);
  if (!parsed) {
    return std::nullopt;
  }
  return CaptureDependencyPaths(*parsed, error_message);
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
    const std::vector<fs::path> &excluded_paths,
    std::string_view comparison =
        "between the original source and rewritten host") {
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
  std::cerr << "mdslc++: dependency resolution changed " << comparison;
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
    if (!SameFileSnapshot(current->metadata, snapshot.metadata) ||
        current->contents != snapshot.contents) {
      std::cerr << "mdslc++: dependency changed during compilation after "
                << phase << ": " << dependency.path
                << "; content or path identity changed; refusing to emit an "
                   "object from stale generated code\n";
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

bool RequireMutationPathsDisjointFromDependencies(
    const std::vector<DependencySnapshot> &dependencies,
    const std::vector<std::pair<std::string_view, const fs::path *>>
        &mutation_paths) {
  for (const auto &[role, mutation_path] : mutation_paths) {
    for (const DependencySnapshot &dependency : dependencies) {
      if (PathsReferToSameLocation(*mutation_path, dependency.path)) {
        std::cerr << "mdslc++: " << role
                  << " must not overwrite or alias an authenticated input "
                     "dependency: "
                  << dependency.path << '\n';
        return false;
      }
    }
  }
  return true;
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
         argument == "-mllvm" || argument == "/I" || argument == "/D" ||
         argument == "/U" || argument == "/Fo" || argument == "/Fe";
}

bool CompilerOptionConsumesNextArgument(std::string_view argument) noexcept {
  return IsWindowsHost() ? support::clang_cl_option_consumes_next_v1(argument)
                         : OptionConsumesNextArgument(argument);
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
         argument.starts_with("-fplugin=") ||
         argument == "-resource-dir" ||
         argument.starts_with("-resource-dir=");
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
  if (support::process_launch_backend_v1() ==
      support::ProcessLaunchBackendV1::windows_create_process_w) {
    return support::quote_windows_command_line_argument_v1(argument);
  }
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

int RunCommand(
    std::vector<std::string> command, bool verbose,
    std::vector<support::EnvironmentOverrideV1> environment = {}) {
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

  support::ProcessRequestV1 request;
  request.argv = std::move(command);
  request.environment = std::move(environment);
  request.capture_stdout = false;
  request.capture_stderr = false;
  const support::ProcessResultV1 result = support::run_process_v1(request);
  if (!result.launched) {
    std::cerr << "mdslc++: failed to start " << request.argv.front() << ": "
              << result.error << '\n';
    return 1;
  }
  if (!result.error.empty()) {
    std::cerr << "mdslc++: process execution failed: " << result.error << '\n';
    return 1;
  }
  return static_cast<int>(result.exit_code);
}

std::vector<support::EnvironmentOverrideV1> CompilerChildEnvironment(
    std::vector<support::EnvironmentOverrideV1> additions = {}) {
  std::vector<support::EnvironmentOverrideV1> sanitized =
      support::compiler_environment_sanitization_v1();
  additions.insert(additions.end(),
                   std::make_move_iterator(sanitized.begin()),
                   std::make_move_iterator(sanitized.end()));
  return additions;
}

std::optional<support::ProcessResultV1>
RunCapturedCommand(
    std::vector<std::string> command, bool verbose,
    std::vector<support::EnvironmentOverrideV1> environment = {}) {
  if (command.empty() || command.front().empty()) {
    std::cerr << "mdslc++: refusing to execute an empty command\n";
    return std::nullopt;
  }
  if (verbose) {
    std::cerr << "mdslc++:";
    for (const std::string &argument : command) {
      std::cerr << ' ' << QuoteForDisplay(argument);
    }
    std::cerr << '\n';
  }
  support::ProcessRequestV1 request;
  request.argv = std::move(command);
  request.environment = std::move(environment);
  const support::ProcessResultV1 result = support::run_process_v1(request);
  if (!result.launched || !result.error.empty()) {
    std::cerr << "mdslc++: failed to execute " << request.argv.front() << ": "
              << result.error << '\n';
    return std::nullopt;
  }
  return result;
}

std::string PathArgument(const fs::path &path) {
  std::string error;
  const std::optional<std::string> encoded =
      support::path_to_utf8_v1(path, error);
  if (!encoded) {
    std::cerr << "mdslc++: cannot represent path as UTF-8: " << error << '\n';
    return {};
  }
  return *encoded;
}

std::optional<fs::path> PathFromArgument(std::string_view value,
                                         std::string_view role) {
  std::string error;
  std::optional<fs::path> path = support::path_from_utf8_v1(value, error);
  if (!path) {
    std::cerr << "mdslc++: invalid UTF-8 " << role << " path: " << error
              << '\n';
  }
  return path;
}

bool IsWindowsHost() {
  return support::process_launch_backend_v1() ==
         support::ProcessLaunchBackendV1::windows_create_process_w;
}

std::optional<CompilerToolchain> DiscoverCompilerToolchain(
    bool require_archiver) {
  std::string error;
  std::optional<fs::path> compiler;
  CompilerFlavor flavor = CompilerFlavor::ClangGnu;
  if (IsWindowsHost()) {
    flavor = CompilerFlavor::ClangCl;
    compiler = support::find_executable_v1(MDSLC_DEFAULT_CLANGXX, error);
    if (!compiler) {
      compiler = support::find_executable_v1("clang-cl.exe", error);
    }
  } else {
    // Preserve the configured clang++ spelling for invocation. Resolving its
    // symlink to `clang` would change C++ standard-library link behavior.
    const std::optional<fs::path> resolved =
        support::find_executable_v1(MDSLC_DEFAULT_CLANGXX, error);
    if (resolved) compiler = fs::path(MDSLC_DEFAULT_CLANGXX);
  }
  if (!compiler) {
    std::cerr << "mdslc++: cannot locate the configured Clang driver: "
              << error << '\n';
    return std::nullopt;
  }
  std::vector<std::string> version_command{PathArgument(*compiler)};
  if (flavor == CompilerFlavor::ClangCl) {
    version_command.emplace_back("--no-default-config");
  }
  version_command.emplace_back("--version");
  const std::optional<support::ProcessResultV1> version = RunCapturedCommand(
      std::move(version_command), false, CompilerChildEnvironment());
  if (!version || version->exit_code != 0 ||
      (version->stdout_text + version->stderr_text)
              .find("clang version 21.1.8") == std::string::npos) {
    std::cerr << "mdslc++: compiler must be the coherent Clang 21.1.8 "
                 "driver: "
              << *compiler << '\n';
    return std::nullopt;
  }

  std::vector<std::string> resource_command{PathArgument(*compiler)};
  if (flavor == CompilerFlavor::ClangCl) {
    resource_command.emplace_back("--no-default-config");
  }
  resource_command.emplace_back("-print-resource-dir");
  const std::optional<support::ProcessResultV1> resource_result =
      RunCapturedCommand(std::move(resource_command), false,
                         CompilerChildEnvironment());
  if (!resource_result || resource_result->exit_code != 0) {
    std::cerr << "mdslc++: cannot query the coherent Clang 21.1.8 resource "
                 "directory\n";
    return std::nullopt;
  }
  std::string resource_text = resource_result->stdout_text;
  while (!resource_text.empty() &&
         (resource_text.back() == '\r' || resource_text.back() == '\n' ||
          resource_text.back() == ' ' || resource_text.back() == '\t')) {
    resource_text.pop_back();
  }
  std::string resource_error;
  const std::optional<fs::path> resource_input =
      support::path_from_utf8_v1(resource_text, resource_error);
  if (!resource_input) {
    std::cerr << "mdslc++: Clang returned an invalid UTF-8 resource path: "
              << resource_error << '\n';
    return std::nullopt;
  }
  const fs::path resource_directory =
      support::normalize_path_v1(*resource_input, true, resource_error);
  if (!resource_error.empty() ||
      !fs::is_regular_file(resource_directory / "include" / "stddef.h")) {
    std::cerr << "mdslc++: coherent Clang resource headers are unavailable at "
              << resource_directory << '\n';
    return std::nullopt;
  }
  const fs::path compiler_prefix =
      NormalizedPath(compiler->parent_path().parent_path());
  std::error_code relative_error;
  const fs::path relative_resource =
      fs::relative(resource_directory, compiler_prefix, relative_error);
  if (relative_error || relative_resource.empty() ||
      relative_resource.is_absolute() ||
      *relative_resource.begin() == "..") {
    std::cerr << "mdslc++: Clang resource directory is outside the selected "
                 "21.1.8 toolchain prefix: "
              << resource_directory << '\n';
    return std::nullopt;
  }

  fs::path archiver;
  if (require_archiver) {
    const fs::path adjacent_archiver = compiler->parent_path() / "llvm-lib.exe";
    const std::optional<fs::path> discovered = support::find_executable_v1(
        PathArgument(adjacent_archiver), error);
    if (!discovered) {
      std::cerr << "mdslc++: Windows compile-only output requires llvm-lib "
                   "from the same LLVM 21.1.8 tool directory as clang-cl: "
                << error << '\n';
      return std::nullopt;
    }
    // llvm-lib intentionally has no --version interface.  Same-directory
    // discovery authenticates it against the already version-checked
    // clang-cl distribution.  When llvm-config is shipped, require its exact
    // version as an additional coherence check.
    const fs::path adjacent_config = compiler->parent_path() / "llvm-config.exe";
    std::string config_error;
    if (const std::optional<fs::path> config = support::find_executable_v1(
            PathArgument(adjacent_config), config_error)) {
      const std::optional<support::ProcessResultV1> archive_version =
          RunCapturedCommand({PathArgument(*config), "--version"}, false,
                             CompilerChildEnvironment());
      if (!archive_version || archive_version->exit_code != 0 ||
          archive_version->stdout_text.find("21.1.8") == std::string::npos) {
        std::cerr << "mdslc++: llvm-lib tool directory does not match LLVM "
                     "21.1.8\n";
        return std::nullopt;
      }
    }
    archiver = *discovered;
  }
  return CompilerToolchain{flavor, *compiler, archiver,
                           resource_directory};
}

int RunCompilerCommand(std::vector<std::string> command, bool verbose) {
  if (!IsWindowsHost()) {
    return RunCommand(std::move(command), verbose,
                      CompilerChildEnvironment());
  }
  if (command.empty()) {
    return RunCommand(std::move(command), verbose,
                      CompilerChildEnvironment());
  }

  if (verbose) {
    std::cerr << "mdslc++:";
    for (const std::string &argument : command) {
      std::cerr << ' ' << QuoteForDisplay(argument);
    }
    std::cerr << '\n';
  }
  std::string error;
  std::optional<support::TempDirectoryV1> temporary =
      support::create_temp_directory_v1("mdslc-response", error);
  if (!temporary) {
    std::cerr << "mdslc++: cannot create response-file directory: " << error
              << '\n';
    return 1;
  }
  const fs::path response = temporary->path() / "arguments.rsp";
  const std::vector<std::string> arguments(command.begin() + 1, command.end());
  if (!support::write_response_file_utf8_v1(
          response, arguments, support::ResponseFileSyntaxV1::windows,
          error)) {
    std::cerr << "mdslc++: cannot write compiler response file: " << error
              << '\n';
    return 1;
  }
  return RunCommand({command.front(), "@" + PathArgument(response)}, false,
                    CompilerChildEnvironment());
}

std::optional<support::ProcessResultV1>
RunCompilerCommandCaptured(
    std::vector<std::string> command, bool verbose,
    std::vector<support::EnvironmentOverrideV1> environment = {}) {
  if (!IsWindowsHost()) {
    return RunCapturedCommand(std::move(command), verbose,
                              CompilerChildEnvironment(
                                  std::move(environment)));
  }
  if (command.empty()) return std::nullopt;
  if (verbose) {
    std::cerr << "mdslc++:";
    for (const std::string &argument : command) {
      std::cerr << ' ' << QuoteForDisplay(argument);
    }
    std::cerr << '\n';
  }
  std::string error;
  std::optional<support::TempDirectoryV1> temporary =
      support::create_temp_directory_v1("mdslc-response", error);
  if (!temporary) {
    std::cerr << "mdslc++: cannot create response-file directory: " << error
              << '\n';
    return std::nullopt;
  }
  const fs::path response = temporary->path() / "arguments.rsp";
  const std::vector<std::string> arguments(command.begin() + 1, command.end());
  if (!support::write_response_file_utf8_v1(
          response, arguments, support::ResponseFileSyntaxV1::windows,
          error)) {
    std::cerr << "mdslc++: cannot write compiler response file: " << error
              << '\n';
    return std::nullopt;
  }
  return RunCapturedCommand(
      {command.front(), "@" + PathArgument(response)}, false,
      CompilerChildEnvironment(std::move(environment)));
}

std::optional<WrapperArguments> ParseWrapperArguments(int argc, char **argv) {
  WrapperArguments parsed;
  bool after_option_terminator = false;
  bool after_windows_link = false;
  bool previous_option_consumes_argument = false;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (IsWindowsHost() && argument.starts_with('@')) {
      std::cerr << "mdslc++: user response files are forbidden; pass explicit "
                   "compiler arguments so the driver can validate them\n";
      return std::nullopt;
    }
    if (after_windows_link) {
      parsed.compiler_arguments.emplace_back(argument);
      continue;
    }
    if (after_option_terminator) {
      parsed.compiler_arguments.emplace_back(argument);
      continue;
    }
    if (IsWindowsHost() && !previous_option_consumes_argument &&
        support::windows_option_equals_v1(argument, "link")) {
      parsed.compiler_arguments.emplace_back(argument);
      after_windows_link = true;
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
    if (IsWindowsHost() && !previous_option_consumes_argument &&
        (WindowsCompilerOptionEquals(argument, "TC") ||
         (support::windows_option_body_v1(argument).size() > 2 &&
          (WindowsCompilerOptionStartsWith(argument, "Tc") ||
           WindowsCompilerOptionStartsWith(argument, "Tp"))))) {
      std::cerr << "mdslc++: .mdsl inputs are valid C++ and cannot be compiled "
                   "with clang-cl C-language mode "
                << argument << '\n';
      return std::nullopt;
    }
    if (IsWindowsHost() && !previous_option_consumes_argument &&
        support::windows_option_starts_with_v1(argument, "clang:")) {
      std::cerr << "mdslc++: opaque /clang: forwarding is forbidden; use an "
                   "audited driver option directly\n";
      return std::nullopt;
    }
    if (IsWindowsHost() && !previous_option_consumes_argument &&
        (argument == "--driver-mode" ||
         argument.starts_with("--driver-mode="))) {
      std::cerr << "mdslc++: compiler driver mode is fixed to clang-cl and "
                   "cannot be overridden\n";
      return std::nullopt;
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
        argument.starts_with("--semantic-pipeline=")) {
      if (parsed.semantic_pipeline_was_explicit) {
        std::cerr << "mdslc++: --semantic-pipeline may be specified only "
                     "once\n";
        return std::nullopt;
      }
      const std::string_view name =
          argument.substr(std::string_view("--semantic-pipeline=").size());
      const auto pipeline = ParseSemanticPipeline(name);
      if (!pipeline) return std::nullopt;
      parsed.semantic_pipeline = *pipeline;
      parsed.semantic_pipeline_was_explicit = true;
      continue;
    }
    if (!previous_option_consumes_argument &&
        argument == "--semantic-pipeline") {
      std::cerr << "mdslc++: use --semantic-pipeline=capture-v0 or "
                   "--semantic-pipeline=matcore-mlir\n";
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
      const std::optional<fs::path> prefix = PathFromArgument(
          argument.substr(
              std::string_view("--tool-prefix-for-testing=").size()),
          "tool-prefix");
      if (!prefix || prefix->empty() || !prefix->is_absolute()) {
        std::cerr << "mdslc++: --tool-prefix-for-testing requires an absolute "
                     "test fixture prefix\n";
        return std::nullopt;
      }
      parsed.tool_prefix_for_testing = *prefix;
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
    if (!previous_option_consumes_argument &&
        argument.starts_with("--matcore-depfile=")) {
      if (parsed.dependency_file) {
        std::cerr << "mdslc++: --matcore-depfile may be specified only once\n";
        return std::nullopt;
      }
      const std::string_view value =
          argument.substr(std::string_view("--matcore-depfile=").size());
      if (value.empty() || value == "-" || UnsafeConsumedCompilerValue(value)) {
        std::cerr << "mdslc++: --matcore-depfile requires a non-response file "
                     "path\n";
        return std::nullopt;
      }
      parsed.dependency_file = std::string(value);
      continue;
    }
    if (!previous_option_consumes_argument &&
        argument == "--matcore-depfile") {
      std::cerr << "mdslc++: use --matcore-depfile=<path>\n";
      return std::nullopt;
    }

    parsed.compiler_arguments.emplace_back(argument);
    if (previous_option_consumes_argument) {
      previous_option_consumes_argument = false;
    } else {
      previous_option_consumes_argument =
          CompilerOptionConsumesNextArgument(argument);
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
  if (!parsed.cpu_pipeline && parsed.semantic_pipeline_was_explicit) {
    std::cerr << "mdslc++: --semantic-pipeline requires "
                 "--matcore-target=cpu\n";
    return std::nullopt;
  }
  if (parsed.semantic_pipeline == SemanticPipelineMode::MatcoreMlir &&
      parsed.frontend != FrontendMode::Native) {
    std::cerr << "mdslc++: --semantic-pipeline=matcore-mlir requires the "
                 "authenticated native frontend\n";
    return std::nullopt;
  }
  if (!parsed.cpu_pipeline && parsed.tool_prefix_for_testing) {
    std::cerr << "mdslc++: --tool-prefix-for-testing requires "
                 "--matcore-target=cpu\n";
    return std::nullopt;
  }
  if (!parsed.cpu_pipeline && parsed.dependency_file) {
    std::cerr << "mdslc++: --matcore-depfile requires --matcore-target=cpu\n";
    return std::nullopt;
  }
  return parsed;
}

std::vector<std::string> BuildDirectCommand(
    const WrapperArguments &arguments, const CompilerToolchain &toolchain) {
  std::vector<std::string> command{PathArgument(toolchain.compiler)};
  if (toolchain.flavor == CompilerFlavor::ClangCl) {
    command.emplace_back("--driver-mode=cl");
    command.emplace_back("--no-default-config");
    if (arguments.save_temps) {
      command.emplace_back("/clang:-save-temps=obj");
    }
    const bool has_cpp_mode =
        std::any_of(arguments.compiler_arguments.begin(),
                    arguments.compiler_arguments.end(), [](const auto &arg) {
                      return WindowsCompilerOptionEquals(arg, "TP");
                    });
    if (!has_cpp_mode) command.emplace_back("/TP");
    command.insert(command.end(), arguments.compiler_arguments.begin(),
                   arguments.compiler_arguments.end());
    return command;
  }
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

bool RecordMdslInput(CpuInvocation &invocation, std::string_view input);

std::optional<CpuInvocation>
ParseClangClCpuInvocation(const WrapperArguments &arguments) {
  CpuInvocation invocation;
  if (arguments.dependency_file) {
    invocation.dependency_mode = "-MD";
    invocation.dependency_file = *arguments.dependency_file;
  }
  for (std::size_t index = 0; index < arguments.compiler_arguments.size();
       ++index) {
    const std::string &argument = arguments.compiler_arguments[index];
    if (support::windows_option_equals_v1(argument, "link")) {
      if (invocation.compile_only) {
        std::cerr << "mdslc++: /link arguments are invalid with /c\n";
      } else {
        std::cerr << "mdslc++: user /link arguments are not supported by the "
                     "validated CPU pipeline; compile with /c and perform "
                     "the final link explicitly\n";
      }
      return std::nullopt;
    }
    if (WindowsCompilerOptionEquals(argument, "c")) {
      invocation.compile_only = true;
      continue;
    }
    const auto set_output = [&](std::string value) {
      if (!invocation.output.empty()) {
        std::cerr << "mdslc++: CPU pipeline accepts one output option\n";
        return false;
      }
      invocation.output = std::move(value);
      return true;
    };
    if (WindowsCompilerOptionEquals(argument, "Fo") ||
        WindowsCompilerOptionEquals(argument, "Fe") ||
        argument == "-o") {
      if (++index == arguments.compiler_arguments.size()) {
        std::cerr << "mdslc++: " << argument << " requires an output path\n";
        return std::nullopt;
      }
      if (!set_output(arguments.compiler_arguments[index])) {
        return std::nullopt;
      }
      continue;
    }
    const bool conventional_dash_output =
        argument.size() > 3 && argument.front() == '-' &&
        argument[1] == 'F';
    if (((argument.front() == '/' &&
          (WindowsCompilerOptionStartsWith(argument, "Fo") ||
           WindowsCompilerOptionStartsWith(argument, "Fe"))) ||
         (conventional_dash_output &&
          (WindowsCompilerOptionStartsWith(argument, "Fo") ||
           WindowsCompilerOptionStartsWith(argument, "Fe")))) &&
        support::windows_option_body_v1(argument).size() > 2) {
      std::string value = argument.substr(3);
      if (value.starts_with(':')) value.erase(value.begin());
      if (value.empty()) {
        std::cerr << "mdslc++: " << argument.substr(0, 3)
                  << " requires an output path\n";
        return std::nullopt;
      }
      if (!set_output(std::move(value))) return std::nullopt;
      continue;
    }
    if (WindowsCompilerOptionEquals(argument, "TP") ||
        argument == "--driver-mode=cl")
      continue;
    if (WindowsCompilerOptionEquals(argument, "TC") ||
        WindowsCompilerOptionStartsWith(argument, "Tc") ||
        WindowsCompilerOptionStartsWith(argument, "Tp")) {
      std::cerr << "mdslc++: .mdsl inputs require clang-cl C++ mode; "
                << argument << " is forbidden\n";
      return std::nullopt;
    }
    if (WindowsCompilerOptionEquals(argument, "LD") ||
        WindowsCompilerOptionEquals(argument, "LDd") ||
        argument.starts_with('@') ||
        WindowsCompilerOptionEquals(argument, "E") ||
        WindowsCompilerOptionEquals(argument, "P") ||
        WindowsCompilerOptionEquals(argument, "EP") ||
        WindowsCompilerOptionStartsWith(argument, "sourceDependencies")) {
      std::cerr << "mdslc++: compiler argument is incompatible with the CPU "
                   "extraction pipeline: "
                << argument << '\n';
      return std::nullopt;
    }
    if (support::clang_cl_option_consumes_next_v1(argument)) {
      if (++index == arguments.compiler_arguments.size()) {
        std::cerr << "mdslc++: clang-cl option requires a value: " << argument
                  << '\n';
        return std::nullopt;
      }
      if (UnsafeConsumedCompilerValue(arguments.compiler_arguments[index])) {
        std::cerr << "mdslc++: nested response-file expansion is forbidden "
                     "in the value for "
                  << argument << '\n';
        return std::nullopt;
      }
      invocation.compile_arguments.push_back(argument);
      invocation.compile_arguments.push_back(arguments.compiler_arguments[index]);
      if (support::clang_cl_option_is_link_context_v1(argument)) {
        invocation.link_context_arguments.push_back(argument);
        invocation.link_context_arguments.push_back(
            arguments.compiler_arguments[index]);
      }
      continue;
    }
    const auto compiler_argument_risk =
        support::classify_untrusted_compiler_argument_v1(argument, true);
    if (compiler_argument_risk != support::CompilerArgumentRiskV1::none) {
      std::cerr << "mdslc++: "
                << support::compiler_argument_risk_message_v1(
                       compiler_argument_risk)
                << ": " << argument << '\n';
      return std::nullopt;
    }
    if (argument.starts_with('/') || argument.starts_with('-')) {
      invocation.compile_arguments.push_back(argument);
      if (argument.starts_with("/fsanitize=") ||
          argument.starts_with("-fsanitize=")) {
        invocation.link_context_arguments.push_back(argument);
      }
      if (support::clang_cl_option_is_link_context_v1(argument) ||
          argument.starts_with("--target=") || argument == "-m32" ||
          argument == "-m64" || argument == "/MD" ||
          argument == "/MDd" || argument == "-MD" ||
          argument == "-MDd" || argument == "/MT" ||
          argument == "/MTd" || argument == "-MT" ||
          argument == "-MTd") {
        invocation.link_context_arguments.push_back(argument);
      }
      continue;
    }
    if (!RecordMdslInput(invocation, argument)) return std::nullopt;
  }

  if (invocation.input.empty()) {
    std::cerr << "mdslc++: --matcore-target=cpu requires one .mdsl input\n";
    return std::nullopt;
  }
  if (invocation.output.empty()) {
    std::cerr << "mdslc++: CPU pipeline requires an explicit /Fo, /Fe, or -o "
                 "output\n";
    return std::nullopt;
  }
  if (!invocation.dependency_mode.empty() &&
      invocation.dependency_file.empty()) {
    const std::optional<fs::path> output_path =
        PathFromArgument(invocation.output, "output");
    if (!output_path) return std::nullopt;
    fs::path dependency(*output_path);
    dependency.replace_extension(".d");
    invocation.dependency_file = PathArgument(dependency);
  }
  if (invocation.compile_only && invocation.has_link_only_arguments) {
    std::cerr << "mdslc++: /link arguments are invalid with /c\n";
    return std::nullopt;
  }
  const std::optional<fs::path> output_path =
      PathFromArgument(invocation.output, "output");
  if (!output_path) return std::nullopt;
  const std::optional<fs::path> input_path =
      PathFromArgument(invocation.input, "input");
  if (!input_path) return std::nullopt;
  if (PathsReferToSameLocation(*input_path, *output_path)) {
    std::cerr << "mdslc++: output path must not overwrite the input .mdsl "
                 "source\n";
    return std::nullopt;
  }
  if (invocation.compile_only &&
      !support::ascii_case_equal_v1(
          PathArgument(output_path->extension()), ".lib")) {
    std::cerr << "mdslc++: Windows /c mode emits a static library containing "
                 "the generated COFF objects; output must use .lib\n";
    return std::nullopt;
  }
  return invocation;
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
  if (IsWindowsHost()) return ParseClangClCpuInvocation(arguments);
  CpuInvocation invocation;
  if (arguments.dependency_file) {
    invocation.dependency_mode = "-MD";
    invocation.dependency_file = *arguments.dependency_file;
  }
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
      if (arguments.dependency_file) {
        std::cerr << "mdslc++: --matcore-depfile cannot be combined with "
                  << argument << "\n";
        return std::nullopt;
      }
      invocation.dependency_mode = argument;
      continue;
    }
    if (argument == "-MF") {
      if (arguments.dependency_file) {
        std::cerr << "mdslc++: --matcore-depfile cannot be combined with -MF\n";
        return std::nullopt;
      }
      if (++index == arguments.compiler_arguments.size()) {
        std::cerr << "mdslc++: -MF requires a dependency-file path\n";
        return std::nullopt;
      }
      invocation.dependency_file = arguments.compiler_arguments[index];
      continue;
    }
    if (argument.starts_with("-MF") && argument.size() > 3) {
      if (arguments.dependency_file) {
        std::cerr << "mdslc++: --matcore-depfile cannot be combined with -MF\n";
        return std::nullopt;
      }
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
    const std::optional<fs::path> output_path =
        PathFromArgument(invocation.output, "output");
    if (!output_path) return std::nullopt;
    fs::path dependency(*output_path);
    dependency.replace_extension(".d");
    invocation.dependency_file = PathArgument(dependency);
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
  if (IsWindowsHost()) {
    return ToolLayout{.extractor = binary_directory / "matcore-extract.exe",
                      .include_directory = include_directory,
                      .runtime_directory = runtime_directory,
                      .runtime_library =
                          runtime_directory / "matcore_runtime.lib",
                      .runtime_binary =
                          binary_directory / "matcore_runtime.dll"};
  }
  const fs::path runtime = runtime_directory / "libmatcore_runtime.so";
  return ToolLayout{.extractor = binary_directory / "matcore-extract",
                    .include_directory = include_directory,
                    .runtime_directory = runtime_directory,
                    .runtime_library = runtime,
                    .runtime_binary = runtime};
}

bool SameLayout(const ToolLayout &left, const ToolLayout &right) {
  return NormalizedPath(left.extractor) == NormalizedPath(right.extractor) &&
         NormalizedPath(left.include_directory) ==
             NormalizedPath(right.include_directory) &&
         NormalizedPath(left.runtime_directory) ==
             NormalizedPath(right.runtime_directory) &&
         NormalizedPath(left.runtime_library) ==
             NormalizedPath(right.runtime_library) &&
         NormalizedPath(left.runtime_binary) ==
             NormalizedPath(right.runtime_binary);
}

bool IsUsableLayout(const ToolLayout &layout) {
  std::string executable_error;
  const std::optional<fs::path> extractor = support::find_executable_v1(
      PathArgument(layout.extractor), executable_error);
  return extractor.has_value() &&
         fs::is_regular_file(layout.include_directory / "matcore" / "mdsl.h") &&
         fs::is_regular_file(layout.include_directory / "matcore" /
                             "runtime_c.h") &&
         fs::is_regular_file(layout.runtime_library) &&
         fs::is_regular_file(layout.runtime_binary);
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
  std::string executable_error;
  const std::optional<fs::path> running_executable =
      support::current_executable_path_v1(executable_error);
  if (!running_executable) {
    std::cerr << "mdslc++: cannot resolve the running driver: "
              << executable_error << '\n';
    return std::nullopt;
  }
  const fs::path &executable = *running_executable;

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
               "the platform runtime binary/import library in one build or "
               "install prefix\n";
  return std::nullopt;
}

std::optional<fs::path> CreateTemporaryDirectory() {
  std::string error;
  std::optional<support::TempDirectoryV1> temporary =
      support::create_temp_directory_v1("mdslc", error);
  if (!temporary) {
    std::cerr << "mdslc++: cannot create a temporary directory: " << error
              << '\n';
    return std::nullopt;
  }
  return temporary->release();
}

GeneratedArtifacts MakeArtifactPaths(const fs::path &directory,
                                     std::string_view stem) {
  const std::string prefix(stem);
  const std::string object_suffix = IsWindowsHost() ? ".obj" : ".o";
  return {.host_source = directory / (prefix + ".host.cpp"),
          .host_overlay = directory / (prefix + ".host-overlay.yaml"),
          .ir = directory / (prefix + ".matcore.json"),
          .semantic_ir = directory / (prefix + ".semantic.mlir"),
          .sites_header = directory / (prefix + ".sites.h"),
          .stubs_source = directory / (prefix + ".stubs.cpp"),
          .backend_source = directory / (prefix + ".backend.cpp"),
          .host_object = directory / (prefix + ".host" + object_suffix),
          .stubs_object = directory / (prefix + ".stubs" + object_suffix),
          .backend_object = directory / (prefix + ".backend" + object_suffix)};
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
         << JsonString(PathArgument(virtual_source))
         << ",\n"
            "      \"use-external-name\": false,\n"
            "      \"external-contents\": "
         << JsonString(PathArgument(rewritten_source))
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
                              const fs::path &source_directory,
                              const CompilerToolchain &toolchain,
                              bool include_resource_directory = true) {
  bool has_language_standard = false;
  for (const std::string &argument : invocation.compile_arguments) {
    if (argument.starts_with("-std=") || argument.starts_with("/std:")) {
      has_language_standard = true;
      break;
    }
  }
  if (toolchain.flavor == CompilerFlavor::ClangCl) {
    command.emplace_back("--driver-mode=cl");
    command.emplace_back("--no-default-config");
    if (!has_language_standard) command.emplace_back("/std:c++20");
    if (include_resource_directory) {
      command.emplace_back("-resource-dir=" +
                           PathArgument(toolchain.resource_directory));
    }
    command.emplace_back("/I");
    command.emplace_back(PathArgument(layout.include_directory));
    command.emplace_back("/I");
    command.emplace_back(PathArgument(source_directory));
    command.insert(command.end(), invocation.compile_arguments.begin(),
                   invocation.compile_arguments.end());
    return;
  }
  if (!has_language_standard) {
    command.emplace_back("-std=c++20");
  }
  if (include_resource_directory) {
    command.emplace_back("-resource-dir=" +
                         PathArgument(toolchain.resource_directory));
  }
  command.emplace_back("-I" + PathArgument(layout.include_directory));
  command.emplace_back("-iquote");
  command.emplace_back(PathArgument(source_directory));
  command.insert(command.end(), invocation.compile_arguments.begin(),
                 invocation.compile_arguments.end());
}

int CompileGeneratedSource(const fs::path &source, const fs::path &object,
                           const CpuInvocation &invocation,
                           const ToolLayout &layout,
                           const fs::path &source_directory,
                           const CompilerToolchain &toolchain, bool verbose) {
  std::vector<std::string> command{PathArgument(toolchain.compiler)};
  AppendCompileEnvironment(command, invocation, layout, source_directory,
                           toolchain);
  if (toolchain.flavor == CompilerFlavor::ClangCl) {
    command.insert(command.end(), {"/TP", "/c", PathArgument(source),
                                   "/Fo" + PathArgument(object)});
  } else {
    command.insert(command.end(), {"-x", "c++", "-c", PathArgument(source),
                                   "-o", PathArgument(object)});
  }
  return RunCompilerCommand(std::move(command), verbose);
}

int CompileRewrittenHost(const fs::path &virtual_source,
                         const fs::path &overlay,
                         const fs::path &object,
                         const CpuInvocation &invocation,
                         const ToolLayout &layout,
                         const fs::path &source_directory,
                         const CompilerToolchain &toolchain, bool verbose) {
  std::vector<std::string> command{PathArgument(toolchain.compiler)};
  AppendCompileEnvironment(command, invocation, layout, source_directory,
                           toolchain);
  if (toolchain.flavor == CompilerFlavor::ClangCl) {
    command.insert(command.end(),
                   {"-Xclang", "-ivfsoverlay", "-Xclang",
                    PathArgument(overlay), "/TP", "/c",
                    PathArgument(virtual_source),
                    "/Fo" + PathArgument(object)});
  } else {
    command.insert(command.end(),
                   {"-ivfsoverlay", PathArgument(overlay), "-x", "c++", "-c",
                    PathArgument(virtual_source), "-o", PathArgument(object)});
  }
  return RunCompilerCommand(std::move(command), verbose);
}

std::string EscapeMakePath(std::string_view path);

int GenerateClangClDependencyFile(
    std::vector<std::string> command, const fs::path &source,
    const fs::path &object_target, const fs::path &dependency_output,
    bool verbose) {
  command.insert(command.end(),
                 {"/TP", "/Zs", "/showIncludes", PathArgument(source)});
  const std::optional<support::ProcessResultV1> result =
      RunCompilerCommandCaptured(
          std::move(command), verbose,
          {{"VSLANG", std::optional<std::string>("1033")}});
  if (!result) return 1;

  constexpr std::string_view prefix = "Note: including file:";
  std::vector<std::string> dependencies{PathArgument(source)};
  std::string ordinary_stdout;
  std::size_t begin = 0;
  while (begin <= result->stdout_text.size()) {
    const std::size_t end = result->stdout_text.find('\n', begin);
    std::string_view line(result->stdout_text.data() + begin,
                          (end == std::string::npos
                               ? result->stdout_text.size()
                               : end) - begin);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.starts_with(prefix)) {
      line.remove_prefix(prefix.size());
      while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) {
        line.remove_prefix(1);
      }
      if (!line.empty()) dependencies.emplace_back(line);
    } else if (!line.empty()) {
      ordinary_stdout.append(line);
      ordinary_stdout.push_back('\n');
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (!ordinary_stdout.empty()) std::cout << ordinary_stdout;
  if (!result->stderr_text.empty()) std::cerr << result->stderr_text;
  if (result->exit_code != 0) return static_cast<int>(result->exit_code);

  std::sort(dependencies.begin(), dependencies.end());
  dependencies.erase(std::unique(dependencies.begin(), dependencies.end()),
                     dependencies.end());
  std::ofstream output(dependency_output, std::ios::binary | std::ios::trunc);
  if (!output) {
    std::cerr << "mdslc++: cannot create clang-cl dependency output: "
              << dependency_output << '\n';
    return 1;
  }
  output << EscapeMakePath(PathArgument(object_target)) << ':';
  for (const std::string &dependency : dependencies) {
    output << " \\\n  " << EscapeMakePath(dependency);
  }
  output << '\n';
  return output ? 0 : 1;
}

int GenerateDependencyFile(const CpuInvocation &invocation,
                           std::string_view dependency_mode,
                           const ToolLayout &layout,
                           const fs::path &source_directory,
                           const fs::path &source,
                           const fs::path &object_target,
                           const fs::path &dependency_output,
                           const CompilerToolchain &toolchain, bool verbose) {
  std::vector<std::string> command{PathArgument(toolchain.compiler)};
  AppendCompileEnvironment(command, invocation, layout, source_directory,
                           toolchain);
  if (toolchain.flavor == CompilerFlavor::ClangCl) {
    (void)dependency_mode;
    return GenerateClangClDependencyFile(
        std::move(command), source, object_target, dependency_output, verbose);
  }
  command.insert(command.end(),
                 {"-x", "c++", std::string(dependency_mode), "-MF",
                  PathArgument(dependency_output), "-MQ",
                  PathArgument(object_target), "-fsyntax-only",
                  PathArgument(source)});
  return RunCompilerCommand(std::move(command), verbose);
}

int GenerateRewrittenHostDependencyFile(
    const CpuInvocation &invocation, std::string_view dependency_mode,
    const ToolLayout &layout, const fs::path &source_directory,
    const fs::path &virtual_source, const fs::path &overlay,
    const fs::path &object_target, const fs::path &dependency_output,
    const CompilerToolchain &toolchain, bool verbose) {
  std::vector<std::string> command{PathArgument(toolchain.compiler)};
  AppendCompileEnvironment(command, invocation, layout, source_directory,
                           toolchain);
  if (toolchain.flavor == CompilerFlavor::ClangCl) {
    (void)dependency_mode;
    command.insert(command.end(),
                   {"-Xclang", "-ivfsoverlay", "-Xclang",
                    PathArgument(overlay)});
    return GenerateClangClDependencyFile(std::move(command), virtual_source,
                                         object_target, dependency_output,
                                         verbose);
  }
  command.insert(command.end(),
                 {"-ivfsoverlay", PathArgument(overlay), "-x", "c++",
                  std::string(dependency_mode), "-MF",
                  PathArgument(dependency_output), "-MQ",
                  PathArgument(object_target), "-fsyntax-only",
                  PathArgument(virtual_source)});
  return RunCompilerCommand(std::move(command), verbose);
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
        character == ':' || character == '\\') {
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
  merged << EscapeMakePath(PathArgument(object_target)) << ':';
  for (const fs::path &dependency : dependencies) {
    merged << " \\\n  " << EscapeMakePath(PathArgument(dependency));
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
  static std::atomic<std::uint64_t> sequence{0};
  std::random_device entropy;
  const std::uint64_t nonce =
      (static_cast<std::uint64_t>(entropy()) << 32U) ^ entropy() ^
      static_cast<std::uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count()) ^
      sequence.fetch_add(1, std::memory_order_relaxed);
  fs::path temporary = destination;
  temporary += ".tmp." + std::to_string(nonce);
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
  std::string publish_error;
  if (!support::replace_file_atomically_v1(temporary, destination,
                                           publish_error)) {
    std::cerr << "mdslc++: failed to publish dependency file " << destination
              << ": " << publish_error << '\n';
    std::error_code ignored;
    fs::remove(temporary, ignored);
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
  if (IsWindowsHost() && wrapper.frontend == FrontendMode::AstJsonBootstrap) {
    std::cerr << "mdslc++: the AST-JSON bootstrap frontend is unavailable "
                 "on Windows; use the native LibTooling frontend\n";
    return 2;
  }
  const std::optional<CompilerToolchain> toolchain =
      DiscoverCompilerToolchain(IsWindowsHost() && invocation.compile_only);
  if (!toolchain) return 1;

  const std::optional<ToolLayout> layout =
      DiscoverToolLayout(wrapper.tool_prefix_for_testing);
  if (!layout) {
    return 1;
  }

  std::error_code error;
  const std::optional<fs::path> requested_output =
      PathFromArgument(invocation.output, "output");
  if (!requested_output) return 2;
  const fs::path output = fs::absolute(*requested_output, error);
  if (error || output.filename().empty()) {
    std::cerr << "mdslc++: invalid output path: " << invocation.output << '\n';
    return 2;
  }
  std::string output_path_error;
  if (!support::prospective_output_path_supported_v1(output,
                                                     output_path_error)) {
    std::cerr << "mdslc++: unsupported output path: " << output_path_error
              << ": " << output << '\n';
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

  std::string stem = PathArgument(output.stem());
  if (stem.empty()) {
    stem = PathArgument(output.filename());
  }
  const GeneratedArtifacts artifacts = MakeArtifactPaths(workspace, stem);
  const std::optional<fs::path> requested_input =
      PathFromArgument(invocation.input, "input");
  if (!requested_input) return 2;
  const fs::path source_absolute = fs::absolute(*requested_input, error);
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
    const std::optional<fs::path> requested_dependency =
        PathFromArgument(invocation.dependency_file, "dependency output");
    if (!requested_dependency) return 2;
    dependency_output = fs::absolute(*requested_dependency, error);
    if (error || dependency_output.filename().empty() ||
        !fs::is_directory(dependency_output.parent_path())) {
      std::cerr << "mdslc++: invalid dependency-file path: "
                << invocation.dependency_file << '\n';
      return 2;
    }
    std::string dependency_path_error;
    if (!support::prospective_output_path_supported_v1(
            dependency_output, dependency_path_error)) {
      std::cerr << "mdslc++: unsupported dependency-file path: "
                << dependency_path_error << ": " << dependency_output << '\n';
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
    if (wrapper.semantic_pipeline == SemanticPipelineMode::MatcoreMlir &&
        PathsReferToSameLocation(artifacts.semantic_ir, dependency_output)) {
      std::cerr << "mdslc++: dependency file must not overwrite or alias the "
                   "generated semantic MLIR artifact: "
                << artifacts.semantic_ir << '\n';
      return 2;
    }
  }
  error.clear();
  if (fs::is_directory(output, error) && !error) {
    std::cerr << "mdslc++: requested output is a directory: " << output
              << '\n';
    return 2;
  }
  if (!invocation.dependency_mode.empty()) {
    error.clear();
    if (fs::is_directory(dependency_output, error) && !error) {
      std::cerr << "mdslc++: requested dependency output is a directory: "
                << dependency_output << '\n';
      return 2;
    }
  }

  // The private consistency closure is intentionally independent of the
  // user's public depfile mode. In particular, -MD is required here so an
  // -isystem header cannot change between extraction and compilation.
  int result = GenerateDependencyFile(
      invocation, "-MD", *layout, source_directory, source_absolute, output,
      private_source_dependency, *toolchain, wrapper.verbose);
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

  std::vector<std::pair<std::string_view, const fs::path *>> mutation_paths{
      {"requested output", &output},
      {"generated host source", &artifacts.host_source},
      {"generated host overlay", &artifacts.host_overlay},
      {"generated Matcore IR", &artifacts.ir},
      {"generated sites header", &artifacts.sites_header},
      {"generated stubs source", &artifacts.stubs_source},
      {"generated backend source", &artifacts.backend_source},
      {"generated host object", &artifacts.host_object},
      {"generated stubs object", &artifacts.stubs_object},
      {"generated backend object", &artifacts.backend_object},
  };
  if (wrapper.semantic_pipeline == SemanticPipelineMode::MatcoreMlir) {
    mutation_paths.emplace_back("generated Matcore semantic MLIR",
                                &artifacts.semantic_ir);
  }
  if (!invocation.dependency_mode.empty()) {
    mutation_paths.emplace_back("requested dependency output",
                                &dependency_output);
  }
  if (!RequireMutationPathsDisjointFromDependencies(dependency_closure,
                                                     mutation_paths)) {
    return 2;
  }

  // Only clear user-visible destinations after the compiler-authenticated
  // dependency closure proves that none of them is an input to this
  // translation unit. This ordering protects included files (and aliases of
  // them) from destructive output or --save-temps collisions.
  error.clear();
  fs::remove(output, error);
  if (error) {
    std::cerr << "mdslc++: cannot clear the requested output before "
                 "compilation: "
              << output << ": " << error.message() << '\n';
    return 2;
  }
  if (!invocation.dependency_mode.empty()) {
    error.clear();
    fs::remove(dependency_output, error);
    if (error) {
      std::cerr << "mdslc++: cannot clear the requested dependency output "
                   "before compilation: "
                << dependency_output << ": " << error.message() << '\n';
      return 2;
    }
  }

  std::vector<fs::path> public_dependency_files;
  if (!invocation.dependency_mode.empty()) {
    result = GenerateDependencyFile(
        invocation, invocation.dependency_mode, *layout, source_directory,
        source_absolute, output, public_source_dependency, *toolchain,
        wrapper.verbose);
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
      PathArgument(layout->extractor),
      "--frontend=" + std::string(FrontendName(wrapper.frontend)),
      "--input",
      PathArgument(source_absolute),
      "--ir-version=1",
      "--ir-out",
      PathArgument(artifacts.ir),
      "--rewrite-out",
      PathArgument(artifacts.host_source),
      "--sites-out",
      PathArgument(artifacts.sites_header),
      "--stubs-out",
      PathArgument(artifacts.stubs_source),
      "--backend-out",
      PathArgument(artifacts.backend_source)};
  extract_command.emplace_back(
      "--semantic-pipeline=" +
      std::string(SemanticPipelineName(wrapper.semantic_pipeline)));
  if (wrapper.semantic_pipeline == SemanticPipelineMode::MatcoreMlir) {
    extract_command.emplace_back("--semantic-ir-out");
    extract_command.emplace_back(PathArgument(artifacts.semantic_ir));
  }
  if (wrapper.verbose) {
    extract_command.emplace_back("--verbose");
  }
  std::vector<std::string> frontend_arguments{
      PathArgument(toolchain->compiler)};
  AppendCompileEnvironment(frontend_arguments, invocation, *layout,
                           source_directory, *toolchain, false);
  frontend_arguments.emplace_back(PathArgument(source_absolute));
  if (IsWindowsHost()) {
    const fs::path compiler_arguments_file =
        *dependency_workspace / "compiler-arguments.v1";
    std::string argument_file_error;
    if (!support::write_argument_file_v1(compiler_arguments_file,
                                         frontend_arguments,
                                         argument_file_error)) {
      std::cerr << "mdslc++: cannot write bounded frontend argument file: "
                << argument_file_error << '\n';
      return 1;
    }
    extract_command.emplace_back("--compiler-arguments-file");
    extract_command.emplace_back(PathArgument(compiler_arguments_file));
  } else {
    extract_command.emplace_back("--");
    extract_command.insert(extract_command.end(), frontend_arguments.begin(),
                           frontend_arguments.end());
  }

  result = RunCommand(std::move(extract_command), wrapper.verbose,
                      CompilerChildEnvironment());
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
      !RequireGeneratedFile(artifacts.backend_source) ||
      (wrapper.semantic_pipeline == SemanticPipelineMode::MatcoreMlir &&
       !RequireGeneratedFile(artifacts.semantic_ir))) {
    return 1;
  }
  if (!WriteHostOverlay(artifacts.host_overlay, source_absolute,
                        fs::absolute(artifacts.host_source))) {
    return 1;
  }
  std::vector<fs::path> generated_source_artifacts{
      artifacts.host_source, artifacts.host_overlay, artifacts.ir,
      artifacts.sites_header, artifacts.stubs_source,
      artifacts.backend_source};
  if (wrapper.semantic_pipeline == SemanticPipelineMode::MatcoreMlir)
    generated_source_artifacts.push_back(artifacts.semantic_ir);
  std::string generated_snapshot_error;
  std::optional<std::vector<DependencySnapshot>> generated_snapshots =
      CaptureDependencyPaths(generated_source_artifacts,
                             generated_snapshot_error);
  if (!generated_snapshots) {
    std::cerr << "mdslc++: cannot snapshot generated artifacts: "
              << generated_snapshot_error << '\n';
    return 1;
  }
  AppendDependencyClosure(dependency_closure,
                          std::move(*generated_snapshots));
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "generated artifact snapshotting")) {
    return 1;
  }

  const fs::path private_host_dependency =
      *dependency_workspace / "private-host-closure.d";
  const fs::path private_host_post_compile_dependency =
      *dependency_workspace / "private-host-post-compile.d";
  const fs::path private_host_final_dependency =
      *dependency_workspace / "private-host-final.d";
  const fs::path private_stubs_dependency =
      *dependency_workspace / "private-stubs-closure.d";
  const fs::path private_stubs_post_compile_dependency =
      *dependency_workspace / "private-stubs-post-compile.d";
  const fs::path private_stubs_final_dependency =
      *dependency_workspace / "private-stubs-final.d";
  const fs::path private_backend_dependency =
      *dependency_workspace / "private-backend-closure.d";
  const fs::path private_backend_post_compile_dependency =
      *dependency_workspace / "private-backend-post-compile.d";
  const fs::path private_backend_final_dependency =
      *dependency_workspace / "private-backend-final.d";
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

  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "before generated host dependency scanning")) {
    return 1;
  }
  result = GenerateRewrittenHostDependencyFile(
      invocation, "-MD", *layout, source_directory, source_absolute,
      artifacts.host_overlay, output, private_host_dependency,
      *toolchain, wrapper.verbose);
  if (result != 0) {
    return result;
  }
  std::vector<fs::path> generated_host_paths{
      artifacts.host_source, artifacts.host_overlay, artifacts.ir,
      artifacts.sites_header, artifacts.stubs_source,
      artifacts.backend_source, artifacts.host_object,
      artifacts.stubs_object, artifacts.backend_object};
  if (wrapper.semantic_pipeline == SemanticPipelineMode::MatcoreMlir)
    generated_host_paths.push_back(artifacts.semantic_ir);
  const auto recheck_host_resolution = [&](const fs::path &depfile,
                                           std::string_view phase) {
    const int scan_result = GenerateRewrittenHostDependencyFile(
        invocation, "-MD", *layout, source_directory, source_absolute,
        artifacts.host_overlay, output, depfile, *toolchain,
        wrapper.verbose);
    if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                                dependency_closure, phase)) {
      return 1;
    }
    if (scan_result != 0) {
      return scan_result;
    }
    return RequireSameDependencyResolution(private_source_dependency, depfile,
                                           generated_host_paths)
               ? 0
               : 1;
  };
  const auto recheck_generated_resolution =
      [&](const fs::path &source, const fs::path &baseline_depfile,
          const fs::path &rescan_depfile, std::string_view phase,
          std::string_view comparison) {
        const int scan_result = GenerateDependencyFile(
            invocation, "-MD", *layout, source_directory, source, output,
            rescan_depfile, *toolchain, wrapper.verbose);
        if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                                    dependency_closure, phase)) {
          return 1;
        }
        if (scan_result != 0) {
          return scan_result;
        }
        return RequireSameDependencyResolution(
                   baseline_depfile, rescan_depfile, {}, comparison)
                   ? 0
                   : 1;
      };
  if (!RequireSameDependencyResolution(private_source_dependency,
                                       private_host_dependency,
                                       generated_host_paths) ||
      !capture_generated_closure(private_host_dependency,
                                 "generated host dependency scanning")) {
    return 1;
  }
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "before generated stub dependency scanning")) {
    return 1;
  }
  result = GenerateDependencyFile(
      invocation, "-MD", *layout, source_directory, artifacts.stubs_source,
      output, private_stubs_dependency, *toolchain, wrapper.verbose);
  if (result != 0 ||
      !capture_generated_closure(private_stubs_dependency,
                                 "generated stub dependency scanning")) {
    return result != 0 ? result : 1;
  }
  if (!CompilationInputsMatch(
          source_absolute, *source_snapshot, dependency_closure,
          "before generated backend dependency scanning")) {
    return 1;
  }
  result = GenerateDependencyFile(
      invocation, "-MD", *layout, source_directory, artifacts.backend_source,
      output, private_backend_dependency, *toolchain, wrapper.verbose);
  if (result != 0 ||
      !capture_generated_closure(private_backend_dependency,
                                 "generated backend dependency scanning")) {
    return result != 0 ? result : 1;
  }

  if (!invocation.dependency_mode.empty()) {
    const auto scan_public_generated = [&](const fs::path &source,
                                           const fs::path &depfile,
                                           bool rewritten_host) {
      if (!CompilationInputsMatch(
              source_absolute, *source_snapshot, dependency_closure,
              "before public generated dependency scanning")) {
        return 1;
      }
      const int scan_result =
          rewritten_host
              ? GenerateRewrittenHostDependencyFile(
                    invocation, invocation.dependency_mode, *layout,
                    source_directory, source_absolute, artifacts.host_overlay,
                    output, depfile, *toolchain, wrapper.verbose)
              : GenerateDependencyFile(
                    invocation, invocation.dependency_mode, *layout,
                    source_directory, source, output, depfile,
                    *toolchain, wrapper.verbose);
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

  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "before generated host compilation")) {
    return 1;
  }
  result = CompileRewrittenHost(source_absolute, artifacts.host_overlay,
                                artifacts.host_object, invocation, *layout,
                                source_directory, *toolchain,
                                wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "generated host compilation")) {
    return 1;
  }
  if (result != 0) {
    return result;
  }
  result = recheck_host_resolution(private_host_post_compile_dependency,
                                   "post-compile host dependency scanning");
  if (result != 0) {
    fs::remove(artifacts.host_object, error);
    return result;
  }
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "before generated stub compilation")) {
    return 1;
  }
  result = CompileGeneratedSource(artifacts.stubs_source,
                                  artifacts.stubs_object, invocation, *layout,
                                  source_directory, *toolchain,
                                  wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "generated stub compilation")) {
    return 1;
  }
  if (result != 0) {
    return result;
  }
  result = recheck_generated_resolution(
      artifacts.stubs_source, private_stubs_dependency,
      private_stubs_post_compile_dependency,
      "post-compile stub dependency scanning",
      "between baseline and post-compile generated stubs");
  if (result != 0) {
    fs::remove(artifacts.stubs_object, error);
    return result;
  }
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "before generated backend compilation")) {
    return 1;
  }
  result = CompileGeneratedSource(artifacts.backend_source,
                                  artifacts.backend_object, invocation,
                                  *layout, source_directory, *toolchain,
                                  wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure,
                              "generated backend compilation")) {
    return 1;
  }
  if (result != 0) {
    return result;
  }
  result = recheck_generated_resolution(
      artifacts.backend_source, private_backend_dependency,
      private_backend_post_compile_dependency,
      "post-compile backend dependency scanning",
      "between baseline and post-compile generated backend");
  if (result != 0) {
    fs::remove(artifacts.backend_object, error);
    return result;
  }

  std::vector<std::string> link_command;
  if (toolchain->flavor == CompilerFlavor::ClangCl &&
      invocation.compile_only) {
    link_command = {PathArgument(toolchain->archiver),
                    "/OUT:" + PathArgument(output),
                    PathArgument(artifacts.host_object),
                    PathArgument(artifacts.stubs_object),
                    PathArgument(artifacts.backend_object)};
  } else {
    link_command = {PathArgument(toolchain->compiler)};
    if (toolchain->flavor == CompilerFlavor::ClangCl) {
      link_command.emplace_back("--driver-mode=cl");
      link_command.emplace_back("--no-default-config");
    }
    link_command.insert(link_command.end(),
                        invocation.link_context_arguments.begin(),
                        invocation.link_context_arguments.end());
    link_command.insert(link_command.end(),
                        {PathArgument(artifacts.host_object),
                         PathArgument(artifacts.stubs_object),
                         PathArgument(artifacts.backend_object)});
    if (invocation.compile_only) {
      link_command.insert(link_command.end(),
                          {"-r", "-o", PathArgument(output)});
    } else if (toolchain->flavor == CompilerFlavor::ClangCl) {
      link_command.emplace_back(PathArgument(layout->runtime_library));
      link_command.emplace_back("/Fe:" + PathArgument(output));
      if (!invocation.link_arguments.empty()) {
        link_command.emplace_back("/link");
        link_command.insert(link_command.end(),
                            invocation.link_arguments.begin(),
                            invocation.link_arguments.end());
      }
    } else {
      link_command.insert(link_command.end(),
                          invocation.link_arguments.begin(),
                          invocation.link_arguments.end());
      link_command.emplace_back("-L" +
                                PathArgument(layout->runtime_directory));
      link_command.emplace_back("-Xlinker");
      link_command.emplace_back("-rpath");
      link_command.emplace_back("-Xlinker");
      link_command.emplace_back(PathArgument(layout->runtime_directory));
      link_command.emplace_back("-lmatcore_runtime");
      link_command.emplace_back("-o");
      link_command.emplace_back(PathArgument(output));
    }
  }
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure, "before linking")) {
    return 1;
  }
  result = RunCompilerCommand(std::move(link_command), wrapper.verbose);
  if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                              dependency_closure, "linking")) {
    fs::remove(output, error);
    return 1;
  }
  if (result != 0) {
    fs::remove(output, error);
    return result;
  }
  error.clear();
  if (!fs::is_regular_file(output, error) || error) {
    std::cerr << "mdslc++: compiler/linker reported success but did not "
                 "produce the requested regular output file: "
              << output;
    if (error) std::cerr << ": " << error.message();
    std::cerr << '\n';
    fs::remove(output, error);
    return 1;
  }
  result = recheck_host_resolution(private_host_final_dependency,
                                   "final host dependency scanning");
  if (result != 0) {
    fs::remove(output, error);
    return result;
  }
  result = recheck_generated_resolution(
      artifacts.stubs_source, private_stubs_dependency,
      private_stubs_final_dependency, "final stub dependency scanning",
      "between baseline and final generated stubs");
  if (result != 0) {
    fs::remove(output, error);
    if (!invocation.dependency_mode.empty()) {
      fs::remove(dependency_output, error);
    }
    return result;
  }
  result = recheck_generated_resolution(
      artifacts.backend_source, private_backend_dependency,
      private_backend_final_dependency, "final backend dependency scanning",
      "between baseline and final generated backend");
  if (result != 0) {
    fs::remove(output, error);
    if (!invocation.dependency_mode.empty()) {
      fs::remove(dependency_output, error);
    }
    return result;
  }
  if (!invocation.dependency_mode.empty()) {
    std::vector<fs::path> generated_dependencies{
        artifacts.host_source,   artifacts.host_overlay,
        artifacts.ir,            artifacts.sites_header,
        artifacts.stubs_source,  artifacts.backend_source,
        artifacts.host_object,   artifacts.stubs_object,
        artifacts.backend_object};
    if (wrapper.semantic_pipeline == SemanticPipelineMode::MatcoreMlir)
      generated_dependencies.push_back(artifacts.semantic_ir);
    if (!CompilationInputsMatch(source_absolute, *source_snapshot,
                                dependency_closure,
                                "before dependency publication") ||
        !MergeDependencyFiles(public_dependency_files, output,
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

int MdslcMain(int argc, char **argv) {
  const std::optional<WrapperArguments> wrapper =
      ParseWrapperArguments(argc, argv);
  if (!wrapper) {
    return 2;
  }
  if (!wrapper->cpu_pipeline) {
    const std::optional<CompilerToolchain> toolchain =
        DiscoverCompilerToolchain(false);
    if (!toolchain) return 1;
    return RunCompilerCommand(BuildDirectCommand(*wrapper, *toolchain),
                              wrapper->verbose);
  }

  const std::optional<CpuInvocation> invocation =
      ParseCpuInvocation(*wrapper);
  if (!invocation) {
    return 2;
  }
  return RunCpuPipeline(*wrapper, *invocation);
}

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t **argv) {
  std::string error;
  const std::optional<std::vector<std::string>> utf8_arguments =
      support::wide_arguments_to_utf8_v1(argc, argv, error);
  if (!utf8_arguments) {
    std::cerr << "mdslc++: cannot decode Windows process arguments: " << error
              << '\n';
    return 2;
  }
  std::vector<char *> narrow_arguments;
  narrow_arguments.reserve(utf8_arguments->size());
  for (const std::string &argument : *utf8_arguments) {
    narrow_arguments.push_back(const_cast<char *>(argument.c_str()));
  }
  return MdslcMain(static_cast<int>(narrow_arguments.size()),
                   narrow_arguments.data());
}
#else
int main(int argc, char **argv) { return MdslcMain(argc, argv); }
#endif
