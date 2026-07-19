#include "mdslc_config.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct WrapperArguments {
  bool verbose = false;
  bool save_temps = false;
  bool cpu_pipeline = false;
  std::vector<std::string> compiler_arguments;
};

struct CpuInvocation {
  bool compile_only = false;
  bool has_link_only_arguments = false;
  std::string input;
  std::string output;
  std::vector<std::string> compile_arguments;
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
  fs::path ir;
  fs::path sites_header;
  fs::path stubs_source;
  fs::path backend_source;
  fs::path host_object;
  fs::path stubs_object;
  fs::path backend_object;
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
         argument == "-isystem" || argument == "-iquote" ||
         argument == "-idirafter" || argument == "-include" ||
         argument == "-imacros" || argument == "-iprefix" ||
         argument == "-iwithprefix" || argument == "-iwithprefixbefore" ||
         argument == "-MF" || argument == "-MT" || argument == "-MQ" ||
         argument == "-Xclang" || argument == "-Xlinker" ||
         argument == "-Xassembler" || argument == "-Xpreprocessor" ||
         argument == "-mllvm";
}

bool IsExtractionIncompatibleArgument(std::string_view argument) {
  return argument == "-E" || argument == "-S" ||
         argument == "-emit-llvm" || argument == "-fsyntax-only" ||
         argument == "-M" || argument == "-MM" || argument == "-MD" ||
         argument == "-MMD" || argument == "-MJ" || argument == "-MF" ||
         argument == "-MT" || argument == "-MQ" || argument == "-###" ||
         argument == "-Xclang" || argument == "-load" ||
         argument.starts_with("-fplugin=");
}

bool IsUnsupportedLinkerModeValue(std::string_view argument) {
  return argument == "-shared" || argument == "--shared" ||
         argument == "-static" || argument == "--static" ||
         argument == "-static-pie" || argument == "--static-pie" ||
         argument == "-pie" || argument == "--pie" ||
         argument == "-no-pie" || argument == "--no-pie" ||
         argument == "-r" || argument == "--relocatable";
}

bool ContainsUnsafeWlComponent(std::string_view argument) {
  constexpr std::string_view prefix = "-Wl,";
  if (!argument.starts_with(prefix)) {
    return false;
  }
  argument.remove_prefix(prefix.size());
  while (!argument.empty()) {
    const std::size_t comma = argument.find(',');
    const std::string_view component = argument.substr(0, comma);
    if (IsUnsupportedLinkerModeValue(component) || component.starts_with('@')) {
      return true;
    }
    if (comma == std::string_view::npos) {
      break;
    }
    argument.remove_prefix(comma + 1);
  }
  return false;
}

bool IsUnsupportedFinalLinkMode(std::string_view argument) {
  return argument == "-shared" || argument == "-static" ||
         argument == "-static-pie" || argument == "-pie" ||
         argument == "-no-pie" || argument == "-r" ||
         argument == "--relocatable" || argument == "-nostdlib" ||
         argument == "-nodefaultlibs" || argument.starts_with('@') ||
         argument.starts_with("-Xlinker=@") ||
         ContainsUnsafeWlComponent(argument);
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
      if (argument == "-Xlinker" &&
          (IsUnsupportedLinkerModeValue(value) || value.starts_with('@'))) {
        std::cerr << "mdslc++: linker mode " << value
                  << " is not implemented by the CPU bootstrap; use -c and "
                     "perform that link explicitly with clang++\n";
        return std::nullopt;
      }
      if (IsLinkOptionWithValue(argument)) {
        invocation.has_link_only_arguments = true;
        invocation.link_arguments.emplace_back(argument);
        invocation.link_arguments.emplace_back(value);
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
        if (argument == "-pthread" || argument.starts_with("-fsanitize=")) {
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
  if (invocation.compile_only && invocation.has_link_only_arguments) {
    std::cerr << "mdslc++: link-only arguments are invalid with the CPU "
                 "pipeline's -c mode\n";
    return std::nullopt;
  }
  return invocation;
}

std::optional<ToolLayout> DiscoverToolLayout() {
  std::error_code error;
  const fs::path executable = fs::canonical("/proc/self/exe", error);
  if (error) {
    std::cerr << "mdslc++: cannot resolve the running driver: "
              << error.message() << '\n';
    return std::nullopt;
  }

  const fs::path root = executable.parent_path().parent_path();
  ToolLayout layout{.extractor = root / "bin" / "matcore-extract",
                    .include_directory = root / "include",
                    .runtime_directory = root / "lib",
                    .runtime_library =
                        root / "lib" / "libmatcore_runtime.so"};
  if (::access(layout.extractor.c_str(), X_OK) != 0) {
    std::cerr << "mdslc++: relative extractor is unavailable: "
              << layout.extractor << '\n';
    return std::nullopt;
  }
  if (!fs::is_regular_file(layout.include_directory / "matcore" / "mdsl.h") ||
      !fs::is_regular_file(layout.include_directory / "matcore" /
                           "runtime_c.h")) {
    std::cerr << "mdslc++: build-tree Matcore headers are unavailable under "
              << layout.include_directory << '\n';
    return std::nullopt;
  }
  if (!fs::exists(layout.runtime_library)) {
    std::cerr << "mdslc++: relative CPU runtime is unavailable: "
              << layout.runtime_library << '\n';
    return std::nullopt;
  }
  return layout;
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
          .ir = directory / (prefix + ".matcore.json"),
          .sites_header = directory / (prefix + ".sites.h"),
          .stubs_source = directory / (prefix + ".stubs.cpp"),
          .backend_source = directory / (prefix + ".backend.cpp"),
          .host_object = directory / (prefix + ".host.o"),
          .stubs_object = directory / (prefix + ".stubs.o"),
          .backend_object = directory / (prefix + ".backend.o")};
}

void AppendCompileEnvironment(std::vector<std::string> &command,
                              const CpuInvocation &invocation,
                              const ToolLayout &layout,
                              const fs::path &generated_directory,
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
  command.insert(command.end(), invocation.compile_arguments.begin(),
                 invocation.compile_arguments.end());
  command.emplace_back("-I" + layout.include_directory.string());
  command.emplace_back("-I" + generated_directory.string());
  command.emplace_back("-iquote");
  command.emplace_back(source_directory.string());
}

int CompileGeneratedSource(const fs::path &source, const fs::path &object,
                           const CpuInvocation &invocation,
                           const ToolLayout &layout,
                           const fs::path &generated_directory,
                           const fs::path &source_directory, bool verbose) {
  std::vector<std::string> command{MDSLC_DEFAULT_CLANGXX};
  AppendCompileEnvironment(command, invocation, layout, generated_directory,
                           source_directory);
  command.insert(command.end(), {"-fPIC", "-x", "c++", "-c",
                                 source.string(), "-o", object.string()});
  return RunCommand(std::move(command), verbose);
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
  const std::optional<ToolLayout> layout = DiscoverToolLayout();
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
  if (source_absolute.lexically_normal() == output.lexically_normal()) {
    std::cerr << "mdslc++: output path must not overwrite the input .mdsl "
                 "source\n";
    return 2;
  }
  const fs::path source_directory = source_absolute.parent_path();

  std::vector<std::string> extract_command{
      layout->extractor.string(),
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
  AppendCompileEnvironment(extract_command, invocation, *layout, workspace,
                           source_directory);
  extract_command.emplace_back(invocation.input);

  int result = RunCommand(std::move(extract_command), wrapper.verbose);
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

  result = CompileGeneratedSource(artifacts.host_source, artifacts.host_object,
                                  invocation, *layout, workspace,
                                  source_directory, wrapper.verbose);
  if (result != 0) {
    return result;
  }
  result = CompileGeneratedSource(artifacts.stubs_source,
                                  artifacts.stubs_object, invocation, *layout,
                                  workspace, source_directory, wrapper.verbose);
  if (result != 0) {
    return result;
  }
  result = CompileGeneratedSource(artifacts.backend_source,
                                  artifacts.backend_object, invocation,
                                  *layout, workspace, source_directory,
                                  wrapper.verbose);
  if (result != 0) {
    return result;
  }

  std::vector<std::string> link_command{
      MDSLC_DEFAULT_CLANGXX, artifacts.host_object.string(),
      artifacts.stubs_object.string(), artifacts.backend_object.string()};
  if (invocation.compile_only) {
    link_command.insert(link_command.end(),
                        {"-r", "-o", output.string()});
  } else {
    link_command.insert(link_command.end(), invocation.link_arguments.begin(),
                        invocation.link_arguments.end());
    link_command.emplace_back("-L" + layout->runtime_directory.string());
    link_command.emplace_back("-Wl,-rpath," +
                              layout->runtime_directory.string());
    link_command.emplace_back("-lmatcore_runtime");
    link_command.emplace_back("-o");
    link_command.emplace_back(output.string());
  }
  return RunCommand(std::move(link_command), wrapper.verbose);
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
