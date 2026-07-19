#include "mdslc_config.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

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

int RunCompiler(std::vector<std::string> command) {
  std::vector<char *> arguments;
  arguments.reserve(command.size() + 1);
  for (std::string &argument : command) {
    arguments.push_back(argument.data());
  }
  arguments.push_back(nullptr);

  const pid_t child = fork();
  if (child < 0) {
    std::cerr << "mdslc++: failed to start clang++: " << std::strerror(errno)
              << '\n';
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
    std::cerr << "mdslc++: failed while waiting for clang++: "
              << std::strerror(errno) << '\n';
    return 1;
  }

  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }

  std::cerr << "mdslc++: clang++ terminated with an unknown process status\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  bool verbose = false;
  bool previous_option_consumes_argument = false;
  bool injected_cpp_language_is_active = false;
  std::vector<std::string> command{MDSLC_DEFAULT_CLANGXX};

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];

    if (!previous_option_consumes_argument && argument == "--verbose") {
      verbose = true;
      continue;
    }
    if (!previous_option_consumes_argument && argument == "--save-temps") {
      command.emplace_back("-save-temps=obj");
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

  if (verbose) {
    std::cerr << "mdslc++:";
    for (const std::string &argument : command) {
      std::cerr << ' ' << QuoteForDisplay(argument);
    }
    std::cerr << '\n';
    std::cerr.flush();
  }

  return RunCompiler(std::move(command));
}
