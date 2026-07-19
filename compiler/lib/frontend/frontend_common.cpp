#include "frontend.h"

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

namespace matcore::mdslc::frontend {

std::string stableSourceIdentity(const std::string &canonical_input) {
  const std::filesystem::path input(canonical_input);
  std::filesystem::path directory = input.parent_path();
  std::error_code error;
  while (!directory.empty()) {
    if (std::filesystem::exists(directory / ".git", error) && !error) {
      const std::filesystem::path relative =
          std::filesystem::relative(input, directory, error);
      if (!error && !relative.empty()) {
        return relative.lexically_normal().generic_string();
      }
    }
    error.clear();
    const std::filesystem::path parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return input.lexically_normal().generic_string();
}

namespace {

std::filesystem::path normalizedPath(const std::string &path) {
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    absolute = path;
  }
  error.clear();
  const std::filesystem::path canonical =
      std::filesystem::weakly_canonical(absolute, error);
  return error ? absolute.lexically_normal() : canonical;
}

bool samePath(const std::string &left, const std::filesystem::path &right) {
  return normalizedPath(left) == right;
}

} // namespace

std::string stableCompilationIdentity(const Options &options) {
  const std::filesystem::path input = normalizedPath(options.input_path);
  const std::filesystem::path source_directory = input.parent_path();
  std::vector<std::filesystem::path> trusted_include_directories;
  trusted_include_directories.reserve(options.trusted_public_headers.size());
  for (const std::string &header : options.trusted_public_headers) {
    trusted_include_directories.push_back(normalizedPath(header).parent_path()
                                              .parent_path());
  }

  const auto is_implicit_include = [&](const std::string &path) {
    const std::filesystem::path normalized = normalizedPath(path);
    if (normalized == source_directory) {
      return true;
    }
    for (const std::filesystem::path &trusted : trusted_include_directories) {
      if (normalized == trusted) {
        return true;
      }
    }
    return false;
  };

  std::vector<std::string> semantic_arguments;
  for (std::size_t index = 0; index < options.compiler_arguments.size();
       ++index) {
    const std::string &argument = options.compiler_arguments[index];
    if (!argument.starts_with('-') && samePath(argument, input)) {
      continue;
    }
    if (argument == "-x") {
      if (index + 1 < options.compiler_arguments.size()) {
        ++index;
      }
      continue;
    }
    if (argument == "-fsyntax-only" ||
        argument == "-fno-color-diagnostics" || argument == "-std=c++20") {
      continue;
    }
    if ((argument == "-I" || argument == "-iquote") &&
        index + 1 < options.compiler_arguments.size() &&
        is_implicit_include(options.compiler_arguments[index + 1])) {
      ++index;
      continue;
    }
    if (argument.starts_with("-I") && argument.size() > 2 &&
        is_implicit_include(argument.substr(2))) {
      continue;
    }
    semantic_arguments.push_back(argument);
  }

  std::ostringstream identity;
  for (const std::string &argument : semantic_arguments) {
    identity << argument.size() << ':' << argument << ';';
  }
  return identity.str();
}

std::string makeStableSiteId(std::string_view source_identity,
                             std::string_view compilation_identity,
                             std::string_view source, std::uint64_t offset,
                             std::string_view kind) {
  std::uint64_t left = 14695981039346656037ULL;
  std::uint64_t right = 7809847782465536322ULL;
  const auto append = [&left, &right](std::string_view value) {
    for (const unsigned char character : value) {
      left ^= character;
      left *= 1099511628211ULL;
      right ^= static_cast<unsigned char>(character + 0x9dU);
      right *= 14029467366897019727ULL;
    }
    left ^= 0xffU;
    left *= 1099511628211ULL;
    right ^= 0x5aU;
    right *= 14029467366897019727ULL;
  };
  append("matcore-site-v0");
  append(source_identity);
  if (!compilation_identity.empty()) {
    append(compilation_identity);
  }
  append(source);
  append(std::to_string(offset));
  append(kind);
  std::ostringstream formatted;
  formatted << "mc_" << std::hex << std::setfill('0') << std::setw(16) << left
            << std::setw(16) << right;
  return formatted.str();
}

} // namespace matcore::mdslc::frontend
