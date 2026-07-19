#include "frontend.h"

#include <filesystem>
#include <iomanip>
#include <sstream>

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

std::string makeStableSiteId(std::string_view source_identity,
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
  append(source);
  append(std::to_string(offset));
  append(kind);
  std::ostringstream formatted;
  formatted << "mc_" << std::hex << std::setfill('0') << std::setw(16) << left
            << std::setw(16) << right;
  return formatted.str();
}

} // namespace matcore::mdslc::frontend
