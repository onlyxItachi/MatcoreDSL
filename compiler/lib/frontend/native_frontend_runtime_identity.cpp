#include "frontend.h"
#include <clang/Basic/Version.h>
#include <filesystem>
#include <string_view>

#if defined(__linux__)
#include <link.h>
#endif

// Shared startup identity queries only. Keeping these unchanged queries out of
// the AST-admission TU lets inspection clients reuse the authoritative check
// without linking an unrelated AST consumer and its allocator instantiations.
namespace matcore::mdslc::frontend {

std::string nativeClangRuntimeVersionV1() {
  return clang::getClangFullVersion();
}

std::optional<std::string> nativeClangRuntimeLibraryPathV1(std::string &error) {
  error.clear();
#if defined(__linux__)
  struct SearchState {
    std::optional<std::string> path;
    bool ambiguous = false;
  } state;
  const auto inspect = [](dl_phdr_info *info, std::size_t, void *opaque) {
    auto &search = *static_cast<SearchState *>(opaque);
    if (info == nullptr || info->dlpi_name == nullptr ||
        info->dlpi_name[0] == '\0') {
      return 0;
    }
    const std::filesystem::path candidate(info->dlpi_name);
    const std::string filename = candidate.filename().string();
    if (!std::string_view(filename).starts_with("libclang-cpp.so"))
      return 0;
    if (search.path && *search.path != info->dlpi_name) {
      search.ambiguous = true;
      return 1;
    }
    search.path = info->dlpi_name;
    return 0;
  };
  dl_iterate_phdr(inspect, &state);
  if (state.ambiguous) {
    error = "multiple clang-cpp shared runtimes are loaded";
    return std::nullopt;
  }
  if (!state.path) {
    error = "loaded clang-cpp shared runtime path is unavailable";
    return std::nullopt;
  }
  return state.path;
#else
  error = "loaded clang-cpp shared runtime path is unsupported on this host";
  return std::nullopt;
#endif
}

} // namespace matcore::mdslc::frontend
