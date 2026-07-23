#ifndef MATCORE_MDSLC_FRONTEND_H
#define MATCORE_MDSLC_FRONTEND_H

#include "../ir/matcore_ir.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace matcore::mdslc::frontend {

struct Diagnostic {
  std::string file;
  unsigned line = 0;
  unsigned column = 0;
  std::string message;
};

struct Options {
  std::string input_path;
  std::string clang_path = "/usr/bin/clang++-21";
  // Authenticated Clang builtin-header directory. It is an implementation
  // input, not part of the source-level compilation identity.
  std::string clang_resource_directory;
  std::vector<std::string> compiler_arguments;
  std::vector<std::string> trusted_public_headers;
  std::size_t maximum_ast_bytes = 512U * 1024U * 1024U;
  bool verbose = false;
};

struct Result {
  ir::Module module;
  std::vector<Diagnostic> diagnostics;
  // Exact bytes parsed for source ranges and later consumed by codegen.
  std::string source_snapshot;
};

class Frontend {
public:
  virtual ~Frontend() = default;
  virtual bool extract(const Options &options, Result &result) = 0;
};

// Bootstrap-only fallback. It executes Clang parsing/Sema and consumes Clang's
// structural JSON AST. The interface is intentionally independent of this
// representation so LibTooling can replace it without changing the IR layer.
std::unique_ptr<Frontend> createClangAstJsonBootstrapFrontend();

// Supported frontend. It runs Clang parsing/Sema in-process and authenticates
// operation declarations and source ranges through LibTooling APIs.
std::unique_ptr<Frontend> createClangLibToolingFrontend();

// Shared naming contract used by every frontend producer.
std::string stableSourceIdentity(const std::string &canonical_input);
std::string stableCompilationIdentity(const Options &options);
std::string makeStableSiteId(std::string_view source_identity,
                             std::string_view compilation_identity,
                             std::string_view source, std::uint64_t offset,
                             std::string_view kind);

} // namespace matcore::mdslc::frontend

#endif // MATCORE_MDSLC_FRONTEND_H
