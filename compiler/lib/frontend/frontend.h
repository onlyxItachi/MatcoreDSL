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
  std::vector<std::string> compiler_arguments;
  std::vector<std::string> trusted_public_headers;
  std::size_t maximum_ast_bytes = 512U * 1024U * 1024U;
  bool verbose = false;
};

struct Result {
  ir::Module module;
  std::vector<Diagnostic> diagnostics;
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

} // namespace matcore::mdslc::frontend

#endif // MATCORE_MDSLC_FRONTEND_H
