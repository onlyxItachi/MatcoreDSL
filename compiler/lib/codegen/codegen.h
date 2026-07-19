#ifndef MATCORE_MDSLC_CODEGEN_H
#define MATCORE_MDSLC_CODEGEN_H

#include "../ir/matcore_ir.h"

#include <string>
#include <string_view>

namespace matcore::mdslc::codegen {

struct Artifacts {
  std::string rewritten_host;
  std::string sites_header;
  std::string stubs_source;
  std::string backend_source;
};

// Generates all four textual artifacts as one transaction. sites_include is
// the quoted-include spelling used by the rewritten host and stubs source.
bool generate(const ir::Module &module, std::string_view original_source,
              std::string_view sites_include, Artifacts &artifacts,
              std::string &error);

} // namespace matcore::mdslc::codegen

#endif // MATCORE_MDSLC_CODEGEN_H
