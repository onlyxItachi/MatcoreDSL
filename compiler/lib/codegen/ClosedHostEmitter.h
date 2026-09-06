#ifndef MATCORE_MDSLC_CLOSED_HOST_EMITTER_H
#define MATCORE_MDSLC_CLOSED_HOST_EMITTER_H

#include "../frontend/ClosedRegionAdmission.h"
#include <optional>
#include <string>
#include <vector>

namespace matcore::mdslc::codegen {

struct ClosedHostFrontier {
  std::uint64_t id = 0;
  closed_region::Operation::Kind kind;
  closed_region::SourceSite source;
  std::vector<closed_region::SourceSite> helper_calls;
};

// Compiler output, not importable authority. The generated C++ orchestrates
// fixed adapter operations; no source AST, semantic JSON or MLIR is interpreted
// at runtime. An ordinary compiler/linker remains part of the trusted toolchain.
struct ClosedHostEmission {
  std::string declaration;
  std::string implementation;
  std::string symbol;
  std::string source_sha256;
  std::string host_context_sha256;
  std::string semantic_sha256;
  std::vector<ClosedHostFrontier> frontiers;
  std::uint64_t completion_frontier = 0;
};

struct ClosedHostEmissionResult {
  std::optional<ClosedHostEmission> emission;
  std::string error;
  explicit operator bool() const { return emission.has_value(); }
};

// Requires the real-host immutable admission seal, rebuilds and pairs the exact
// semantic witness, and discharges only the bounded synchronous host adapter
// mapping. It does not grant the old inspection dialect generic execution.
ClosedHostEmissionResult emitClosedHostV1(
    const frontend::AuthenticatedClosedRegionEvidence &evidence);

} // namespace matcore::mdslc::codegen
#endif
