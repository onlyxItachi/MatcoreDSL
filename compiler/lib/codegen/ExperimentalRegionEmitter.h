#pragma once

#include "ClosedHostEmitter.h"

namespace matcore::mdslc::codegen {

// Implementation policy belongs to compilation/dispatch, not mathematical IR
// or source syntax. Forced unsupported candidates fail rather than fall back.
enum class ClosedCpuPolicy {
  Automatic, NativeStrict, GeneratedStrict, ExistingNative, OpenBLAS,
  GeneratedReassociate
};

struct ExperimentalRegionEmission {
  ClosedHostEmission contract;
  std::string helper_cpp;
  std::string helper_symbol;
  std::string host_symbol;
  std::vector<std::string> retired_value_helpers;
};
struct ExperimentalRegionEmissionResult {
  std::optional<ExperimentalRegionEmission> emission;
  std::string error;
  explicit operator bool() const { return emission.has_value(); }
};

// Requires the new named-function admission seal and its re-admitted binding.
// Emits a compiler-owned implementation TU, not a replacement host source.
// Clang compiles the original frozen host; a separate ABI-checked LLVM thunk
// connects only its authenticated function to this generated implementation.
ExperimentalRegionEmissionResult emitExperimentalRegion(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    ClosedCpuPolicy policy = ClosedCpuPolicy::Automatic);

} // namespace matcore::mdslc::codegen
