#pragma once

#include "../frontend/ClosedRegionHostInputs.h"

#include <string>

namespace matcore::mdslc::codegen::detail {

// A compiler implementation utility, not an admission or execution issuer.
// Compiles the original main file without text replacement: ordinary C++ source
// coordinates and preprocessing remain unchanged. Every compiler query uses
// the closed admission snapshot and fails on uncaptured dependency lookups.
// Returns host LLVM IR in memory; publishes no executable or filesystem output.
bool compileFrozenHostToLLVM(
    const frontend::closed_region_host::HostInputSnapshot &snapshot,
    std::string &llvm_ir, std::string &error);

} // namespace matcore::mdslc::codegen::detail
