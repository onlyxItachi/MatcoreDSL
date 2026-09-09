#pragma once
#include "ClosedRegionAdmission.h"
#include "ClosedRegionHostInputs.h"
#include <map>
#include <optional>
#include <vector>

namespace matcore::mdslc::frontend::detail {
struct ProgramRegionOwner {
  std::size_t owner;
  ClosedRegionEntryBinding binding;
};
using ProgramRegionOwners = std::map<std::string, ProgramRegionOwner>;
struct ProgramHostInterface {
  unsigned main_definitions = 0;
  std::vector<std::string> declarations;
  std::vector<std::string> foreign_definitions;
  std::string preprocessing;
  bool operator==(const ProgramHostInterface &) const = default;
};
struct ProgramHostInterfaceResult {
  std::shared_ptr<const closed_region_host::HostInputSnapshot> snapshot;
  ProgramHostInterface interface;
  std::string error;
  explicit operator bool() const { return snapshot != nullptr && error.empty(); }
};
// Captures a normal host TU, or inspects an already sealed region TU snapshot.
// Both paths replay the same source interface before LLVM can consume it.
// Neither the source binding nor this result is a serialized module authority.
ProgramHostInterfaceResult inspectProgramHost(
    const Options &, const std::string &working_directory,
    const ExperimentalRegionHeaders &, const ProgramRegionOwners &, std::size_t unit,
    std::shared_ptr<const closed_region_host::HostInputSnapshot> snapshot = {});
} // namespace matcore::mdslc::frontend::detail
