#ifndef MATCORE_MDSLC_CLOSED_HOST_DERIVED_PLAN_H
#define MATCORE_MDSLC_CLOSED_HOST_DERIVED_PLAN_H

#include "../frontend/ClosedRegionAdmission.h"
#include <memory>
#include <optional>

namespace matcore::mdslc::codegen {

// Private compilation choice, orthogonal to candidate/target selection. None
// retains the original read-at-frontier snapshot realization.
enum class ClosedHostOptimization { None, PublicationReadForwarding };
const char *closedHostOptimizationName(ClosedHostOptimization) noexcept;

struct ClosedHostForwardedRead {
  std::uint64_t read_frontier = 0;
  // This dominating successful publication is the checked resource version.
  // Every later publication invalidates it, even to a distinct descriptor.
  std::uint64_t publication_frontier = 0;
  closed_region::Id resource = 0;
  closed_region::Id value = 0;
  bool operator==(const ClosedHostForwardedRead &) const = default;
};
struct ClosedHostDerivedPlanResult;

// In-process immutable derivation only. Neither an editable Program, serialized
// attributes, an untrusted proposal nor a successful bool verifier can construct
// one. The payload owns its original admission seal. Consumption still replays
// exact source/witness pairing and recomputes bounded legality; it does not relax
// the original paired-witness contract or discharge a runtime predicate.
class ClosedHostDerivedPlan {
public:
  bool valid() const noexcept { return static_cast<bool>(payload_); }
  // Moved-from queries diagnose invalid use rather than dereference a payload.
  ClosedHostOptimization optimization() const;
  const std::vector<ClosedHostForwardedRead> &forwardedReads() const;
  const std::string &identity() const;
  const std::string &semanticIdentity() const;

private:
  struct Payload;
  explicit ClosedHostDerivedPlan(std::shared_ptr<const Payload>);
  std::shared_ptr<const Payload> payload_;
  friend ClosedHostDerivedPlanResult deriveClosedHostPlan(
      const frontend::AuthenticatedClosedRegionEvidence &, ClosedHostOptimization);
  friend bool verifyClosedHostPlan(
      const frontend::AuthenticatedClosedRegionEvidence &,
      const ClosedHostDerivedPlan &, std::string &);
};

struct ClosedHostDerivedPlanResult {
  std::optional<ClosedHostDerivedPlan> plan;
  std::string error;
  explicit operator bool() const { return plan.has_value(); }
};

ClosedHostDerivedPlanResult deriveClosedHostPlan(
    const frontend::AuthenticatedClosedRegionEvidence &,
    ClosedHostOptimization = ClosedHostOptimization::None);
bool verifyClosedHostPlan(const frontend::AuthenticatedClosedRegionEvidence &,
                          const ClosedHostDerivedPlan &, std::string &error);

// Untrusted diagnostic/test proposal. Complete canonical-list equality is
// required, including count/order and original frontier numbering. This returns
// no seal or executable authority: the issuer always recomputes its own plan.
bool verifyClosedHostForwardingProposal(
    const frontend::AuthenticatedClosedRegionEvidence &, ClosedHostOptimization,
    const std::vector<ClosedHostForwardedRead> &, std::string &error);

} // namespace matcore::mdslc::codegen
#endif
