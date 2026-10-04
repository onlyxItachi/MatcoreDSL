#include "ClosedHostDerivedPlan.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace matcore::mdslc::codegen {
namespace {
namespace cr = closed_region;
constexpr unsigned plan_revision = 2;
// Binds this derivation to the reviewed strict pair issuer/adapter contract.
// It does not encode that issuer's panel width into semantic Program records.
constexpr unsigned strict_pair_contract_revision = 1;

std::string hash(const std::string &bytes) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes)), true);
}
bool digest(const std::string &value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
struct Snapshot {
  std::string semantic_sha256;
  std::string host_context_identity;
};

bool authenticatedSnapshot(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    Snapshot &snapshot, std::string &error) {
  // This query also rejects a moved-from evidence before payload access.
  if (!evidence.hasHostContext()) {
    error = "closed host derivation requires authenticated real host context";
    return false;
  }
  const auto &program = evidence.program();
  const auto &host = evidence.hostContextIdentity();
  if (program.regions.size() != 1 || !digest(program.source_sha256) ||
      !host.starts_with("sha256:") || !digest(host.substr(7))) {
    error = "closed host derivation requires one region and complete identities";
    return false;
  }
  mlir::MLIRContext context;
  auto witness = cr::buildModule(program, context);
  if (!witness) { error = witness.error; return false; }
  if (!frontend::verifyClosedRegionMatchesEvidence(evidence, *witness.module, error))
    return false;
  snapshot.semantic_sha256 = hash(cr::printModule(*witness.module));
  snapshot.host_context_identity = host;
  return true;
}

// One current binding is sufficient: all descriptors MAY alias. Reads, GEMMs
// and owning observations do not write host resources. No descriptor inequality
// supplies disjointness. A branch is a conservative entry/join barrier; its own
// straight-line sub-body may establish a fresh local dominating publication.
class Derivation {
public:
  explicit Derivation(ClosedHostOptimization optimization)
      : enabled_(optimization == ClosedHostOptimization::PublicationReadForwarding) {}
  void body(const std::vector<cr::Operation> &operations) {
    for (const auto &operation : operations) {
      const auto frontier = next_frontier_++;
      switch (operation.kind) {
      case cr::Operation::Kind::Read:
        if (enabled_ && current_ && current_->resource == operation.resource) {
          auto forwarding = *current_;
          forwarding.read_frontier = frontier;
          reads_.push_back(forwarding);
        }
        break;
      case cr::Operation::Kind::Publish:
        current_ = ClosedHostForwardedRead{
            0, frontier, operation.resource, operation.lhs};
        break;
      case cr::Operation::Kind::ShapeIf:
        current_.reset();
        body(operation.then_body);
        current_.reset();
        body(operation.else_body);
        current_.reset();
        break;
      case cr::Operation::Kind::Gemm:
      case cr::Operation::Kind::Observe:
        break;
      }
    }
  }
  std::vector<ClosedHostForwardedRead> take() { return std::move(reads_); }
private:
  bool enabled_;
  std::uint64_t next_frontier_ = 1;
  std::optional<ClosedHostForwardedRead> current_;
  std::vector<ClosedHostForwardedRead> reads_;
};

std::vector<ClosedHostForwardedRead> deriveReads(
    const cr::Program &program, ClosedHostOptimization optimization) {
  Derivation derivation(optimization);
  derivation.body(program.regions.front().body);
  return derivation.take();
}

using UseCounts = std::map<cr::Id, std::uint64_t>;
void dimensionUse(const cr::Dimension &dimension, UseCounts &uses) {
  if (dimension.kind == cr::Dimension::Kind::ValueRows ||
      dimension.kind == cr::Dimension::Kind::ValueColumns)
    ++uses[dimension.reference];
}
void countUses(const std::vector<cr::Operation> &body, UseCounts &uses) {
  for (const auto &operation : body) {
    switch (operation.kind) {
    case cr::Operation::Kind::Read:
      dimensionUse(operation.rows, uses);
      dimensionUse(operation.columns, uses);
      break;
    case cr::Operation::Kind::Gemm:
      ++uses[operation.lhs];
      ++uses[operation.rhs];
      break;
    case cr::Operation::Kind::Publish:
      ++uses[operation.lhs];
      break;
    case cr::Operation::Kind::Observe:
      break;
    case cr::Operation::Kind::ShapeIf:
      dimensionUse(operation.condition_lhs, uses);
      dimensionUse(operation.condition_rhs, uses);
      countUses(operation.then_body, uses);
      countUses(operation.else_body, uses);
      break;
    }
  }
}

class PairDerivation {
public:
  explicit PairDerivation(const UseCounts &uses) : uses_(uses) {}
  void body(const std::vector<cr::Operation> &operations,
            std::map<cr::Id, std::uint64_t> reads = {}) {
    for (std::size_t index = 0; index < operations.size(); ++index) {
      const auto &operation = operations[index];
      const auto frontier = next_frontier_++;
      if (operation.kind == cr::Operation::Kind::Read)
        reads.emplace(operation.result, frontier);
      if (operation.kind == cr::Operation::Kind::ShapeIf) {
        // Each arm inherits only dominating immutable snapshots. Sibling-local
        // definitions never escape; both arms still consume original IDs.
        body(operation.then_body, reads);
        body(operation.else_body, reads);
      }
      if (index + 1 == operations.size() ||
          operation.kind != cr::Operation::Kind::Gemm ||
          operation.numerical_profile != cr::NumericalProfile::StrictF32)
        continue;
      const auto &consumer = operations[index + 1];
      const auto use = uses_.find(operation.result);
      if (consumer.kind != cr::Operation::Kind::Gemm ||
          consumer.numerical_profile != cr::NumericalProfile::StrictF32 ||
          consumer.lhs != operation.result || use == uses_.end() || use->second != 1 ||
          !reads.contains(operation.lhs) || !reads.contains(operation.rhs) ||
          !reads.contains(consumer.rhs))
        continue;
      pairs_.push_back({frontier, next_frontier_++, operation.lhs, operation.rhs,
                        consumer.rhs, operation.result, consumer.result,
                        reads.at(operation.lhs), reads.at(operation.rhs),
                        reads.at(consumer.rhs)});
      ++index; // Nonoverlapping, exactly adjacent windows only.
    }
  }
  std::vector<ClosedHostStrictFusedPair> take() { return std::move(pairs_); }
private:
  const UseCounts &uses_;
  std::uint64_t next_frontier_ = 1;
  std::vector<ClosedHostStrictFusedPair> pairs_;
};

std::vector<ClosedHostStrictFusedPair> derivePairs(
    const cr::Program &program, ClosedHostOptimization optimization) {
  if (optimization != ClosedHostOptimization::StrictFusedPair) return {};
  UseCounts uses;
  countUses(program.regions.front().body, uses);
  PairDerivation derivation(uses);
  derivation.body(program.regions.front().body);
  return derivation.take();
}

std::string planIdentity(const Snapshot &snapshot, ClosedHostOptimization optimization,
                         const std::vector<ClosedHostForwardedRead> &reads,
                         const std::vector<ClosedHostStrictFusedPair> &pairs) {
  std::ostringstream bytes;
  bytes << "matcore.closed_host.derived_plan:" << plan_revision << ':'
        << closedHostOptimizationName(optimization) << ':'
        << snapshot.semantic_sha256 << ':' << snapshot.host_context_identity
        << ':' << reads.size();
  for (const auto &read : reads)
    bytes << ':' << read.read_frontier << ':' << read.publication_frontier
          << ':' << read.resource << ':' << read.value;
  bytes << ':' << strict_pair_contract_revision << ':' << pairs.size();
  for (const auto &pair : pairs)
    bytes << ':' << pair.first_frontier << ':' << pair.second_frontier
          << ':' << pair.a << ':' << pair.b << ':' << pair.d
          << ':' << pair.intermediate << ':' << pair.result
          << ':' << pair.a_read_frontier << ':' << pair.b_read_frontier
          << ':' << pair.d_read_frontier;
  return hash(bytes.str());
}
} // namespace

struct ClosedHostDerivedPlan::Payload {
  frontend::AuthenticatedClosedRegionEvidence evidence;
  unsigned revision;
  ClosedHostOptimization optimization;
  Snapshot snapshot;
  std::vector<ClosedHostForwardedRead> reads;
  std::vector<ClosedHostStrictFusedPair> pairs;
  std::string identity;
};

const char *closedHostOptimizationName(ClosedHostOptimization optimization) noexcept {
  switch (optimization) {
  case ClosedHostOptimization::None: return "none";
  case ClosedHostOptimization::PublicationReadForwarding: return "publication-read-forwarding";
  case ClosedHostOptimization::StrictFusedPair: return "strict-fused-pair";
  }
  return nullptr;
}
ClosedHostDerivedPlan::ClosedHostDerivedPlan(std::shared_ptr<const Payload> payload)
    : payload_(std::move(payload)) {}
ClosedHostOptimization ClosedHostDerivedPlan::optimization() const {
  if (!payload_) throw std::logic_error("moved-from closed host derivation has no authority");
  return payload_->optimization;
}
const std::vector<ClosedHostForwardedRead> &ClosedHostDerivedPlan::forwardedReads() const {
  if (!payload_) throw std::logic_error("moved-from closed host derivation has no authority");
  return payload_->reads;
}
const std::vector<ClosedHostStrictFusedPair> &ClosedHostDerivedPlan::strictFusedPairs() const {
  if (!payload_) throw std::logic_error("moved-from closed host derivation has no authority");
  return payload_->pairs;
}
const std::string &ClosedHostDerivedPlan::identity() const {
  if (!payload_) throw std::logic_error("moved-from closed host derivation has no authority");
  return payload_->identity;
}
const std::string &ClosedHostDerivedPlan::semanticIdentity() const {
  if (!payload_) throw std::logic_error("moved-from closed host derivation has no authority");
  return payload_->snapshot.semantic_sha256;
}

ClosedHostDerivedPlanResult deriveClosedHostPlan(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    ClosedHostOptimization optimization) {
  ClosedHostDerivedPlanResult result;
  if (!closedHostOptimizationName(optimization)) {
    result.error = "unknown closed host optimization";
    return result;
  }
  Snapshot snapshot;
  if (!authenticatedSnapshot(evidence, snapshot, result.error)) return result;
  auto reads = deriveReads(evidence.program(), optimization);
  auto pairs = derivePairs(evidence.program(), optimization);
  if (optimization == ClosedHostOptimization::StrictFusedPair && pairs.empty()) {
    result.error = "strict-fused-pair requires an eligible adjacent pure strict GEMM pair with read inputs and a private single-use intermediate";
    return result;
  }
  auto identity = planIdentity(snapshot, optimization, reads, pairs);
  result.plan = ClosedHostDerivedPlan(std::make_shared<const ClosedHostDerivedPlan::Payload>(
      ClosedHostDerivedPlan::Payload{evidence, plan_revision, optimization,
                                   std::move(snapshot), std::move(reads),
                                   std::move(pairs), std::move(identity)}));
  return result;
}

bool verifyClosedHostPlan(const frontend::AuthenticatedClosedRegionEvidence &evidence,
                          const ClosedHostDerivedPlan &plan, std::string &error) {
  if (!plan.payload_) {
    error = "moved-from closed host derivation has no authority";
    return false;
  }
  const auto &payload = *plan.payload_;
  if (payload.revision != plan_revision ||
      !closedHostOptimizationName(payload.optimization)) {
    error = "unsupported closed host derivation revision or optimization";
    return false;
  }
  Snapshot snapshot;
  if (!authenticatedSnapshot(evidence, snapshot, error)) return false;
  const auto reads = deriveReads(evidence.program(), payload.optimization);
  const auto pairs = derivePairs(evidence.program(), payload.optimization);
  if (!payload.evidence.hasHostContext() ||
      snapshot.semantic_sha256 != payload.snapshot.semantic_sha256 ||
      snapshot.host_context_identity != payload.snapshot.host_context_identity ||
      payload.evidence.hostContextIdentity() != snapshot.host_context_identity ||
      reads != payload.reads ||
      pairs != payload.pairs ||
      (payload.optimization == ClosedHostOptimization::StrictFusedPair && pairs.empty()) ||
      planIdentity(snapshot, payload.optimization, reads, pairs) != payload.identity) {
    error = "closed host derivation does not match authenticated source or bounded legality";
    return false;
  }
  return true;
}

bool verifyClosedHostForwardingProposal(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    ClosedHostOptimization optimization,
    const std::vector<ClosedHostForwardedRead> &proposal, std::string &error) {
  if (!closedHostOptimizationName(optimization) ||
      optimization == ClosedHostOptimization::StrictFusedPair) {
    error = "unknown closed host optimization";
    return false;
  }
  Snapshot snapshot;
  if (!authenticatedSnapshot(evidence, snapshot, error)) return false;
  if (deriveReads(evidence.program(), optimization) != proposal) {
    error = "forwarding proposal differs from the complete bounded source derivation";
    return false;
  }
  return true;
}

bool verifyClosedHostFusedPairProposal(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    ClosedHostOptimization optimization,
    const std::vector<ClosedHostStrictFusedPair> &proposal, std::string &error) {
  if (!closedHostOptimizationName(optimization) ||
      optimization == ClosedHostOptimization::PublicationReadForwarding) {
    error = "unknown or incompatible closed host pair optimization";
    return false;
  }
  Snapshot snapshot;
  if (!authenticatedSnapshot(evidence, snapshot, error)) return false;
  const auto pairs = derivePairs(evidence.program(), optimization);
  if ((optimization == ClosedHostOptimization::StrictFusedPair && pairs.empty()) ||
      pairs != proposal) {
    error = "strict pair proposal differs from the complete bounded source derivation";
    return false;
  }
  return true;
}
} // namespace matcore::mdslc::codegen
