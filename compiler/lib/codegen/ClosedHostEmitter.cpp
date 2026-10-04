#include "ClosedHostEmitter.h"
#include <array>
#include <map>
#include <sstream>
#include <utility>

namespace matcore::mdslc::codegen {
namespace {
namespace cr = closed_region;

std::string value(cr::Id id) { return "value_" + std::to_string(id); }
std::string resource(cr::Id id) { return "resource_" + std::to_string(id); }
std::string shape(cr::Id id) { return "shape_" + std::to_string(id); }

// No expression evaluation: all cases are already typed, effect-free semantic
// dimensions produced by admission. In particular unsigned shape comparison is
// not narrowed to an MLIR index, signed host integer or size_t.
std::string dimension(const cr::Dimension &dim) {
  switch (dim.kind) {
  case cr::Dimension::Kind::Literal:
    return "std::uint64_t{" + std::to_string(dim.literal) + "}";
  case cr::Dimension::Kind::ShapeParameter:
    return shape(dim.reference);
  case cr::Dimension::Kind::ValueRows:
    return value(dim.reference) + ".rows()";
  case cr::Dimension::Kind::ValueColumns:
    return value(dim.reference) + ".columns()";
  }
  return {}; // Unreachable after the authoritative verifier.
}

class Emitter {
public:
  Emitter(ClosedHostEmission &result, const ClosedHostDerivedPlan &plan)
      : result_(result) {
    for (const auto &read : plan.forwardedReads())
      forwarding_.emplace(read.read_frontier, read.value);
    for (const auto &pair : plan.strictFusedPairs())
      pairs_.emplace(pair.first_frontier, pair);
  }

  void body(const std::vector<cr::Operation> &operations, unsigned depth) {
    const std::string indent(depth * 2, ' ');
    for (std::size_t index = 0; index < operations.size(); ++index) {
      const auto &op = operations[index];
      const auto frontier = next_frontier_++;
      result_.frontiers.push_back(
          {frontier, op.kind, op.site, op.helper_calls});
      const auto at = std::to_string(frontier);
      std::string call;
      switch (op.kind) {
      case cr::Operation::Kind::Read:
        output_ << indent << "mch::Value " << value(op.result) << ";\n";
        if (const auto forwarded = forwarding_.find(frontier);
            forwarded != forwarding_.end())
          call = "readForwarded(" + at + ", " + resource(op.resource) + ", " +
                 dimension(op.rows) + ", " + dimension(op.columns) + ", " +
                 value(forwarded->second) + ", " + value(op.result) + ")";
        else
          call = "read(" + at + ", " + resource(op.resource) + ", " +
                 dimension(op.rows) + ", " + dimension(op.columns) + ", " +
                 value(op.result) + ")";
        break;
      case cr::Operation::Kind::Gemm:
        if (const auto selected = pairs_.find(frontier); selected != pairs_.end()) {
          const auto &pair = selected->second;
          // The verified canonical plan selects the immediately following
          // operation, but BOTH unchanged original source ledgers survive.
          const auto &consumer = operations.at(++index);
          result_.frontiers.push_back(
              {next_frontier_++, consumer.kind, consumer.site, consumer.helper_calls});
          output_ << indent << "mch::Value " << value(pair.result) << ";\n";
          call = "gemmStrictFusedPair(" + at + ", " +
                 std::to_string(pair.second_frontier) + ", " + value(pair.a) + ", " +
                 value(pair.b) + ", " + value(pair.d) +
                 ", mch::Numeric::strict_f32, mch::Numeric::strict_f32, " +
                 value(pair.result) + ")";
          break;
        }
        output_ << indent << "mch::Value " << value(op.result) << ";\n";
        call = "gemm(" + at + ", " + value(op.lhs) + ", " + value(op.rhs) +
               ", mch::Numeric::" +
               (op.numerical_profile == cr::NumericalProfile::StrictF32
                    ? "strict_f32"
                    : "reassociate_f32") +
               ", " + value(op.result) + ")";
        break;
      case cr::Operation::Kind::Publish:
        call = "publish(" + at + ", " + value(op.lhs) + ", " +
               resource(op.resource) + ")";
        break;
      case cr::Operation::Kind::Observe:
        call = "observe(" + at + ", " + resource(op.resource) + ")";
        break;
      case cr::Operation::Kind::ShapeIf: {
        constexpr std::array<const char *, 6> comparisons{
            "<", "<=", "==", "!=", ">", ">="};
        output_ << indent << "if (" << dimension(op.condition_lhs) << " "
                << comparisons[static_cast<unsigned>(op.comparison)] << " "
                << dimension(op.condition_rhs) << ") {\n";
        body(op.then_body, depth + 1);
        output_ << indent << "} else {\n";
        body(op.else_body, depth + 1);
        output_ << indent << "}\n";
        break;
      }
      }
      if (!call.empty())
        output_ << indent << "if (!session." << call
                << ") return session.status();\n";
    }
  }

  std::string takeBody() {
    result_.completion_frontier = next_frontier_;
    output_ << "  return session.complete(" << next_frontier_ << ");\n";
    return output_.str();
  }

private:
  ClosedHostEmission &result_;
  std::map<std::uint64_t, cr::Id> forwarding_;
  std::map<std::uint64_t, ClosedHostStrictFusedPair> pairs_;
  std::ostringstream output_;
  std::uint64_t next_frontier_ = 1;
};
} // namespace

ClosedHostEmissionResult emitClosedHostV1(
    const frontend::AuthenticatedClosedRegionEvidence &evidence) {
  auto derived = deriveClosedHostPlan(evidence, ClosedHostOptimization::None);
  if (!derived) return {{}, std::move(derived.error)};
  return emitClosedHostV1(evidence, *derived.plan);
}

ClosedHostEmissionResult emitClosedHostV1(
    const frontend::AuthenticatedClosedRegionEvidence &evidence,
    const ClosedHostDerivedPlan &plan) {
  ClosedHostEmissionResult result;
  if (!verifyClosedHostPlan(evidence, plan, result.error)) return result;
  const auto &program = evidence.program();
  const auto &host_identity = evidence.hostContextIdentity();

  ClosedHostEmission emission;
  emission.source_sha256 = program.source_sha256;
  emission.host_context_sha256 = host_identity.substr(7);
  emission.semantic_sha256 = plan.semanticIdentity();
  // A TU may contain multiple selected regions under the same source/context
  // hashes. Bind the complete paired graph, including its selected function and
  // source sites, rather than allowing those different functions to collide.
  emission.plan_sha256 = plan.identity();
  emission.optimization = plan.optimization();
  emission.symbol = "region_" + emission.semantic_sha256 + "_" + emission.plan_sha256;

  // By-value descriptor bindings match the source parameter boundary. Resource
  // validity is deliberately NOT checked here: a late or untaken-arm resource
  // must not prevent an earlier required publication from occurring.
  std::map<std::uint64_t, std::string> parameters;
  const auto &region = program.regions.front();
  for (const auto &item : region.resources)
    parameters.emplace(item.parameter_index,
                       "mch::ResourceView " + resource(item.id));
  for (const auto &item : region.shape_parameters)
    parameters.emplace(item.parameter_index,
                       "std::uint64_t " + shape(item.id));
  std::string signature = "mch::Status " + emission.symbol +
                          "(mch::Session &session";
  for (const auto &[index, parameter] : parameters) {
    (void)index;
    signature += ", " + parameter;
  }
  signature += ") noexcept";
  const std::string prefix =
      "#include <closed_host_v1.h>\n"
      "namespace matcore::mdslc::generated_closed_host_v1 {\n"
      "namespace mch = matcore::mdslc::runtime::closed_host_v1;\n";
  emission.declaration = prefix + signature + ";\n}\n";

  // This private entry takes a pristine invocation object. Reject reuse before
  // any resource operation. A previously failing invocation retains its first
  // failure. A completed/partially successful invocation cannot be continued
  // by accidentally calling a second compiled region into the same session.
  const std::string pristine =
      "  const auto entry = session.status();\n"
      "  if (!entry) return entry;\n"
      "  if (entry.completed || entry.completed_frontier != 0 ||\n"
      "      entry.completed_effect_frontier != 0 || entry.publications != 0 ||\n"
      "      entry.observations != 0) {\n"
      "    auto rejected = entry;\n"
      "    rejected.code = mch::Code::invalid_frontier;\n"
      "    return rejected;\n"
      "  }\n";
  std::string unused_bindings;
  for (const auto &item : region.resources)
    unused_bindings += "  (void)" + resource(item.id) + ";\n";
  for (const auto &item : region.shape_parameters)
    unused_bindings += "  (void)" + shape(item.id) + ";\n";
  Emitter emitter(emission, plan);
  emitter.body(region.body, 1);
  emission.implementation = prefix + signature + " {\n" + pristine + unused_bindings +
                            emitter.takeBody() + "}\n}\n";
  result.emission = std::move(emission);
  return result;
}
} // namespace matcore::mdslc::codegen
