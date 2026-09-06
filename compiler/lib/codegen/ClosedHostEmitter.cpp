#include "ClosedHostEmitter.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <map>
#include <sstream>
#include <utility>

namespace matcore::mdslc::codegen {
namespace {
namespace cr = closed_region;

bool digest(const std::string &value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

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
  explicit Emitter(ClosedHostEmission &result) : result_(result) {}

  void body(const std::vector<cr::Operation> &operations, unsigned depth) {
    const std::string indent(depth * 2, ' ');
    for (const auto &op : operations) {
      const auto frontier = next_frontier_++;
      result_.frontiers.push_back(
          {frontier, op.kind, op.site, op.helper_calls});
      const auto at = std::to_string(frontier);
      std::string call;
      switch (op.kind) {
      case cr::Operation::Kind::Read:
        output_ << indent << "mch::Value " << value(op.result) << ";\n";
        call = "read(" + at + ", " + resource(op.resource) + ", " +
               dimension(op.rows) + ", " + dimension(op.columns) + ", " +
               value(op.result) + ")";
        break;
      case cr::Operation::Kind::Gemm:
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
  std::ostringstream output_;
  std::uint64_t next_frontier_ = 1;
};
} // namespace

ClosedHostEmissionResult emitClosedHostV1(
    const frontend::AuthenticatedClosedRegionEvidence &evidence) {
  ClosedHostEmissionResult result;
  if (!evidence.hasHostContext()) {
    result.error = "closed host emission requires authenticated real host context";
    return result;
  }
  const auto &program = evidence.program();
  const auto &host_identity = evidence.hostContextIdentity();
  if (program.regions.size() != 1 || !digest(program.source_sha256) ||
      !host_identity.starts_with("sha256:") ||
      !digest(host_identity.substr(7))) {
    result.error = "closed host emission requires one region and complete identities";
    return result;
  }
  mlir::MLIRContext context;
  auto witness = cr::buildModule(program, context);
  if (!witness) {
    result.error = witness.error;
    return result;
  }
  if (!frontend::verifyClosedRegionMatchesEvidence(
          evidence, *witness.module, result.error))
    return result;

  ClosedHostEmission emission;
  emission.source_sha256 = program.source_sha256;
  emission.host_context_sha256 = host_identity.substr(7);
  const auto semantic_text = cr::printModule(*witness.module);
  emission.semantic_sha256 = llvm::toHex(
      llvm::SHA256::hash(llvm::arrayRefFromStringRef(semantic_text)), true);
  // A TU may contain multiple selected regions under the same source/context
  // hashes. Bind the complete paired graph, including its selected function and
  // source sites, rather than allowing those different functions to collide.
  emission.symbol = "region_" + emission.semantic_sha256;

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
      "#include \"closed_host_v1.h\"\n"
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
  Emitter emitter(emission);
  emitter.body(region.body, 1);
  emission.implementation = prefix + signature + " {\n" + pristine + unused_bindings +
                            emitter.takeBody() + "}\n}\n";
  result.emission = std::move(emission);
  return result;
}
} // namespace matcore::mdslc::codegen
