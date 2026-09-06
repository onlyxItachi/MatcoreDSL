#ifndef MATCORE_MDSLC_MLIR_CLOSED_REGION_H
#define MATCORE_MDSLC_MLIR_CLOSED_REGION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <cstdint>
#include <string>
#include <vector>

namespace matcore::mdslc::closed_region {

// Private, transient admission records. These are neither a serialized IR nor
// execution/source authority. Only the frontend's immutable native seal can
// authenticate their origin. No Clang objects cross this boundary.
using Id = std::uint64_t;
struct SourceSite {
  std::uint64_t offset = 0;
  std::uint64_t length = 0;
  std::uint64_t line = 0;
  std::uint64_t column = 0;
};
struct Dimension {
  enum class Kind { Literal, ShapeParameter, ValueRows, ValueColumns };
  Kind kind = Kind::Literal;
  std::uint64_t literal = 0;
  Id reference = 0;
};
enum class NumericalProfile { StrictF32, ReassociateF32 };
enum class Comparison { Less, LessEqual, Equal, NotEqual, Greater, GreaterEqual };
struct Operation {
  enum class Kind { Read, Gemm, Publish, Observe, ShapeIf };
  Kind kind = Kind::Read;
  SourceSite site;
  // A helper's own source site remains `site`; its admitted call sites are
  // retained separately, outermost first. Helpers are expanded, not executed.
  std::vector<SourceSite> helper_calls;
  Id result = 0;
  Id resource = 0;
  Id lhs = 0;
  Id rhs = 0;
  Dimension rows;
  Dimension columns;
  NumericalProfile numerical_profile = NumericalProfile::StrictF32;
  Comparison comparison = Comparison::Equal;
  Dimension condition_lhs;
  Dimension condition_rhs;
  std::vector<Operation> then_body;
  std::vector<Operation> else_body;
};
struct Resource {
  Id id = 0;
  std::string name;
  std::uint64_t parameter_index = 0;
};
struct ShapeParameter {
  Id id = 0;
  std::string name;
  std::uint64_t parameter_index = 0;
};
struct Region {
  std::string name;
  SourceSite site;
  std::vector<Resource> resources;
  std::vector<ShapeParameter> shape_parameters;
  std::vector<Operation> body;
};
struct Program {
  std::string source_identity;
  std::string source_sha256;
  std::string header_sha256;
  std::string compiler_identity;
  std::vector<Region> regions;
};
struct Result {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string error;
  explicit operator bool() const { return static_cast<bool>(module); }
};

// Read denotes immutable contents at the ordered resource frontier, not a
// live pointer view. All resources MAY alias. A global resource-state token
// conservatively orders publications and subsequent reads; it is not storage
// identity, noalias evidence, an allocation strategy or a transaction.
// Reads/publications request a bounded dense row-major f32 resource view; the
// adapter's layout/capacity compatibility is required, never proved here.
// Every GEMM retains an ordered check even when its result is dead. Each
// result rounds to f32; within-GEMM reassociation never authorizes cross-GEMM
// reassociation. Publication/observation remain ordered, potentially fallible
// effects, with partial mutation and no rollback explicitly possible.
bool verifyProgram(const Program &program, std::string &error);
void registerDialects(mlir::MLIRContext &context);
Result buildModule(const Program &program, mlir::MLIRContext &context);
bool verifyModule(mlir::ModuleOp module, std::string &error);
// Exact untransformed admission seam only. This authenticates nothing by
// itself: callers must obtain `program` from sealed/re-admitted source.
bool verifyModuleMatchesProgram(const Program &program, mlir::ModuleOp module,
                                std::string &error);
std::string printModule(mlir::ModuleOp module);

} // namespace matcore::mdslc::closed_region
#endif
