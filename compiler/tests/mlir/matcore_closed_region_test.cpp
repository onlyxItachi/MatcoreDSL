#include "MatcoreClosedRegion.h"
#include "MatcoreCpuRuntimeLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include <functional>
#include <iostream>
#include <limits>

namespace {
namespace cr = matcore::mdslc::closed_region;
unsigned checks = 0, failures = 0;
void check(bool value, const std::string &message) {
  ++checks;
  if (!value) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}
cr::Dimension literal(std::uint64_t value) {
  return {cr::Dimension::Kind::Literal, value, 0};
}
cr::Operation operation(cr::Operation::Kind kind, unsigned offset) {
  cr::Operation op;
  op.kind = kind;
  op.site = {offset, 1, offset + 1, 1};
  return op;
}
cr::Operation read(cr::Id result, cr::Id resource, unsigned rows, unsigned columns) {
  auto op = operation(cr::Operation::Kind::Read, result);
  op.result = result;
  op.resource = resource;
  op.rows = literal(rows);
  op.columns = literal(columns);
  return op;
}
cr::Operation gemm(cr::Id result, cr::Id lhs, cr::Id rhs) {
  auto op = operation(cr::Operation::Kind::Gemm, result);
  op.result = result;
  op.lhs = lhs;
  op.rhs = rhs;
  return op;
}
cr::Operation publish(cr::Id value, cr::Id resource) {
  auto op = operation(cr::Operation::Kind::Publish, value + 10);
  op.lhs = value;
  op.resource = resource;
  return op;
}
cr::Program program() {
  cr::Program p;
  p.source_identity = "synthetic_semantic_unit_test_not_source_authenticated";
  p.source_sha256 = std::string(64, 'a');
  p.header_sha256 = std::string(64, 'b');
  p.compiler_identity = "synthetic_model_test";
  cr::Region r;
  r.name = "region";
  r.site = {0, 100, 1, 1};
  r.resources = {{1, "A", 0}, {2, "B", 1}, {3, "C", 2}, {4, "D", 3}, {5, "E", 4}};
  r.shape_parameters = {{1, "m", 5}, {2, "n", 6}, {3, "k", 7}};
  r.body = {read(1, 1, 2, 3), read(2, 2, 3, 2), gemm(3, 1, 2), publish(3, 3),
            read(4, 4, 2, 2), gemm(5, 4, 3), publish(5, 5)};
  p.regions.push_back(r);
  return p;
}
llvm::SmallVector<mlir::Operation *> all(mlir::ModuleOp module, llvm::StringRef suffix) {
  llvm::SmallVector<mlir::Operation *> result;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == ("mdsl_admission." + suffix).str())
      result.push_back(op);
  });
  return result;
}
void reject(const cr::Program &p, mlir::ModuleOp original,
            const std::function<void(mlir::ModuleOp)> &mutation,
            const std::string &message, bool standalone = true) {
  mlir::OwningOpRef<mlir::ModuleOp> changed = original.clone();
  mutation(*changed);
  std::string error;
  mlir::ScopedDiagnosticHandler silence(changed->getContext(), [](mlir::Diagnostic &) { return mlir::success(); });
  if (standalone)
    check(!cr::verifyModule(*changed, error) && !error.empty(), message + " self-consistency rejects");
  check(!cr::verifyModuleMatchesProgram(p, *changed, error) && !error.empty(), message + " pairing rejects");
}
void modelReject(cr::Program p, const std::function<void(cr::Program &)> &mutation,
                 const std::string &message) {
  mutation(p);
  std::string error;
  check(!cr::verifyProgram(p, error) && !error.empty(), message);
}
} // namespace

int main() {
  mlir::MLIRContext context;
  cr::registerDialects(context);
  auto p = program();
  auto built = cr::buildModule(p, context);
  check(static_cast<bool>(built), "static RHS region builds: " + built.error);
  if (!built)
    return 1;
  std::string error;
  check(cr::verifyModuleMatchesProgram(p, *built.module, error), "synthetic record pairs, without authenticating source");
  auto printed = cr::printModule(*built.module);
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(printed, &context);
  check(parsed && cr::verifyModuleMatchesProgram(p, *parsed, error), "registered MLIR parse/print preserves exact pairing");
  check(all(*built.module, "check_gemm").size() == 2, "each GEMM has a retained checked-failure frontier");
  check(printed.find("all_resources_may_alias") != std::string::npos, "resource names never assert noalias");
  check(printed.find("preserve_live_immutable_values") != std::string::npos, "publication retains old-value preservation obligation");
  std::vector<matcore::mdslc::mlir_lowering::CpuRuntimeDispatchRecordV1> records(1);
  records[0].site_id = "must_be_cleared_on_rejection";
  check(!matcore::mdslc::mlir_lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
            *built.module, records, error) && records.empty() && !error.empty(),
        "actual legacy CPU lowerer rejects private admission and clears stale records");
  mlir::OwningOpRef<mlir::ModuleOp> forged = built.module->clone();
  (*forged)->removeAttr("mdsl_admission.authority");
  (*forged)->setAttr("mdsl.producer", mlir::StringAttr::get(&context, "clang-libtooling-v1"));
  (*forged)->setAttr("mdsl.capability", mlir::StringAttr::get(&context, "validated_cpu"));
  (*forged)->setAttr("mdsl.retry_safe", mlir::BoolAttr::get(&context, true));
  (*forged)->setAttr("mdsl.target", mlir::StringAttr::get(&context, "cpu"));
  records.emplace_back();
  check(!matcore::mdslc::mlir_lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
            *forged, records, error) && records.empty() && !error.empty(),
        "forged producer/target/retry labels cannot acquire actual CPU authority");

  auto lhs = p;
  lhs.regions[0].body[5].lhs = 3;
  lhs.regions[0].body[5].rhs = 4;
  auto lhsModule = cr::buildModule(lhs, context);
  check(lhsModule && cr::verifyModuleMatchesProgram(lhs, *lhsModule.module, error), "lhs carried value also builds");
  auto reassociate = p;
  reassociate.regions[0].body[2].numerical_profile = cr::NumericalProfile::ReassociateF32;
  auto reassociateModule = cr::buildModule(reassociate, context);
  check(reassociateModule && cr::verifyModuleMatchesProgram(reassociate, *reassociateModule.module, error),
        "within-GEMM reassociation is an explicit per-operation admission profile");
  auto oldValue = p;
  oldValue.regions[0].body[3].resource = 1;
  oldValue.regions[0].body[5].rhs = 1;
  oldValue.regions[0].body[4].columns = literal(2);
  // A is 2x3, so D(2x2)*old A produces 2x3 after A's resource was published.
  auto oldModule = cr::buildModule(oldValue, context);
  check(oldModule && cr::verifyModuleMatchesProgram(oldValue, *oldModule.module, error), "old immutable value survives a publication to its possible backing resource");

  auto dynamic = p;
  dynamic.regions[0].body[0].rows = {cr::Dimension::Kind::ShapeParameter, 0, 1};
  dynamic.regions[0].body[0].columns = {cr::Dimension::Kind::ShapeParameter, 0, 3};
  dynamic.regions[0].body[1].rows = {cr::Dimension::Kind::ShapeParameter, 0, 3};
  dynamic.regions[0].body[1].columns = {cr::Dimension::Kind::ShapeParameter, 0, 2};
  dynamic.regions[0].body[4].columns = {cr::Dimension::Kind::ValueRows, 0, 3};
  auto dynamicModule = cr::buildModule(dynamic, context);
  check(dynamicModule && cr::verifyModuleMatchesProgram(dynamic, *dynamicModule.module, error), "dynamic shape relationships and value-axis queries build without discharging checks");

  auto branched = p;
  auto branch = operation(cr::Operation::Kind::ShapeIf, 40);
  branch.condition_lhs = {cr::Dimension::Kind::ShapeParameter, 0, 1};
  branch.condition_rhs = literal(4);
  branch.comparison = cr::Comparison::Less;
  auto observation = operation(cr::Operation::Kind::Observe, 41);
  observation.resource = 3;
  branch.then_body = {publish(3, 3), observation};
  branch.else_body = {publish(3, 4)};
  branched.regions[0].body.insert(branched.regions[0].body.begin() + 4, branch);
  auto branchModule = cr::buildModule(branched, context);
  check(branchModule && cr::verifyModuleMatchesProgram(branched, *branchModule.module, error), "both shape branches retain ordered effects and conservative join");
  if (branchModule) {
    auto function = *branchModule.module->getOps<mlir::func::FuncOp>().begin();
    check(function.getArgument(5).getType().isSignlessInteger(64),
          "symbolic C++ unsigned-long-long extent retains all 64 bits, not target index");
    unsigned unsignedComparisons = 0;
    branchModule.module->walk([&](mlir::arith::CmpIOp cmp) {
      if (cmp.getPredicate() == mlir::arith::CmpIPredicate::ult &&
          cmp.getLhs().getType().isSignlessInteger(64))
        ++unsignedComparisons;
    });
    check(unsignedComparisons == 1, "runtime shape branch keeps unsigned comparison including high-bit values");
    reject(branched, *branchModule.module, [](mlir::ModuleOp m) {
      m.walk([&](mlir::arith::CmpIOp cmp) { cmp.setPredicate(mlir::arith::CmpIPredicate::slt); });
    }, "signed comparison cannot replace unsigned C++ shape semantics");
    reject(branched, *branchModule.module, [](mlir::ModuleOp m) {
      all(m, "shape_if")[0]->setAttr("then_epoch", mlir::StringAttr::get(m.getContext(), "entry"));
    }, "forged branch resource join");
    reject(branched, *branchModule.module, [](mlir::ModuleOp m) {
      auto *observe = all(m, "observe")[0];
      observe->getResult(0).replaceAllUsesWith(observe->getOperand(0));
      observe->erase();
    }, "dropped guaranteed observation", false);
  }

  reject(p, *built.module, [](mlir::ModuleOp m) {
    m->setAttr("mdsl_admission.authority", mlir::StringAttr::get(m.getContext(), "execution"));
  }, "forged execution authority");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    m->setAttr("mdsl_admission.aliasing", mlir::StringAttr::get(m.getContext(), "noalias"));
  }, "unproved resource disjointness");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "read")[2]->setAttr("resource_epoch", mlir::StringAttr::get(m.getContext(), "entry"));
  }, "stale late-read epoch");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    auto *read = all(m, "read")[2];
    read->setOperand(0, all(m, "begin")[0]->getResult(0));
  }, "read bypasses publication order");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "check_gemm")[1]->setAttr("evidence", mlir::StringAttr::get(m.getContext(), "discharged"));
  }, "invented guard discharge");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    auto guards = all(m, "check_gemm");
    guards[1]->setOperand(0, guards[0]->getResult(0));
  }, "second validation cannot bypass first publication and late read");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "publish")[0]->setAttr("failure", mlir::StringAttr::get(m.getContext(), "atomic_rollback"));
  }, "invented atomic publication");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "gemm")[1]->setAttr("cross_operation_reassociation", mlir::StringAttr::get(m.getContext(), "permitted"));
  }, "cross-GEMM numerical reassociation");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "gemm")[1]->setAttr("operation_boundary_rounding", mlir::StringAttr::get(m.getContext(), "f64"));
  }, "intermediate f32 rounding cannot disappear");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "gemm")[1]->setAttr("signed_zero", mlir::StringAttr::get(m.getContext(), "ignored"));
  }, "signed zero is not silently ignored");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "check_gemm")[0]->setAttr("fp_status_traps", mlir::StringAttr::get(m.getContext(), "discharged"));
  }, "FP status/trap adaptation is not proved by admission");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    for (auto *op : {all(m, "check_gemm")[0], all(m, "gemm")[0]}) {
      op->setAttr("numerical_profile", mlir::StringAttr::get(m.getContext(), "reassociate_f32"));
      op->setAttr("reduction_order", mlir::StringAttr::get(m.getContext(), "reassociation_permitted"));
      op->setAttr("multiply_add_contraction", mlir::StringAttr::get(m.getContext(), "permitted"));
    }
  }, "coherent numerical weakening still contradicts paired source", false);
  reject(p, *built.module, [](mlir::ModuleOp m) {
    auto *r = all(m, "read")[0];
    r->getResult(0).setType(mlir::RankedTensorType::get({9, 3}, mlir::Float32Type::get(m.getContext())));
  }, "read cannot invent a tensor shape contradictory to explicit extent");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    auto *g = all(m, "gemm")[1];
    auto operand = g->getOperand(1);
    g->setOperand(1, g->getOperand(2));
    g->setOperand(2, operand);
  }, "noncommuting GEMM operands swapped");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "gemm")[0]->setAttr("value_id", mlir::IntegerAttr::get(mlir::IntegerType::get(m.getContext(), 128), 3));
  }, "oversized identity metadata");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    m->setAttr("mdsl_admission.source_sha256", mlir::StringAttr::get(m.getContext(), std::string(64, 'c')));
  }, "different well-formed source identity", false);
  reject(p, *built.module, [](mlir::ModuleOp m) {
    auto *read = all(m, "read")[2];
    read->setAttr("frontier", all(m, "read")[0]->getAttr("frontier"));
  }, "duplicate effect frontier cannot disguise resource state");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    auto *read = all(m, "read")[0];
    read->setAttr("noalias", mlir::BoolAttr::get(m.getContext(), true));
  }, "extra noalias attribute is not evidence");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "read")[0]->setAttr("storage_view", mlir::StringAttr::get(m.getContext(), "arbitrary_strided_f32"));
  }, "read cannot invent a different resource layout");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    all(m, "publish")[0]->removeAttr("storage_view");
  }, "publication cannot lose its required destination view");
  reject(p, *built.module, [](mlir::ModuleOp m) {
    auto *begin = all(m, "begin")[0];
    mlir::OpBuilder b(begin);
    b.setInsertionPointAfter(begin);
    auto one = b.create<mlir::arith::ConstantIntOp>(b.getUnknownLoc(), 1, 64);
    auto zero = b.create<mlir::arith::ConstantIntOp>(b.getUnknownLoc(), 0, 64);
    b.create<mlir::arith::DivSIOp>(b.getUnknownLoc(), one, zero);
  }, "memory-free division UB cannot enter the closed vocabulary");

  modelReject(p, [](cr::Program &v) { v.regions[0].body[5].rhs = 999; }, "unknown value rejects");
  modelReject(p, [](cr::Program &v) { v.regions[0].body[0].resource = 999; }, "unknown resource rejects");
  modelReject(p, [](cr::Program &v) { v.regions[0].body[4].result = 1; }, "duplicate value identity rejects");
  modelReject(p, [](cr::Program &v) { v.regions[0].body[0].columns = literal(4); }, "static contraction mismatch rejects");
  modelReject(p, [](cr::Program &v) { v.regions[0].body[0].rows = literal(std::numeric_limits<std::uint64_t>::max()); }, "unrepresentable literal rejects");
  modelReject(p, [](cr::Program &v) { v.regions[0].body[0].rows = literal(std::numeric_limits<std::int64_t>::max()); }, "static product overflow rejects");
  modelReject(p, [](cr::Program &v) { v.regions[0].body[2].numerical_profile = static_cast<cr::NumericalProfile>(99); }, "unknown numerical permission rejects");
  modelReject(p, [](cr::Program &v) { v.regions[0].body[3].result = 42; }, "publication cannot invent a value result");
  auto dead = p;
  dead.regions[0].body.push_back(gemm(9, 4, 3));
  auto deadModule = cr::buildModule(dead, context);
  check(deadModule && all(*deadModule.module, "check_gemm").size() == 3,
        "dead mathematical result retains its ordered checked-failure frontier");
  if (deadModule) {
    reject(dead, *deadModule.module, [](mlir::ModuleOp m) {
      auto *g = all(m, "gemm").back();
      auto *guard = all(m, "check_gemm").back();
      g->erase();
      guard->getResult(0).replaceAllUsesWith(guard->getOperand(0));
      guard->erase();
    }, "dead-value elimination cannot erase source-required failure", false);
  }
  modelReject(branched, [](cr::Program &v) {
    v.regions[0].body[4].then_body.push_back(read(90, 1, 2, 2));
    v.regions[0].body[6].lhs = 90;
  }, "branch-local value cannot escape through host-like mutable binding");

  std::cout << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
