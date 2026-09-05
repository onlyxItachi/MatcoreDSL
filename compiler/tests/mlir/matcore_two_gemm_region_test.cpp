#include "MatcoreTwoGemmRegion.h"
#include "MatcoreBufferizedGemmHandoff.h"
#include "MatcoreCpuRuntimeLowering.h"
#include "MatcoreOps.h"
#include "MatcoreRegionGuardLedger.h"
#include "MatcoreV1Bridge.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>

namespace {
namespace bridge = matcore::mdslc::mlir_bridge;
namespace frontend = matcore::mdslc::frontend;
namespace dialect = matcore::mdslc::mlir_dialect;
unsigned checks = 0, failures = 0;
void check(bool value, const std::string &message) {
  ++checks;
  if (!value) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}
template <typename T, typename Root> llvm::SmallVector<T> all(Root module) {
  llvm::SmallVector<T> result;
  module.walk([&](T op) { result.push_back(op); });
  return result;
}
bool capture(frontend::Result &result, bool alternate_options = false) {
  frontend::Options options;
  auto path = std::filesystem::path(MDSLC_REGION_TEST_PUBLIC_HEADER)
                  .parent_path().parent_path().parent_path() /
              "tests/mlir/two_gemm_region_source.mdsl";
  options.input_path = path.string();
  options.clang_path = MDSLC_REGION_TEST_CLANG;
  options.clang_resource_directory = MDSLC_REGION_TEST_RESOURCE_DIR;
  options.trusted_public_headers = {MDSLC_REGION_TEST_PUBLIC_HEADER};
  options.inspect_two_gemm_regions = true;
  auto include = std::filesystem::path(MDSLC_REGION_TEST_PUBLIC_HEADER).parent_path().parent_path();
  options.compiler_arguments = {"-std=c++20", "-O2", "-I" + include.string(), path.string()};
  if (alternate_options)
    options.compiler_arguments.insert(options.compiler_arguments.begin(),
                                      "-DMATCORE_REGION_OPTIONS_CONTROL=1");
  auto native = frontend::createClangLibToolingFrontend();
  bool okay = native->extract(options, result);
  for (const auto &diagnostic : result.diagnostics)
    std::cerr << diagnostic.message << '\n';
  return okay;
}
void reject(mlir::ModuleOp original,
            const frontend::AuthenticatedNativeFrontendEvidenceV1 &seal,
            const std::function<void(mlir::ModuleOp)> &mutate,
            const std::string &message, bool require_valid_ir = false) {
  mlir::OwningOpRef<mlir::ModuleOp> changed = original.clone();
  mutate(*changed);
  std::string error;
  mlir::ScopedDiagnosticHandler silence(changed->getContext(),
      [](mlir::Diagnostic &) { return mlir::success(); });
  if (require_valid_ir)
    check(mlir::succeeded(mlir::verify(*changed)), message + " is upstream-valid IR");
  check(!bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, *changed, error), message);
  check(!error.empty(), message + " diagnostic");
}
mlir::DictionaryAttr field(mlir::Builder &builder, mlir::DictionaryAttr original,
                           llvm::StringRef name, mlir::Attribute value) {
  mlir::NamedAttrList attributes(original);
  attributes.set(name, value);
  return attributes.getDictionary(builder.getContext());
}
void changeFirstSite(mlir::ModuleOp module, llvm::StringRef key,
                     mlir::Attribute value) {
  mlir::Builder b(module.getContext());
  auto function = all<mlir::func::FuncOp>(module).front();
  auto region = function->getAttrOfType<mlir::DictionaryAttr>("mdsl.region");
  auto sites = region.getAs<mlir::ArrayAttr>("sites");
  llvm::SmallVector<mlir::Attribute> updated(sites.begin(), sites.end());
  updated.front() = field(b, mlir::cast<mlir::DictionaryAttr>(updated.front()), key, value);
  function->setAttr("mdsl.region", field(b, region, "sites", b.getArrayAttr(updated)));
}
void changeFirstBinding(mlir::ModuleOp module, llvm::StringRef key,
                        mlir::Attribute value) {
  mlir::Builder b(module.getContext());
  auto function = all<mlir::func::FuncOp>(module).front();
  auto region = function->getAttrOfType<mlir::DictionaryAttr>("mdsl.region");
  auto site = mlir::cast<mlir::DictionaryAttr>(region.getAs<mlir::ArrayAttr>("sites")[0]);
  auto bindings = site.getAs<mlir::ArrayAttr>("bindings");
  llvm::SmallVector<mlir::Attribute> updated(bindings.begin(), bindings.end());
  updated.front() = field(b, mlir::cast<mlir::DictionaryAttr>(updated.front()), key, value);
  changeFirstSite(module, "bindings", b.getArrayAttr(updated));
}
bool optimize(mlir::ModuleOp module, bool generalize) {
  mlir::PassManager passes(module.getContext());
  if (generalize)
    passes.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  passes.addPass(mlir::createCanonicalizerPass());
  passes.addPass(mlir::createCSEPass());
  passes.addPass(mlir::createSymbolDCEPass());
  return mlir::succeeded(passes.run(module));
}
void changeLedgerRow(mlir::ModuleOp module, llvm::StringRef predicate,
                     const std::function<mlir::DictionaryAttr(mlir::Builder &,
                                                             mlir::DictionaryAttr)> &mutate) {
  mlir::Builder b(module.getContext());
  auto guard = all<dialect::RegionGuardOp>(module).front();
  auto ledger = guard.getGuardLedger();
  auto entries = ledger.getAs<mlir::ArrayAttr>("entries");
  llvm::SmallVector<mlir::Attribute> changed(entries.begin(), entries.end());
  bool found = false;
  for (auto &entry : changed) {
    auto row = mlir::cast<mlir::DictionaryAttr>(entry);
    if (row.getAs<mlir::StringAttr>("predicate").getValue() == predicate) {
      entry = mutate(b, row);
      found = true;
      break;
    }
  }
  check(found, "ledger mutation targets an existing predicate");
  guard.setGuardLedgerAttr(field(b, ledger, "entries", b.getArrayAttr(changed)));
}
void testGuardLedger(mlir::ModuleOp module,
                     const frontend::AuthenticatedNativeFrontendEvidenceV1 &seal) {
  using P = bridge::RegionGuardPredicateV1;
  using E = bridge::RegionGuardEvidenceV1;
  using F = bridge::RegionGuardFrontierV1;
  using V = matcore::mdslc::ir::v1::ValueId;
  auto guards = all<dialect::RegionGuardOp>(module);
  for (auto guard : guards) {
    bridge::RegionGuardLedgerV1 ledger;
    std::string error;
    check(bridge::decodeRegionGuardLedgerV1(guard.getGuardLedger(), ledger, error),
          "closed typed guard ledger decodes: " + error);
    std::array<unsigned, 4> classes{};
    unsigned overlaps = 0, post_execution = 0;
    for (const auto &entry : ledger.entries) {
      ++classes[static_cast<unsigned>(entry.evidence)];
      if (entry.predicate == P::PointerAlignmentRequired)
        check(entry.alignment_bytes == 4 && entry.evidence == E::RuntimeValidationRequired,
              "f32 alignment remains a required minimum, not source-proved pointer alignment");
      if (entry.predicate == P::OutputInputNoOverlap) {
        ++overlaps;
        check(entry.subjects.size() == 2 && entry.subjects[0] == V::Output,
              "overlap requirements involve output only; A/B alias remains legal");
      }
      if (entry.frontier == F::ExecutionAndReturn) {
        ++post_execution;
        check(entry.evidence == E::DispatchExecutionObligationRetained,
              "post-write provider/completion obligations are not pre-compute proofs");
      }
    }
    check(classes == std::array<unsigned, 4>{7, 18, 6, 5},
          "all representation/runtime/caller/dispatch obligations are present");
    check(overlaps == 2 && post_execution == 2,
          "both role-specific overlap and both post-execution obligations are retained");
    check(ledger.stage == guard.getStage() &&
              ledger.site_id == guard.getSemanticContract().getAs<mlir::StringAttr>("site_id").getValue(),
          "ledger source identity is scoped to its original call");
  }
  bridge::RegionGuardLedgerV1 aliased;
  std::string error;
  check(bridge::decodeRegionGuardLedgerV1(guards[2].getGuardLedger(), aliased, error) &&
            aliased.bindings[1] == aliased.bindings[2],
        "equal A/B descriptors retain distinct operand-role obligations");
  for (llvm::StringRef value : {"runtime_discharged", "guard_executed", "proven"})
    reject(module, seal, [value](mlir::ModuleOp m) {
      changeLedgerRow(m, "data_nonnull", [value](mlir::Builder &b, mlir::DictionaryAttr row) {
        return field(b, row, "evidence", b.getStringAttr(value));
      });
    }, "editable ledger cannot invent execution/discharge evidence", true);
  for (llvm::StringRef predicate : {"data_nonnull", "pointer_alignment_required",
                                    "backing_host_accessible", "backing_capacity_sufficient",
                                    "backing_lifetime_valid", "backing_access_permitted",
                                    "descriptor_object_valid", "no_conflicting_concurrent_access",
                                    "selected_implementation_eligible"})
    reject(module, seal, [predicate](mlir::ModuleOp m) {
      changeLedgerRow(m, predicate, [](mlir::Builder &b, mlir::DictionaryAttr row) {
        return field(b, row, "evidence", b.getStringAttr("representation_only"));
      });
    }, "representation cannot discharge physical/caller/provider predicate " + predicate.str(), true);
  for (bool duplicate : {false, true})
    reject(module, seal, [duplicate](mlir::ModuleOp m) {
      mlir::Builder b(m.getContext());
      auto guard = all<dialect::RegionGuardOp>(m).front();
      auto ledger = guard.getGuardLedger();
      auto entries = ledger.getAs<mlir::ArrayAttr>("entries");
      llvm::SmallVector<mlir::Attribute> changed(entries.begin(), entries.end());
      if (duplicate)
        changed.push_back(changed.front());
      else
        changed.erase(changed.begin());
      guard.setGuardLedgerAttr(field(b, ledger, "entries", b.getArrayAttr(changed)));
    }, duplicate ? "duplicate ledger obligation rejects" : "missing ledger obligation rejects", true);
  reject(module, seal, [](mlir::ModuleOp m) {
    changeLedgerRow(m, "data_nonnull", [](mlir::Builder &b, mlir::DictionaryAttr row) {
      return field(b, row, "predicate", b.getStringAttr("unreviewed_predicate"));
    });
  }, "unknown predicates cannot extend the bounded ledger", true);
  reject(module, seal, [](mlir::ModuleOp m) {
    changeLedgerRow(m, "data_nonnull", [](mlir::Builder &b, mlir::DictionaryAttr row) {
      return field(b, row, "executed", b.getBoolAttr(true));
    });
  }, "unknown row fields cannot smuggle proof banners", true);
  reject(module, seal, [](mlir::ModuleOp m) {
    mlir::Builder b(m.getContext());
    auto guard = all<dialect::RegionGuardOp>(m).front();
    guard.setGuardLedgerAttr(field(b, guard.getGuardLedger(), "executed", b.getBoolAttr(true)));
  }, "unknown wrapper fields cannot smuggle proof banners", true);
  reject(module, seal, [](mlir::ModuleOp m) {
    auto guards = all<dialect::RegionGuardOp>(m);
    guards[1].setGuardLedgerAttr(guards[0].getGuardLedger());
  }, "second call cannot borrow first call's ledger scope", true);
  for (llvm::StringRef key : {"site_id", "stage", "bindings"})
    reject(module, seal, [key](mlir::ModuleOp m) {
      mlir::Builder b(m.getContext());
      auto guard = all<dialect::RegionGuardOp>(m).front();
      auto ledger = guard.getGuardLedger();
      mlir::Attribute changed = b.getStringAttr("another_source_site");
      if (key == "stage")
        changed = b.getI64IntegerAttr(1);
      if (key == "bindings")
        changed = field(b, ledger.getAs<mlir::DictionaryAttr>("bindings"),
                        "lhs", b.getStringAttr("another_descriptor"));
      guard.setGuardLedgerAttr(field(b, ledger, key, changed));
    }, "ledger site/stage/descriptor bindings each independently pair with source", true);
  reject(module, seal, [](mlir::ModuleOp m) {
    changeLedgerRow(m, "pointer_alignment_required", [](mlir::Builder &b, mlir::DictionaryAttr row) {
      return field(b, row, "alignment_bytes", b.getI64IntegerAttr(8));
    });
  }, "pointer alignment minimum must match source contract", true);
  for (int malformed : {0, 1, 2, 3})
    reject(module, seal, [malformed](mlir::ModuleOp m) {
      changeLedgerRow(m, "pointer_alignment_required", [malformed](mlir::Builder &b, mlir::DictionaryAttr row) {
        mlir::Attribute value = b.getI64IntegerAttr(malformed == 0 ? -4 : 3);
        if (malformed == 2)
          value = b.getI32IntegerAttr(4);
        if (malformed == 3)
          value = b.getIntegerAttr(b.getIntegerType(128), llvm::APInt::getOneBitSet(128, 100));
        return field(b, row, "alignment_bytes", value);
      });
    }, "malformed alignment width/sign/power is rejected without narrowing", true);
  for (llvm::StringRef key : {"argument", "snapshot_stage"})
    for (int malformed : {0, 1, 2, 3})
      reject(module, seal, [key, malformed](mlir::ModuleOp m) {
        mlir::Builder b(m.getContext());
        mlir::Attribute value = b.getI64IntegerAttr(-1);
        if (malformed == 1)
          value = b.getI32IntegerAttr(0);
        if (malformed == 2)
          value = b.getStringAttr("zero");
        if (malformed == 3)
          value = b.getIntegerAttr(b.getIntegerType(128), llvm::APInt::getOneBitSet(128, 100));
        changeFirstBinding(m, key, value);
      }, "malformed descriptor binding integer rejects before getInt", true);
  for (llvm::StringRef key : {"frontier", "subjects"})
    reject(module, seal, [key](mlir::ModuleOp m) {
      changeLedgerRow(m, "data_nonnull", [key](mlir::Builder &b, mlir::DictionaryAttr row) {
        mlir::Attribute value = key == "frontier"
            ? mlir::Attribute(b.getStringAttr("already_executed"))
            : mlir::Attribute(b.getStrArrayAttr({"other_operand"}));
        return field(b, row, key, value);
      });
    }, "unknown frontier/operand role is outside the ledger vocabulary", true);
  for (llvm::StringRef role : {"lhs", "rhs"})
    reject(module, seal, [role](mlir::ModuleOp m) {
      mlir::Builder b(m.getContext());
      auto guard = all<dialect::RegionGuardOp>(m).front();
      auto ledger = guard.getGuardLedger();
      llvm::SmallVector<mlir::Attribute> changed;
      unsigned removed = 0;
      for (auto entry : ledger.getAs<mlir::ArrayAttr>("entries")) {
        auto row = mlir::cast<mlir::DictionaryAttr>(entry);
        if (row.getAs<mlir::StringAttr>("predicate").getValue() == "output_input_no_overlap" &&
            row.getAs<mlir::ArrayAttr>("subjects")[1] == b.getStringAttr(role))
          ++removed;
        else
          changed.push_back(entry);
      }
      check(removed == 1, "role-specific overlap adversary removes exactly its target");
      guard.setGuardLedgerAttr(field(b, ledger, "entries", b.getArrayAttr(changed)));
    }, "both output-versus-input roles require their own overlap obligation", true);
  reject(module, seal, [](mlir::ModuleOp m) {
    changeLedgerRow(m, "contraction_dimension_equal", [](mlir::Builder &b, mlir::DictionaryAttr row) {
      return field(b, row, "predicate", b.getStringAttr("output_rows_equal"));
    });
  }, "K equality cannot be substituted by another dimension equation", true);
  reject(module, seal, [](mlir::ModuleOp m) {
    changeLedgerRow(m, "output_input_no_overlap", [](mlir::Builder &b, mlir::DictionaryAttr row) {
      return field(b, row, "subjects", b.getStrArrayAttr({"lhs", "rhs"}));
    });
  }, "input-input noalias cannot replace output-input overlap obligation", true);
  for (llvm::StringRef predicate : {"provider_state_and_synchronous_completion",
                                    "may_write_before_failure_no_rollback"})
    reject(module, seal, [predicate](mlir::ModuleOp m) {
      changeLedgerRow(m, predicate, [](mlir::Builder &b, mlir::DictionaryAttr row) {
        return field(b, row, "frontier", b.getStringAttr("call_validation_before_compute"));
      });
    }, "post-execution obligation is not an atomic pre-write validation", true);
  for (bool wide : {false, true})
    reject(module, seal, [wide](mlir::ModuleOp m) {
      mlir::Builder b(m.getContext());
      auto guard = all<dialect::RegionGuardOp>(m).front();
      auto contract = guard.getSemanticContract();
      auto output = contract.getAs<mlir::DictionaryAttr>("output_semantics");
      mlir::Attribute value = wide
          ? mlir::Attribute(b.getIntegerAttr(b.getIntegerType(128), llvm::APInt::getOneBitSet(128, 100)))
          : mlir::Attribute(b.getI64IntegerAttr(-4));
      contract = field(b, contract, "output_semantics", field(b, output, "alignment_bytes", value));
      guard.setSemanticContractAttr(contract);
      changeFirstSite(m, "source_contract", contract);
    }, "malformed source alignment rejects before ledger derivation narrows it", true);
  mlir::OwningOpRef<mlir::ModuleOp> coordinated = module.clone();
  mlir::Builder b(module.getContext());
  auto guard = all<dialect::RegionGuardOp>(*coordinated).front();
  auto contract = guard.getSemanticContract();
  contract = field(b, contract, "output_semantics", field(b,
      contract.getAs<mlir::DictionaryAttr>("output_semantics"), "alignment_bytes", b.getI64IntegerAttr(8)));
  guard.setSemanticContractAttr(contract);
  changeFirstSite(*coordinated, "source_contract", contract);
  auto region = all<mlir::func::FuncOp>(*coordinated).front()->getAttrOfType<mlir::DictionaryAttr>("mdsl.region");
  auto site = mlir::cast<mlir::DictionaryAttr>(region.getAs<mlir::ArrayAttr>("sites")[0]);
  auto ledger = bridge::buildRegionGuardLedgerV1(b, contract, site.getAs<mlir::ArrayAttr>("bindings"), 0, error);
  check(static_cast<bool>(ledger), "changed declared requirement has a regenerated diagnostic ledger");
  guard.setGuardLedgerAttr(ledger);
  check(bridge::verifyTwoGemmRegionModuleV1(*coordinated, error),
        "coordinated source contract and regenerated ledger can be self-consistent");
  check(!bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, *coordinated, error),
        "coordinated source/ledger edits still cannot replace immutable native evidence");
}
void testForwardedRoles(mlir::ModuleOp module,
                        const frontend::AuthenticatedNativeFrontendEvidenceV1 &seal) {
  // The original fixtures retain their positions; the new fixtures separately
  // cover RHS-only forwarding and the pre-existing C*C lhs-priority case.
  for (unsigned region : {2u, 3u}) {
    const unsigned forwarded = region == 2 ? 1 : 0;
    const unsigned imported = 1 - forwarded;
    auto function = all<mlir::func::FuncOp>(module)[region];
    auto commits = all<dialect::RegionCommitOp>(function);
    auto guards = all<dialect::RegionGuardOp>(function);
    auto reads = all<dialect::RegionReadOp>(function);
    auto mm = all<mlir::linalg::MatmulOp>(function)[1];
    check(reads.size() == 3 && reads[2].getStage() == 1 &&
              reads[2].getRole() == (imported ? "rhs" : "lhs"),
          "only the nonforwarded role gets a second-call memory snapshot");
    check(mm.getInputs()[forwarded] == commits[0].getCommitted() &&
              mm.getInputs()[imported] == reads[2].getValue(),
          "carried value preserves its mathematical operand position");
    check(reads[2].getChecked() == guards[1].getChecked() &&
              guards[1].getOrder() == commits[0].getOrder(),
          "possibly aliasing other input is read after the first observable commit");
    check((reads[2].getDescriptor() == commits[0].getDescriptor()) == (region == 3),
          "C*C retains its same-descriptor reload; RHS-only retains a distinct binding");
    auto dims = all<mlir::tensor::DimOp>(function);
    check(dims[2].getSource() == mm.getInputs()[0] && dims[2].getConstantIndex() == 0 &&
              dims[3].getSource() == mm.getInputs()[1] && dims[3].getConstantIndex() == 1,
          "output rows follow current lhs and columns current rhs, not producer geometry");
    check(mlir::cast<mlir::RankedTensorType>(mm.getResult(0).getType()).getNumDynamicDims() == 2,
          "source-connected mirror does not invent static shape facts");
    reject(module, seal, [region](mlir::ModuleOp m) {
      auto fn = all<mlir::func::FuncOp>(m)[region];
      auto operation = all<mlir::linalg::MatmulOp>(fn)[1];
      auto lhs = operation.getInputs()[0], rhs = operation.getInputs()[1];
      operation->setOperand(0, rhs);
      operation->setOperand(1, lhs);
    }, "ordered GEMM operands cannot be swapped even with identical dynamic types", true);
    reject(module, seal, [region, forwarded](mlir::ModuleOp m) {
      auto fn = all<mlir::func::FuncOp>(m)[region];
      all<mlir::linalg::MatmulOp>(fn)[1]->setOperand(
          forwarded, all<dialect::RegionReadOp>(fn)[0].getValue());
    }, "carried operand cannot become a same-shaped precommit input", true);
    reject(module, seal, [region, imported](mlir::ModuleOp m) {
      auto fn = all<mlir::func::FuncOp>(m)[region];
      all<mlir::linalg::MatmulOp>(fn)[1]->setOperand(
          imported, all<dialect::RegionCommitOp>(fn)[0].getCommitted());
    }, "the other operand must use its own late read, not an unused read plus dual forwarding", true);
    reject(module, seal, [region](mlir::ModuleOp m) {
      auto fn = all<mlir::func::FuncOp>(m)[region];
      auto read = all<dialect::RegionReadOp>(fn)[2];
      read.getValue().replaceAllUsesWith(all<dialect::RegionCommitOp>(fn)[0].getCommitted());
      read.erase();
    }, "erasing the late read cannot silently introduce both-input SSA forwarding", true);
    for (bool hoist : {false, true})
      reject(module, seal, [region, hoist](mlir::ModuleOp m) {
        auto fn = all<mlir::func::FuncOp>(m)[region];
        auto read = all<dialect::RegionReadOp>(fn)[2];
        read.getCheckedMutable().assign(all<dialect::RegionGuardOp>(fn)[0].getChecked());
        if (hoist)
          read->moveBefore(all<dialect::RegionCommitOp>(fn)[0]);
      }, hoist ? "other-operand read cannot hoist across possibly aliasing output write"
               : "late other-operand read cannot borrow the first call's guard", true);
    reject(module, seal, [region, forwarded](mlir::ModuleOp m) {
      auto read = all<dialect::RegionReadOp>(all<mlir::func::FuncOp>(m)[region])[2];
      read.setRole(forwarded ? "rhs" : "lhs");
    }, "editable read role cannot choose the carried operand", true);
    reject(module, seal, [region](mlir::ModuleOp m) {
      auto guards = all<dialect::RegionGuardOp>(all<mlir::func::FuncOp>(m)[region]);
      auto guard = guards[1];
      auto lhs = guard.getLhs(), rhs = guard.getRhs();
      guard.getLhsMutable().assign(region == 3 ? guards[0].getLhs() : rhs);
      guard.getRhsMutable().assign(lhs);
    }, "source lhs/rhs guard bindings cannot change with carried input", true);
    for (unsigned axis : {0u, 1u}) {
      reject(module, seal, [region, axis](mlir::ModuleOp m) {
        auto fn = all<mlir::func::FuncOp>(m)[region];
        auto operation = all<mlir::linalg::MatmulOp>(fn)[1];
        all<mlir::tensor::DimOp>(fn)[2 + axis].getSourceMutable().assign(operation.getInputs()[1 - axis]);
      }, "output extent cannot use the wrong current operand", true);
      reject(module, seal, [region, axis](mlir::ModuleOp m) {
        auto query = all<mlir::tensor::DimOp>(all<mlir::func::FuncOp>(m)[region])[2 + axis];
        mlir::OpBuilder b(query);
        auto index = mlir::arith::ConstantOp::create(b, query.getLoc(), b.getIndexAttr(1 - axis));
        query.getIndexMutable().assign(index.getResult());
      }, "output extent cannot use the wrong axis of the correct operand", true);
    }
    reject(module, seal, [region, forwarded](mlir::ModuleOp m) {
      auto fn = all<mlir::func::FuncOp>(m)[region];
      mlir::Builder b(m.getContext());
      auto contract = fn->getAttrOfType<mlir::DictionaryAttr>("mdsl.region");
      auto sites = contract.getAs<mlir::ArrayAttr>("sites");
      llvm::SmallVector<mlir::Attribute> updated(sites.begin(), sites.end());
      auto site = mlir::cast<mlir::DictionaryAttr>(updated[1]);
      auto type = mlir::cast<mlir::FunctionType>(site.getAs<mlir::TypeAttr>("tensor_type").getValue());
      llvm::SmallVector<mlir::Type> inputs(type.getInputs());
      inputs[forwarded] = mlir::RankedTensorType::get({2, 3}, b.getF32Type());
      updated[1] = field(b, site, "tensor_type", mlir::TypeAttr::get(b.getFunctionType(inputs, type.getResults())));
      fn->setAttr("mdsl.region", field(b, contract, "sites", b.getArrayAttr(updated)));
    }, "carried input type cannot differ from the produced tensor version", true);
  }
}
void testRegion(frontend::Result &source) {
  check(source.native_evidence && source.native_evidence->valid(), "native source issued sealed region evidence");
  if (!source.native_evidence)
    return;
  auto &seal = *source.native_evidence;
  mlir::MLIRContext context;
  auto result = bridge::deriveAuthenticatedTwoGemmRegionsV1(seal, context);
  check(static_cast<bool>(result), "source-connected regions derive: " + result.error);
  if (!result)
    return;
  auto module = *result.module;
  check(all<mlir::func::FuncOp>(module).size() == 4, "all four real source regions admitted");
  check(all<dialect::RegionCommitOp>(module).size() == 8, "two observable commits per region");
  check(all<dialect::RegionReadOp>(module).size() == 12, "no initial destination-data import");
  auto commits = all<dialect::RegionCommitOp>(module);
  auto matmuls = all<mlir::linalg::MatmulOp>(module);
  check(matmuls[1].getInputs()[0] == commits[0].getCommitted(), "second GEMM uses first post-commit tensor value");
  auto reads = all<dialect::RegionReadOp>(module);
  check(reads[3].getDescriptor() == reads[4].getDescriptor(), "A=A shares descriptor binding without noalias invention");
  check(!mlir::isMemoryEffectFree(commits[1]), "unused last tensor still has observable write");
  std::string error;
  check(bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, module, error), "initial source pairing: " + error);
  testGuardLedger(module, seal);
  testForwardedRoles(module, seal);
  mlir::MLIRContext parsed_context;
  bridge::registerTwoGemmRegionDialectsV1(parsed_context);
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(
      bridge::serializeDeterministicMlir(module), &parsed_context);
  check(static_cast<bool>(parsed), "exported descriptor/order types parse in a fresh context");
  check(parsed && bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, *parsed, error),
        "serialized region retains source pairing across independent MLIR contexts: " + error);
  frontend::Result alternate;
  check(capture(alternate, true) && alternate.native_evidence,
        "same source admits under a separately sealed compilation context");
  if (alternate.native_evidence) {
    check(source.region_capture_identity != alternate.region_capture_identity,
          "compiler options participate in sealed capture identity");
    check(!bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(*alternate.native_evidence, module, error),
          "same arithmetic and source cannot pair with a different compilation seal");
  }

  auto edited_source = source;
  edited_source.two_gemm_regions.clear();
  edited_source.region_capture_identity = "forged";
  auto unchanged = bridge::deriveAuthenticatedTwoGemmRegionsV1(*edited_source.native_evidence, context);
  check(unchanged && bridge::serializeDeterministicMlir(*unchanged.module) ==
                         bridge::serializeDeterministicMlir(module),
        "mutable diagnostics cannot forge sealed admission");

  for (bool generalize : {false, true}) {
    mlir::OwningOpRef<mlir::ModuleOp> optimized = module.clone();
    // Incidental diagnostics and an extra unused pure constant are not semantic identity.
    optimized->walk([&](mlir::Operation *op) { op->setLoc(mlir::UnknownLoc::get(&context)); });
    auto function = all<mlir::func::FuncOp>(*optimized).front();
    mlir::OpBuilder b(&context);
    b.setInsertionPointToStart(&function.getBody().front());
    mlir::arith::ConstantOp::create(b, b.getUnknownLoc(), b.getI64IntegerAttr(987));
    check(bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, *optimized, error),
          "incidental location and pure-operation differences preserve contract");
    check(optimize(*optimized, generalize), "actual upstream transforms succeed");
    check(bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, *optimized, error),
          "upstream transformed region remains paired: " + error);
    check(all<dialect::RegionCommitOp>(*optimized).size() == 8,
          "DCE cannot erase observable commits or public roots");
    if (generalize)
      check(all<mlir::linalg::MatmulOp>(*optimized).empty() &&
                all<mlir::linalg::GenericOp>(*optimized).size() == 16,
            "named operations actually became generic Linalg");
  }
  reject(module, seal, [](mlir::ModuleOp m) {
    auto guards = all<dialect::RegionGuardOp>(m);
    guards[1].getOrderMutable().assign(all<dialect::RegionBeginOp>(m).front().getOrder());
  }, "second guard cannot bypass first commit");
  reject(module, seal, [](mlir::ModuleOp m) {
    auto guards = all<dialect::RegionGuardOp>(m);
    guards[1]->moveBefore(all<dialect::RegionCommitOp>(m).front());
  }, "second failure frontier cannot hoist before first write");
  reject(module, seal, [](mlir::ModuleOp m) {
    auto read = all<dialect::RegionReadOp>(m)[2];
    read.getCheckedMutable().assign(all<dialect::RegionGuardOp>(m).front().getChecked());
  }, "D snapshot cannot use stale guard across possibly aliasing C write");
  reject(module, seal, [](mlir::ModuleOp m) {
    auto mm = all<mlir::linalg::MatmulOp>(m);
    mm[1]->setOperand(0, all<dialect::RegionReadOp>(m).front().getValue());
  }, "same-shaped stale tensor cannot replace produced value");
  reject(module, seal, [](mlir::ModuleOp m) {
    auto commit = all<dialect::RegionCommitOp>(m).front();
    commit.getDescriptorMutable().assign(all<dialect::RegionGuardOp>(m).front().getLhs());
  }, "commit cannot write wrong original storage binding");
  reject(module, seal, [](mlir::ModuleOp m) {
    auto commit = all<dialect::RegionCommitOp>(m)[1];
    commit.getOrder().replaceAllUsesWith(commit.getChecked());
    commit.erase();
  }, "unused final tensor cannot eliminate externally observable write");
  reject(module, seal, [](mlir::ModuleOp m) {
    mlir::Builder b(m.getContext());
    all<dialect::RegionGuardOp>(m).front().setGuardLedgerAttr(b.getDictionaryAttr({}));
  }, "empty guard banner cannot replace descriptor/alias/fenv/policy obligations");
  reject(module, seal, [](mlir::ModuleOp m) {
    mlir::Builder b(m.getContext());
    all<dialect::RegionCommitOp>(m).front().setFailureBehavior("atomic_rollback");
  }, "commit cannot promise rollback or erase partial-write failure");
  reject(module, seal, [](mlir::ModuleOp m) {
    auto dims = all<mlir::tensor::DimOp>(m);
    dims.front().getSourceMutable().assign(all<dialect::RegionReadOp>(m)[1].getValue());
  }, "dynamic output shape cannot be inferred from wrong input axis");
  reject(module, seal, [](mlir::ModuleOp m) {
    mlir::Builder b(m.getContext());
    auto fill = all<mlir::linalg::FillOp>(m).front();
    fill.getInputs().front().getDefiningOp<mlir::arith::ConstantOp>().setValueAttr(b.getF32FloatAttr(1.0));
  }, "overwrite seed must remain positive zero");
  reject(module, seal, [](mlir::ModuleOp m) {
    auto matmul = all<mlir::linalg::MatmulOp>(m).front();
    auto multiply = mlir::cast<mlir::arith::MulFOp>(matmul.getRegion().front().front());
    multiply.setFastmath(mlir::arith::FastMathFlags::fast);
  }, "arbitrary fast-math cannot replace retained numerical profile");
  for (bool change_iterator : {false, true})
    reject(module, seal, [change_iterator](mlir::ModuleOp m) {
      mlir::PassManager passes(m.getContext());
      passes.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
      check(mlir::succeeded(passes.run(m)), "mutation starts from actual upstream generalization");
      auto generic = all<mlir::linalg::GenericOp>(m)[1];
      mlir::Builder b(m.getContext());
      if (change_iterator) {
        llvm::SmallVector<mlir::Attribute> iterators(
            3, mlir::linalg::IteratorTypeAttr::get(m.getContext(), mlir::utils::IteratorType::parallel));
        generic.setIteratorTypesAttr(b.getArrayAttr(iterators));
      } else {
        auto maps = generic.getIndexingMapsArray();
        maps[0] = mlir::AffineMap::get(3, 0,
            {b.getAffineDimExpr(2), b.getAffineDimExpr(0)}, m.getContext());
        generic.setIndexingMapsAttr(b.getAffineMapArrayAttr(maps));
      }
    }, change_iterator ? "parallel iterator cannot replace GEMM reduction"
                       : "transposed input map cannot replace source GEMM indexing", true);
  reject(module, seal, [](mlir::ModuleOp m) {
    mlir::Builder b(m.getContext());
    auto tensor = mlir::RankedTensorType::get(
        {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic}, b.getF64Type());
    auto type = b.getFunctionType({tensor, tensor, tensor}, {tensor});
    changeFirstSite(m, "tensor_type", mlir::TypeAttr::get(type));
  }, "declared dtype cannot differ from actual guarded computation", true);
  for (bool numerical : {false, true}) {
    mlir::OwningOpRef<mlir::ModuleOp> changed = module.clone();
    mlir::Builder b(&context);
    auto guard = all<dialect::RegionGuardOp>(*changed).front();
    auto contract = guard.getSemanticContract();
    if (numerical)
      contract = field(b, contract, "numerical", field(b,
          contract.getAs<mlir::DictionaryAttr>("numerical"), "approximate_math", b.getBoolAttr(true)));
    else
      contract = field(b, contract, "policy", field(b,
          contract.getAs<mlir::DictionaryAttr>("policy"), "target", b.getStringAttr("cuda")));
    guard.setSemanticContractAttr(contract);
    changeFirstSite(*changed, "source_contract", contract);
    check(mlir::succeeded(mlir::verify(*changed)), "coordinated contract mutation remains upstream-valid");
    check(bridge::verifyTwoGemmRegionModuleV1(*changed, error),
          "standalone self-consistency is deliberately not source authentication");
    check(!bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, *changed, error),
          numerical ? "paired verification rejects changed numerical permissions"
                    : "paired verification rejects changed target policy");
  }
  reject(module, seal, [](mlir::ModuleOp m) {
    mlir::Builder b(m.getContext());
    m->setAttr("mdsl.capture_identity", b.getStringAttr("another-compilation"));
  }, "same arithmetic must pair with sealed source/dependency identity");
  reject(module, seal, [](mlir::ModuleOp m) {
    auto function = all<mlir::func::FuncOp>(m).front();
    function.erase();
  }, "source region omission is rejected");
  for (bool scalar_body : {false, true})
    reject(module, seal, [scalar_body](mlir::ModuleOp m) {
      mlir::OpBuilder b(m.getContext());
      if (scalar_body)
        b.setInsertionPointToStart(&all<mlir::linalg::MatmulOp>(m).front().getRegion().front());
      else
        b.setInsertionPointToStart(&all<mlir::func::FuncOp>(m).front().getBody().front());
      auto one = mlir::arith::ConstantOp::create(b, b.getUnknownLoc(), b.getI32IntegerAttr(1));
      auto zero = mlir::arith::ConstantOp::create(b, b.getUnknownLoc(), b.getI32IntegerAttr(0));
      mlir::arith::DivSIOp::create(b, b.getUnknownLoc(), one, zero);
    }, scalar_body ? "memory-free UB cannot hide inside scalar computation"
                   : "memory-free UB is not a harmless incidental operation", true);

  std::vector<matcore::mdslc::mlir_lowering::CpuRuntimeDispatchRecordV1> records;
  check(!matcore::mdslc::mlir_lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(module, records, error) && records.empty(),
        "CPU runtime lowerer rejects inspection region");
  mlir::OwningOpRef<mlir::ModuleOp> forged = module.clone();
  (*forged)->removeAttr("mdsl.analysis_only");
  (*forged)->removeAttr("mdsl.execution_authority");
  (*forged)->setAttr("mdsl.producer", mlir::StringAttr::get(&context, "clang-libtooling-v1"));
  (*forged)->setAttr("mdsl.capability", mlir::StringAttr::get(&context, "validated_cpu"));
  (*forged)->setAttr("mdsl.retry_safe", mlir::BoolAttr::get(&context, true));
  (*forged)->setAttr("mdsl.target", mlir::StringAttr::get(&context, "cpu"));
  check(!matcore::mdslc::mlir_lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(*forged, records, error) && records.empty(),
        "forged capability, retry and target labels cannot make region executable");
}

void testUpstreamStorageControls() {
  mlir::MLIRContext context;
  bridge::registerBufferizedGemmHandoffDialectsV1(context);
  // This control models tensor values only, intentionally omitting the source
  // storage commit. Removing its dead arithmetic is correct upstream behavior.
  constexpr auto pure = R"mlir(module {
    func.func @pure(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>) {
      %z = arith.constant 0.0 : f32
      %i = linalg.fill ins(%z : f32) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
      %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%i : tensor<2x2xf32>) -> tensor<2x2xf32>
      return
    }
  })mlir";
  auto pure_module = mlir::parseSourceString<mlir::ModuleOp>(pure, &context);
  check(pure_module && optimize(*pure_module, false), "upstream pure tensor control canonicalizes");
  check(pure_module && all<mlir::linalg::MatmulOp>(*pure_module).empty(), "DPS alone does not retain output mutation");
  constexpr auto materialized = R"mlir(module {
    func.func @materialized(%a: memref<2x2xf32>, %c: memref<2x2xf32>) {
      %av = bufferization.to_tensor %a restrict : memref<2x2xf32> to tensor<2x2xf32>
      %empty = bufferization.alloc_tensor() : tensor<2x2xf32>
      %z = arith.constant 0.0 : f32
      %i = linalg.fill ins(%z : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
      %r = linalg.matmul ins(%av, %av : tensor<2x2xf32>, tensor<2x2xf32>) outs(%i : tensor<2x2xf32>) -> tensor<2x2xf32>
      bufferization.materialize_in_destination %r in writable %c : (tensor<2x2xf32>, memref<2x2xf32>) -> ()
      return
    }
  })mlir";
  auto buffer_module = mlir::parseSourceString<mlir::ModuleOp>(materialized, &context);
  check(static_cast<bool>(buffer_module), "explicit materialization control parses without A/B noalias assumption");
  if (!buffer_module)
    return;
  check(optimize(*buffer_module, false), "observable materialization survives canonicalization");
  mlir::bufferization::OneShotBufferizationOptions options;
  options.allowUnknownOps = false;
  mlir::bufferization::BufferizationState state;
  mlir::bufferization::BufferizationStatistics statistics;
  check(mlir::succeeded(mlir::bufferization::runOneShotModuleBufferize(*buffer_module, options, state, &statistics)),
        "upstream explicit materialization bufferizes");
  check(all<mlir::memref::AllocOp>(*buffer_module).size() == 1 &&
            all<mlir::memref::CopyOp>(*buffer_module).size() == 1 &&
            all<mlir::memref::DeallocOp>(*buffer_module).empty(),
        "materialization exposes allocation/copy and unresolved ownership, not zero-copy");
  auto copies = all<mlir::memref::CopyOp>(*buffer_module);
  check(copies.size() == 1 && copies.front().getTarget() ==
                                 all<mlir::func::FuncOp>(*buffer_module).front().getArgument(1),
        "materialization copies into the required original output");
  std::string error;
  check(!bridge::verifyTwoGemmRegionModuleV1(*buffer_module, error),
        "buffer-looking control is not a source-authenticated region");
  std::vector<matcore::mdslc::mlir_lowering::CpuRuntimeDispatchRecordV1> records;
  check(!matcore::mdslc::mlir_lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(
            *buffer_module, records, error) && records.empty(),
        "actual CPU lowerer rejects buffer control with unresolved allocation ownership");
}
} // namespace
int main() {
  frontend::Result source;
  check(capture(source), "native two-GEMM source capture succeeds");
  if (source.native_evidence)
    testRegion(source);
  testUpstreamStorageControls();
  std::cout << "two-GEMM region checks: " << checks - failures << '/' << checks << '\n';
  return failures ? 1 : 0;
}
