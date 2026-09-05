#include "MatcoreTwoGemmRegion.h"
#include "MatcoreBufferizedGemmHandoff.h"
#include "MatcoreCpuRuntimeLowering.h"
#include "MatcoreOps.h"
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
template <typename T> llvm::SmallVector<T> all(mlir::ModuleOp module) {
  llvm::SmallVector<T> result;
  module.walk([&](T op) { result.push_back(op); });
  return result;
}
bool capture(frontend::Result &result) {
  frontend::Options options;
  auto path = std::filesystem::path(MDSLC_REGION_TEST_FRONTEND_DIR).parent_path() /
              "mlir/two_gemm_region_source.mdsl";
  options.input_path = path.string();
  options.clang_path = MDSLC_REGION_TEST_CLANG;
  options.clang_resource_directory = MDSLC_REGION_TEST_RESOURCE_DIR;
  options.trusted_public_headers = {MDSLC_REGION_TEST_PUBLIC_HEADER};
  options.inspect_two_gemm_regions = true;
  auto include = std::filesystem::path(MDSLC_REGION_TEST_PUBLIC_HEADER).parent_path().parent_path();
  options.compiler_arguments = {"-std=c++20", "-O2", "-I" + include.string(), path.string()};
  auto native = frontend::createClangLibToolingFrontend();
  bool okay = native->extract(options, result);
  for (const auto &diagnostic : result.diagnostics)
    std::cerr << diagnostic.message << '\n';
  return okay;
}
void reject(mlir::ModuleOp original,
            const frontend::AuthenticatedNativeFrontendEvidenceV1 &seal,
            const std::function<void(mlir::ModuleOp)> &mutate,
            const std::string &message) {
  mlir::OwningOpRef<mlir::ModuleOp> changed = original.clone();
  mutate(*changed);
  std::string error;
  mlir::ScopedDiagnosticHandler silence(changed->getContext(),
      [](mlir::Diagnostic &) { return mlir::success(); });
  check(!bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, *changed, error), message);
  check(!error.empty(), message + " diagnostic");
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
  check(all<mlir::func::FuncOp>(module).size() == 2, "both real source regions admitted");
  check(all<dialect::RegionCommitOp>(module).size() == 4, "two observable commits per region");
  check(all<dialect::RegionReadOp>(module).size() == 6, "no initial destination-data import");
  auto commits = all<dialect::RegionCommitOp>(module);
  auto matmuls = all<mlir::linalg::MatmulOp>(module);
  check(matmuls[1].getInputs()[0] == commits[0].getCommitted(), "second GEMM uses first post-commit tensor value");
  auto reads = all<dialect::RegionReadOp>(module);
  check(reads[3].getDescriptor() == reads[4].getDescriptor(), "A=A shares descriptor binding without noalias invention");
  check(!mlir::isMemoryEffectFree(commits[1]), "unused last tensor still has observable write");
  std::string error;
  check(bridge::verifyTwoGemmRegionMatchesNativeEvidenceV1(seal, module, error), "initial source pairing: " + error);

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
    check(all<dialect::RegionCommitOp>(*optimized).size() == 4,
          "DCE cannot erase observable commits or public roots");
    if (generalize)
      check(all<mlir::linalg::MatmulOp>(*optimized).empty() &&
                all<mlir::linalg::GenericOp>(*optimized).size() == 8,
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
    all<dialect::RegionGuardOp>(m).front().setRequiredGuardsAttr(b.getArrayAttr({}));
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
      auto one = mlir::arith::ConstantIntOp::create(b, b.getUnknownLoc(), 1, 32);
      auto zero = mlir::arith::ConstantIntOp::create(b, b.getUnknownLoc(), 0, 32);
      mlir::arith::DivSIOp::create(b, b.getUnknownLoc(), one, zero);
    }, scalar_body ? "memory-free UB cannot hide inside scalar computation"
                   : "memory-free UB is not a harmless incidental operation");

  std::vector<matcore::mdslc::mlir_lowering::CpuRuntimeDispatchRecordV1> records;
  check(!matcore::mdslc::mlir_lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(module, records, error) && records.empty(),
        "CPU runtime lowerer rejects inspection region");
  mlir::OwningOpRef<mlir::ModuleOp> forged = module.clone();
  forged->removeAttr("mdsl.analysis_only");
  forged->removeAttr("mdsl.execution_authority");
  forged->setAttr("mdsl.producer", mlir::StringAttr::get(&context, "clang-libtooling-v1"));
  check(!matcore::mdslc::mlir_lowering::lowerExplicitGemmToCpuRuntimeDispatchV1(*forged, records, error) && records.empty(),
        "forged labels cannot make region executable");
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
  auto copy = all<mlir::memref::CopyOp>(*buffer_module).front();
  check(copy.getTarget() == all<mlir::func::FuncOp>(*buffer_module).front().getArgument(1),
        "materialization copies into the required original output");
  std::string error;
  check(!bridge::verifyTwoGemmRegionModuleV1(*buffer_module, error),
        "buffer-looking control is not a source-authenticated region or executable candidate");
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
