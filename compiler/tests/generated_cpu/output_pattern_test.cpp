// Independent inspection/issuance falsifiers. Parsing these artifacts does not
// authenticate source or give a supplied module permission to execute.
#include "MatcoreCpuGemmCandidate.h"
#include "MatcoreGemmOutputPattern.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cc = matcore::mdslc::cpu_candidate;
namespace gp = matcore::mdslc::gemm_pattern;
namespace {
unsigned checks = 0, failures = 0, validCorruptions = 0;
void check(bool condition, const std::string &label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL output pattern: %s\n", label.c_str());
  }
}
std::string print(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream, mlir::OpPrintingFlags().useLocalScope());
  return result;
}
mlir::Value constant(mlir::ModuleOp module, std::int64_t value) {
  mlir::Value result;
  module.walk([&](mlir::arith::ConstantIndexOp op) {
    if (!result && op.value() == value) result = op.getResult();
  });
  return result;
}
std::string normalizedMachine(const std::string &text) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseAssemblyString(text, diagnostic, context);
  if (!module) return {};
  module->setTargetTriple(llvm::Triple(""));
  for (auto &function : *module) {
    if (function.getName().starts_with("_mlir_ciface_"))
      function.setName(cc::kStrictGemmCInterfaceV1);
    else if (function.getName().starts_with("__matcore_strict_gemm_f32"))
      function.setName(cc::kStrictGemmSymbolV1);
    for (const auto *name : {"target-cpu", "target-features", "prefer-vector-width"})
      function.removeFnAttr(name);
  }
  std::string result;
  llvm::raw_string_ostream stream(result);
  module->print(stream, nullptr);
  return result;
}
} // namespace

int main() {
  mlir::MLIRContext context;
  auto canonical = cc::buildStrictGemmStagesV1(context);
  check(bool(canonical), "canonical stages build: " + canonical.error);
  if (!canonical) return 1;
  const auto semanticBefore = print(*canonical.semantic);
  const auto structuredBefore = print(*canonical.structured);
  const auto bufferBefore = print(*canonical.bufferized);
  const auto control = cc::issueStrictGemmArtifactV1(context, false);
  check(bool(control), "unchanged scalar issuer builds");
  std::string error;
  std::vector<std::string> schedules, machines;
  constexpr std::array patterns{
      gp::OutputTilePatternV1{1, 1}, gp::OutputTilePatternV1{2, 3},
      gp::OutputTilePatternV1{3, 5}, gp::OutputTilePatternV1{4, 16},
      gp::OutputTilePatternV1{7, 2}, gp::OutputTilePatternV1{64, 64},
      gp::OutputTilePatternV1{1, 64}, gp::OutputTilePatternV1{64, 1}};
  for (const auto tiles : patterns) {
    const auto label = std::to_string(tiles.tile_m) + "x" + std::to_string(tiles.tile_n);
    check(gp::validateOutputTilePatternV1(tiles, error), "accepted tuple " + label);
    auto derived = gp::deriveStrictGemmOutputTiledV1(*canonical.bufferized, tiles, error);
    check(bool(derived), "upstream Transform derives " + label + ": " + error);
    if (!derived) continue;
    check(gp::verifyStrictGemmOutputTiledV1(*derived, tiles, error), "derived verifier " + label);
    check(print(*canonical.bufferized) == bufferBefore,
          "successful derivation keeps canonical input unchanged " + label);
    check(!cc::verifyStrictGemmBufferizedV1(*derived, error),
          "derived payload does not impersonate canonical witness " + label);
    const gp::OutputTilePatternV1 other{tiles.tile_m == 64 ? 63 : tiles.tile_m + 1,
                                       tiles.tile_n};
    check(!gp::verifyStrictGemmOutputTiledV1(*derived, other, error) && !error.empty(),
          "reconstruction binds exact parameter pair " + label);
    const auto artifact = cc::issueStrictGemmArtifactV1(context, false,
        cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxX86_64,
        cc::StrictCpuIsaV1::Baseline, tiles);
    const auto repeated = cc::issueStrictGemmArtifactV1(context, false,
        cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxX86_64,
        cc::StrictCpuIsaV1::Baseline, tiles);
    check(bool(artifact), "issued realization " + label + ": " + artifact.error);
    if (!artifact) continue;
    check(repeated && repeated.manifest == artifact.manifest &&
              repeated.llvm_ir == artifact.llvm_ir &&
              repeated.scheduled_ir == artifact.scheduled_ir &&
              repeated.transform_ir == artifact.transform_ir,
          "deterministic replay " + label);
    check(artifact.semantic_ir == control.semantic_ir &&
              artifact.structured_ir == control.structured_ir &&
              artifact.bufferized_ir == control.bufferized_ir &&
              semanticBefore == print(*canonical.semantic) &&
              structuredBefore == print(*canonical.structured),
          "parameters never enter WHAT/canonical stages " + label);
    check(artifact.scheduled_ir == print(*derived) &&
              artifact.scheduled_ir != bufferBefore &&
              ((tiles.tile_m == 1 && tiles.tile_n == 1) ||
               artifact.llvm_ir != control.llvm_ir) && !artifact.transform_ir.empty(),
          "parameters create computation structure, not metadata only " + label);
    check(artifact.llvm_ir.find("fmul float") != std::string::npos &&
              artifact.llvm_ir.find("fadd float") != std::string::npos &&
              artifact.llvm_ir.find("llvm.fma") == std::string::npos &&
              artifact.llvm_ir.find(" fast ") == std::string::npos,
          "issued arithmetic stays separate strict f32 " + label);
    for (std::size_t i = 0; i < schedules.size(); ++i)
      check(schedules[i] != artifact.scheduled_ir && machines[i] != artifact.llvm_ir,
            "distinct parameter pairs change actual schedule and LLVM " + label);
    schedules.push_back(artifact.scheduled_ir);
    machines.push_back(artifact.llvm_ir);
  }

  constexpr std::array badPatterns{
      gp::OutputTilePatternV1{0, 0}, gp::OutputTilePatternV1{0, 3},
      gp::OutputTilePatternV1{3, 0}, gp::OutputTilePatternV1{-1, 3},
      gp::OutputTilePatternV1{3, -1}, gp::OutputTilePatternV1{65, 1},
      gp::OutputTilePatternV1{1, 65},
      gp::OutputTilePatternV1{std::numeric_limits<std::int64_t>::max(), 2},
      gp::OutputTilePatternV1{2, std::numeric_limits<std::int64_t>::min()}};
  for (const auto tiles : badPatterns) {
    check(!gp::validateOutputTilePatternV1(tiles, error) && !error.empty(),
          "invalid/nonpositive/out-of-range tuple fails closed");
    check(gp::strictGemmOutputTileTransformV1(tiles, error).empty() && !error.empty(),
          "bad tuple has no schedule text");
    check(!gp::deriveStrictGemmOutputTiledV1(*canonical.bufferized, tiles, error) &&
              !error.empty() && print(*canonical.bufferized) == bufferBefore,
          "bad tuple does not mutate its verified input");
    const auto bad = cc::issueStrictGemmArtifactV1(context, false,
        cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxX86_64,
        cc::StrictCpuIsaV1::Baseline, tiles);
    check(!bad && bad.llvm_ir.empty() && !bad.error.empty(), "bad tuple cannot issue a leaf");
  }
  for (const auto policy : {cc::StrictGemmScheduleV1::ScalarMNK,
                             cc::StrictGemmScheduleV1::RowContiguousMKN,
                             static_cast<cc::StrictGemmScheduleV1>(99)}) {
    const auto bad = cc::issueStrictGemmArtifactV1(context, false, policy,
        cc::CpuTargetV1::LinuxX86_64, cc::StrictCpuIsaV1::Baseline, {3, 5});
    check(!bad && bad.llvm_ir.empty() && !bad.error.empty(),
          "non-tiled/unknown policy cannot borrow parameter tuple");
  }
  for (const auto isa : {cc::StrictCpuIsaV1::AVX, cc::StrictCpuIsaV1::AVX2,
                         cc::StrictCpuIsaV1::AVX512F}) {
    const auto bad = cc::issueStrictGemmArtifactV1(context, false,
        cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxAArch64,
        isa, {3, 5});
    check(!bad && bad.llvm_ir.empty() && !bad.error.empty(),
          "AArch64 cannot borrow x86 ISA target features");
  }
  const auto badIsa = cc::issueStrictGemmArtifactV1(context, false,
      cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxX86_64,
      static_cast<cc::StrictCpuIsaV1>(99), {3, 5});
  check(!badIsa && badIsa.llvm_ir.empty(), "unknown ISA fails closed");
  const auto badTarget = cc::issueStrictGemmArtifactV1(context, false,
      cc::StrictGemmScheduleV1::OutputTiledMKN, static_cast<cc::CpuTargetV1>(99),
      cc::StrictCpuIsaV1::Baseline, {3, 5});
  check(!badTarget && badTarget.llvm_ir.empty(), "unknown target fails closed");

  const gp::OutputTilePatternV1 tiles{3, 5};
  auto scheduled = gp::deriveStrictGemmOutputTiledV1(*canonical.bufferized, tiles, error);
  if (!scheduled) return 1;
  const auto reject = [&](const char *label, auto mutate) {
    mlir::OwningOpRef<mlir::ModuleOp> bad = scheduled->clone();
    const auto original = print(*bad);
    const bool hit = mutate(*bad);
    check(hit && print(*bad) != original, std::string("mutation reaches payload: ") + label);
    if (!hit) return;
    mlir::ScopedDiagnosticHandler diagnostics(&context, [](mlir::Diagnostic &) {
      return mlir::success();
    });
    const bool valid = mlir::succeeded(mlir::verify(*bad));
    validCorruptions += valid;
    check(!gp::verifyStrictGemmOutputTiledV1(*bad, tiles, error) && !error.empty(),
          std::string("reject modified IR: ") + label);
    check(print(*scheduled) != print(*bad) && print(*canonical.bufferized) == bufferBefore,
          "rejected inspection does not alter original or canonical payload");
  };
  reject("forged module authority", [&](mlir::ModuleOp module) {
    module->setAttr("mdsl.execution_authority", mlir::UnitAttr::get(&context));
    return true;
  });
  reject("forged nested noalias", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::linalg::GenericOp op) {
      op->setAttr("mdsl.noalias", mlir::UnitAttr::get(&context)); hit = true;
    });
    return hit;
  });
  reject("forged library call", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::linalg::GenericOp op) {
      op->setAttr("library_call", mlir::StringAttr::get(&context, "foreign")); hit = true;
    });
    return hit;
  });
  reject("hidden tensor allocation", [&](mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(&function.getBody().front());
    mlir::memref::AllocOp::create(builder, function.getLoc(),
        mlir::MemRefType::get({1}, builder.getF32Type()));
    return true;
  });
  reject("zero fill repeated inside output tiles", [&](mlir::ModuleOp module) {
    mlir::linalg::FillOp fill;
    mlir::linalg::GenericOp generic;
    module.walk([&](mlir::linalg::FillOp op) { fill = op; });
    module.walk([&](mlir::linalg::GenericOp op) { generic = op; });
    if (!fill || !generic) return false;
    fill->moveBefore(generic);
    return true;
  });
  reject("tile bound", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::scf::ForOp op) {
      if (!hit) { op.getUpperBoundMutable().assign(op.getLowerBound()); hit = true; }
    });
    return hit;
  });
  reject("tile step", [&](mlir::ModuleOp module) {
    bool hit = false;
    const auto zero = constant(module, 0);
    module.walk([&](mlir::scf::ForOp op) {
      if (!hit && zero) { op.getStepMutable().assign(zero); hit = true; }
    });
    return hit;
  });
  reject("tile starts late", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::scf::ForOp op) {
      if (!hit) { op.getLowerBoundMutable().assign(op.getStep()); hit = true; }
    });
    return hit;
  });
  reject("tail clipping affine map", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::affine::AffineMinOp op) {
      if (hit) return;
      const auto map = op.getAffineMap();
      auto changed = mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(),
          {mlir::getAffineConstantExpr(64, &context)}, &context);
      op->setAttr("map", mlir::AffineMapAttr::get(changed)); hit = true;
    });
    return hit;
  });
  reject("subview stride", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::memref::SubViewOp op) {
      if (hit) return;
      auto values = op.getStaticStrides();
      std::vector<std::int64_t> strides(values.begin(), values.end());
      if (!strides.empty()) {
        strides.back() = 2;
        op->setAttr("static_strides", mlir::DenseI64ArrayAttr::get(&context, strides));
        hit = true;
      }
    });
    return hit;
  });
  reject("subview misses first reduction term", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::memref::SubViewOp op) {
      if (hit) return;
      auto values = op.getStaticOffsets();
      std::vector<std::int64_t> offsets(values.begin(), values.end());
      if (offsets.size() == 2 && offsets[1] == 0) {
        offsets[1] = 1;
        op->setAttr("static_offsets", mlir::DenseI64ArrayAttr::get(&context, offsets));
        hit = true;
      }
    });
    return hit;
  });
  reject("full K clipped to zero", [&](mlir::ModuleOp module) {
    bool hit = false;
    const auto zero = constant(module, 0);
    module.walk([&](mlir::memref::SubViewOp op) {
      if (hit || !zero) return;
      auto function = op->getParentOfType<mlir::func::FuncOp>();
      if (op.getSource() != function.getArgument(0)) return;
      for (unsigned i = 1; i < op->getNumOperands(); ++i) {
        auto dim = op->getOperand(i).getDefiningOp<mlir::memref::DimOp>();
        auto axis = dim ? dim.getIndex().getDefiningOp<mlir::arith::ConstantIndexOp>()
                        : mlir::arith::ConstantIndexOp{};
        if (dim && axis && axis.value() == 1) {
          op->setOperand(i, zero); hit = true; break;
        }
      }
    });
    return hit;
  });
  reject("reduction iterator parallelized", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::linalg::GenericOp op) {
      mlir::Builder builder(&context);
      op.setIteratorTypesAttr(builder.getArrayAttr({
          mlir::linalg::IteratorTypeAttr::get(&context, mlir::utils::IteratorType::parallel),
          mlir::linalg::IteratorTypeAttr::get(&context, mlir::utils::IteratorType::parallel),
          mlir::linalg::IteratorTypeAttr::get(&context, mlir::utils::IteratorType::parallel)}));
      hit = true;
    });
    return hit;
  });
  reject("wrong K/lane maps", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::linalg::GenericOp op) {
      auto maps = op.getIndexingMapsArray();
      std::swap(maps[0], maps[1]);
      op.setIndexingMapsAttr(mlir::Builder(&context).getAffineMapArrayAttr(maps)); hit = true;
    });
    return hit;
  });
  reject("input orientation", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::linalg::GenericOp op) {
      const auto lhs = op->getOperand(0);
      op->setOperand(0, op->getOperand(1)); op->setOperand(1, lhs); hit = true;
    });
    return hit;
  });
  reject("output store destination is input", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::linalg::GenericOp op) {
      op->setOperand(2, op->getOperand(0)); hit = true;
    });
    return hit;
  });
  reject("negative zero initialization", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::arith::ConstantOp op) {
      if (op.getType().isF32()) {
        op.setValueAttr(mlir::FloatAttr::get(op.getType(), -0.0)); hit = true;
      }
    });
    return hit;
  });
  reject("FMA permission", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::arith::MulFOp op) {
      op.setFastmath(mlir::arith::FastMathFlags::contract); hit = true;
    });
    return hit;
  });
  reject("reassociation permission", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::arith::AddFOp op) {
      op.setFastmath(mlir::arith::FastMathFlags::reassoc); hit = true;
    });
    return hit;
  });
  reject("accumulator overwritten", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::arith::AddFOp op) { op->setOperand(0, op.getRhs()); hit = true; });
    return hit;
  });
  reject("yield stores wrong scalar", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::linalg::GenericOp op) {
      auto yield = mlir::cast<mlir::linalg::YieldOp>(op.getBody()->back());
      yield->setOperand(0, op.getBody()->getArgument(0)); hit = true;
    });
    return hit;
  });
  reject("wrong returned output identity", [&](mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    auto ret = mlir::cast<mlir::func::ReturnOp>(function.getBody().front().back());
    ret->setOperand(0, function.getArgument(0)); return true;
  });
  reject("foreign symbol", [&](mlir::ModuleOp module) {
    auto function = *module.getOps<mlir::func::FuncOp>().begin();
    function.setName("foreign_pattern"); return true;
  });
  check(validCorruptions >= 12, "mutation corpus includes at least twelve well-formed adversaries");

  mlir::OwningOpRef<mlir::ModuleOp> forged = canonical.bufferized->clone();
  (*forged)->setAttr("mdsl.execution_authority", mlir::UnitAttr::get(&context));
  const auto forgedBefore = print(*forged);
  check(!gp::deriveStrictGemmOutputTiledV1(*forged, tiles, error) &&
            !error.empty() && print(*forged) == forgedBefore,
        "forged input fails before a transform and stays unchanged");

  const auto baseline = cc::issueStrictGemmArtifactV1(context, false,
      cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxX86_64,
      cc::StrictCpuIsaV1::Baseline, tiles);
  const auto normalized = normalizedMachine(baseline.llvm_ir);
  check(baseline && !normalized.empty(), "machine normalization positive control");
  for (const auto isa : {cc::StrictCpuIsaV1::AVX, cc::StrictCpuIsaV1::AVX2,
                         cc::StrictCpuIsaV1::AVX512F}) {
    const auto selected = cc::issueStrictGemmArtifactV1(context, false,
        cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxX86_64,
        isa, tiles);
    check(bool(selected), "closed tiled ISA issuer: " + selected.error);
    check(selected && selected.semantic_ir == baseline.semantic_ir &&
              selected.structured_ir == baseline.structured_ir &&
              selected.bufferized_ir == baseline.bufferized_ir &&
              selected.scheduled_ir == baseline.scheduled_ir &&
              selected.transform_ir == baseline.transform_ir &&
              selected.llvm_ir != baseline.llvm_ir &&
              normalizedMachine(selected.llvm_ir) == normalized,
          "target specialization changes symbols/features, not payload or arithmetic");
  }
  const auto arm = cc::issueStrictGemmArtifactV1(context, false,
      cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxAArch64,
      cc::StrictCpuIsaV1::Baseline, tiles);
  check(arm && arm.semantic_ir == baseline.semantic_ir &&
            arm.scheduled_ir == baseline.scheduled_ir &&
            normalizedMachine(arm.llvm_ir) == normalized,
        "closed ARM baseline changes machine triple only; not physical qualification");
  const auto sanitized = cc::issueStrictGemmArtifactV1(context, true,
      cc::StrictGemmScheduleV1::OutputTiledMKN, cc::CpuTargetV1::LinuxX86_64,
      cc::StrictCpuIsaV1::Baseline, tiles);
  check(sanitized && sanitized.scheduled_ir == baseline.scheduled_ir &&
            sanitized.llvm_ir.find("sanitize_address") != std::string::npos,
        "sanitizer realization retains same verified parameterized payload");
  std::printf("output pattern issuer: %u checks, %u valid-IR corruptions, %u failures\n",
              checks, validCorruptions, failures);
  return failures ? 1 : 0;
}
