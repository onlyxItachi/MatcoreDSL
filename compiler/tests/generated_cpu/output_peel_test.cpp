// Independent inspection falsifiers for the additive compiler-private HOW.
// Well-formed submitted MLIR is still not source or execution authority.
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
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace cc = matcore::mdslc::cpu_candidate;
namespace gp = matcore::mdslc::gemm_pattern;
namespace {
unsigned checks = 0, failures = 0, validMutants = 0;
void check(bool condition, const std::string &label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::fprintf(stderr, "FAIL output peel: %s\n", label.c_str());
  }
}
std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream, mlir::OpPrintingFlags().useLocalScope());
  return text;
}
mlir::Value constant(mlir::ModuleOp module, std::int64_t value) {
  mlir::Value result;
  module.walk([&](mlir::arith::ConstantIndexOp op) {
    if (!result && op.value() == value) result = op.getResult();
  });
  return result;
}
struct Roles {
  mlir::func::FuncOp function;
  mlir::scf::ForOp m, full, tail;
  std::vector<mlir::linalg::GenericOp> products;
};
Roles roles(mlir::ModuleOp module) {
  Roles result;
  result.function = *module.getOps<mlir::func::FuncOp>().begin();
  for (auto loop : result.function.front().getOps<mlir::scf::ForOp>()) {
    result.m = loop;
    break;
  }
  if (!result.m) return result;
  for (auto loop : result.m.getBody()->getOps<mlir::scf::ForOp>()) {
    if (!result.full) result.full = loop;
    else result.tail = loop;
  }
  module.walk([&](mlir::linalg::GenericOp op) { result.products.push_back(op); });
  return result;
}
bool replaceOperand(mlir::Operation *operation, mlir::Value oldValue,
                    mlir::Value replacement) {
  bool hit = false;
  for (auto &operand : operation->getOpOperands())
    if (operand.get() == oldValue) { operand.set(replacement); hit = true; }
  return hit;
}
mlir::memref::SubViewOp view(mlir::linalg::GenericOp op, unsigned operand) {
  return op->getOperand(operand).getDefiningOp<mlir::memref::SubViewOp>();
}
} // namespace

int main() {
  mlir::MLIRContext context;
  auto canonical = cc::buildStrictGemmStagesV1(context);
  check(bool(canonical), "canonical strict stages: " + canonical.error);
  if (!canonical) return 1;
  const auto semantic = print(*canonical.semantic);
  const auto structured = print(*canonical.structured);
  const auto buffered = print(*canonical.bufferized);
  std::string error;
  // All requested representative N values, each with every representative M.
  // These are acceptance points, not an empirical claim for every tuple/target.
  constexpr std::array<std::int64_t, 6> axes{1, 2, 3, 5, 16, 64};
  for (auto m : axes) for (auto n : axes) {
    const gp::OutputTilePatternV1 tiles{m, n};
    const auto label = std::to_string(m) + "x" + std::to_string(n);
    auto derived = gp::deriveStrictGemmOutputPeelNV1(*canonical.bufferized, tiles, error);
    check(bool(derived), "derive " + label + ": " + error);
    if (!derived) continue;
    check(gp::verifyStrictGemmOutputPeelNV1(*derived, tiles, error), "verify " + label);
    auto again = gp::deriveStrictGemmOutputPeelNV1(*canonical.bufferized, tiles, error);
    check(again && print(*again) == print(*derived), "deterministic replay " + label);
    const gp::OutputTilePatternV1 wrongM{m == 64 ? 63 : m + 1, n};
    const gp::OutputTilePatternV1 wrongN{m, n == 64 ? 63 : n + 1};
    check(!gp::verifyStrictGemmOutputPeelNV1(*derived, wrongM, error) && !error.empty(),
          "M parameter identity " + label);
    check(!gp::verifyStrictGemmOutputPeelNV1(*derived, wrongN, error) && !error.empty(),
          "N parameter identity " + label);
    check(!cc::verifyStrictGemmBufferizedV1(*derived, error),
          "HOW cannot impersonate canonical WHAT " + label);
    auto old = gp::deriveStrictGemmOutputTiledV1(*canonical.bufferized, tiles, error);
    check(old && gp::verifyStrictGemmOutputTiledV1(*old, tiles, error),
          "existing unpeeled v1 remains accepted " + label);
    if (n == 1) {
      check(old && print(*old) == print(*derived) &&
                gp::strictGemmOutputPeelNTransformV1(tiles, error) ==
                    gp::strictGemmOutputTileTransformV1(tiles, error),
            "unit N is the exact explicit no-tail identity " + label);
    } else {
      const auto graph = roles(*derived);
      check(graph.full && graph.tail && graph.products.size() == 2,
            "full/tail product roles exist " + label);
      check(old && print(*old) != print(*derived) &&
                !gp::verifyStrictGemmOutputTiledV1(*derived, tiles, error),
            "new graph does not broaden the old v1 verifier " + label);
      check(old && !gp::verifyStrictGemmOutputPeelNV1(*old, tiles, error),
            "unpeeled graph cannot impersonate peeled HOW " + label);
    }
    check(print(*canonical.semantic) == semantic &&
              print(*canonical.structured) == structured &&
              print(*canonical.bufferized) == buffered,
          "successful and rejected inspection keep original stages " + label);
  }
  for (const auto tiles : {gp::OutputTilePatternV1{0, 3}, {-1, 3}, {3, 0},
                           {3, -1}, {65, 3}, {3, 65},
                           {std::numeric_limits<std::int64_t>::max(), 2}}) {
    check(gp::strictGemmOutputPeelNTransformV1(tiles, error).empty() && !error.empty(),
          "invalid tuple has no Transform");
    check(!gp::deriveStrictGemmOutputPeelNV1(*canonical.bufferized, tiles, error) &&
              !error.empty() && print(*canonical.bufferized) == buffered,
          "invalid tuple cannot derive or mutate its input");
  }

  const gp::OutputTilePatternV1 tiles{5, 3};
  auto scheduled = gp::deriveStrictGemmOutputPeelNV1(*canonical.bufferized, tiles, error);
  if (!scheduled) return 1;
  const auto goodText = print(*scheduled);
  const auto reject = [&](const std::string &label, auto mutate) {
    mlir::OwningOpRef<mlir::ModuleOp> bad = scheduled->clone();
    const bool hit = mutate(*bad);
    check(hit && print(*bad) != goodText, "mutation reaches payload: " + label);
    if (!hit) return;
    mlir::ScopedDiagnosticHandler diagnostics(&context, [](mlir::Diagnostic &) {
      return mlir::success();
    });
    const bool valid = mlir::succeeded(mlir::verify(*bad));
    check(valid, "adversary must remain valid MLIR: " + label);
    if (!valid) return; // An invalid-IR refusal is NOT a semantic falsifier.
    ++validMutants;
    const auto badText = print(*bad);
    error.clear();
    const bool accepted = gp::verifyStrictGemmOutputPeelNV1(*bad, tiles, error);
    check(!accepted &&
              error.find("peeled scalar envelope:") != std::string::npos &&
              error.find("upstream replay") == std::string::npos,
          "independent envelope rejects adversary before replay: " + label + ": " + error);
    check(print(*bad) == badText && print(*scheduled) == goodText &&
              print(*canonical.bufferized) == buffered,
          "inspection leaves adversary and original untouched: " + label);
  };
  for (int adjustment : {-1, 1}) {
    reject("split off by " + std::to_string(adjustment), [&](mlir::ModuleOp module) {
      auto op = roles(module).full.getUpperBound().getDefiningOp<mlir::affine::AffineApplyOp>();
      if (!op) return false;
      const auto map = op.getAffineMap();
      op->setAttr("map", mlir::AffineMapAttr::get(mlir::AffineMap::get(
          map.getNumDims(), map.getNumSymbols(), {map.getResult(0) + adjustment}, &context)));
      return true;
    });
  }
  reject("split uses M instead of N", [&](mlir::ModuleOp module) {
    auto graph = roles(module);
    auto op = graph.full.getUpperBound().getDefiningOp<mlir::affine::AffineApplyOp>();
    if (!op || op->getNumOperands() != 1) return false;
    op->setOperand(0, graph.m.getUpperBound()); return true;
  });
  reject("full range overlaps tail", [&](mlir::ModuleOp module) {
    auto graph = roles(module);
    graph.full.getUpperBoundMutable().assign(graph.tail.getUpperBound()); return true;
  });
  reject("full range omitted", [&](mlir::ModuleOp module) {
    roles(module).full.getUpperBoundMutable().assign(constant(module, 0)); return true;
  });
  reject("full origins skip first tile", [&](mlir::ModuleOp module) {
    auto full = roles(module).full;
    full.getLowerBoundMutable().assign(full.getStep()); return true;
  });
  reject("tail starts at zero and overlaps full", [&](mlir::ModuleOp module) {
    roles(module).tail.getLowerBoundMutable().assign(constant(module, 0)); return true;
  });
  reject("tail starts late and leaves a gap", [&](mlir::ModuleOp module) {
    auto tail = roles(module).tail;
    mlir::OpBuilder builder(tail);
    auto one = mlir::arith::ConstantIndexOp::create(builder, tail.getLoc(), 1);
    auto shifted = mlir::arith::AddIOp::create(builder, tail.getLoc(), tail.getLowerBound(), one);
    tail.getLowerBoundMutable().assign(shifted.getResult()); return true;
  });
  reject("tail range omitted", [&](mlir::ModuleOp module) {
    auto tail = roles(module).tail;
    tail.getUpperBoundMutable().assign(tail.getLowerBound()); return true;
  });
  for (unsigned branch : {0U, 1U}) {
    const auto suffix = branch ? " (tail)" : " (full)";
    reject(std::string("N step becomes one") + suffix, [&](mlir::ModuleOp module) {
      auto graph = roles(module);
      (branch ? graph.tail : graph.full).getStepMutable().assign(constant(module, 1));
      return true;
    });
    for (unsigned operand : {1U, 2U}) {
      reject(std::string(operand == 1 ? "B column origin reset" : "C column origin reset") + suffix,
             [&](mlir::ModuleOp module) {
        auto graph = roles(module);
        auto subview = view(graph.products[branch], operand);
        return subview && replaceOperand(subview, (branch ? graph.tail : graph.full).getInductionVar(),
                                         constant(module, 0));
      });
    }
    reject(std::string("complete K clipped on both inputs") + suffix, [&](mlir::ModuleOp module) {
      auto product = roles(module).products[branch];
      auto a = view(product, 0), b = view(product, 1);
      if (!a || !b) return false;
      auto k = mlir::dyn_cast<mlir::Value>(a.getMixedSizes()[1]);
      return k && replaceOperand(a, k, constant(module, 0)) &&
             replaceOperand(b, k, constant(module, 0));
    });
    reject(std::string("K starts at one") + suffix, [&](mlir::ModuleOp module) {
      auto product = roles(module).products[branch];
      for (unsigned operand : {0U, 1U}) {
        auto subview = view(product, operand);
        if (!subview) return false;
        const auto original = subview.getStaticOffsets();
        std::vector<std::int64_t> offsets(original.begin(), original.end());
        offsets[operand == 0 ? 1 : 0] = 1;
        subview->setAttr("static_offsets", mlir::DenseI64ArrayAttr::get(&context, offsets));
      }
      return true;
    });
    reject(std::string("input K stride becomes two") + suffix, [&](mlir::ModuleOp module) {
      auto product = roles(module).products[branch];
      for (unsigned operand : {0U, 1U}) {
        auto subview = view(product, operand);
        if (!subview) return false;
        const auto original = subview.getStaticStrides();
        std::vector<std::int64_t> strides(original.begin(), original.end());
        strides[operand == 0 ? 1 : 0] = 2;
        subview->setAttr("static_strides", mlir::DenseI64ArrayAttr::get(&context, strides));
        subview.getResult().setType(mlir::memref::SubViewOp::inferResultType(
            subview.getSourceType(), subview.getStaticOffsets(), subview.getStaticSizes(), strides));
      }
      return true;
    });
    reject(std::string("input orientation swapped") + suffix, [&](mlir::ModuleOp module) {
      auto product = roles(module).products[branch];
      const auto a = product->getOperand(0);
      product->setOperand(0, product->getOperand(1));
      product->setOperand(1, a); return true;
    });
    reject(std::string("accumulator replaced by product") + suffix, [&](mlir::ModuleOp module) {
      bool hit = false;
      roles(module).products[branch].walk([&](mlir::arith::AddFOp op) {
        op->setOperand(0, op.getRhs()); hit = true;
      });
      return hit;
    });
    reject(std::string("yield omits addition") + suffix, [&](mlir::ModuleOp module) {
      auto product = roles(module).products[branch];
      auto yield = mlir::cast<mlir::linalg::YieldOp>(product.getBody()->back());
      yield->setOperand(0, product.getBody()->getArgument(0)); return true;
    });
    reject(std::string("destination aliases input A") + suffix, [&](mlir::ModuleOp module) {
      auto graph = roles(module);
      auto destination = view(graph.products[branch], 2);
      if (!destination) return false;
      destination->setOperand(0, graph.function.getArgument(0)); return true;
    });
    reject(std::string("multiply contraction permission") + suffix, [&](mlir::ModuleOp module) {
      bool hit = false;
      roles(module).products[branch].walk([&](mlir::arith::MulFOp op) {
        op.setFastmath(mlir::arith::FastMathFlags::contract); hit = true;
      });
      return hit;
    });
    reject(std::string("addition reassociation permission") + suffix, [&](mlir::ModuleOp module) {
      bool hit = false;
      roles(module).products[branch].walk([&](mlir::arith::AddFOp op) {
        op.setFastmath(mlir::arith::FastMathFlags::reassoc); hit = true;
      });
      return hit;
    });
  }
  reject("full C subview row origin reset", [&](mlir::ModuleOp module) {
    auto graph = roles(module);
    auto destination = view(graph.products[0], 2);
    return destination && replaceOperand(destination, graph.m.getInductionVar(), constant(module, 0));
  });
  reject("tail width padded to full tile", [&](mlir::ModuleOp module) {
    auto graph = roles(module);
    auto b = view(graph.products[1], 1), c = view(graph.products[1], 2);
    if (!b || !c) return false;
    auto width = mlir::dyn_cast<mlir::Value>(b.getMixedSizes()[1]);
    return width && replaceOperand(b, width, constant(module, 3)) &&
           replaceOperand(c, width, constant(module, 3));
  });
  reject("full local width becomes tile plus one", [&](mlir::ModuleOp module) {
    auto product = roles(module).products[0];
    for (unsigned operand : {1U, 2U}) {
      auto subview = view(product, operand);
      if (!subview) return false;
      const auto original = subview.getStaticSizes();
      std::vector<std::int64_t> sizes(original.begin(), original.end());
      sizes[1] = 4;
      subview->setAttr("static_sizes", mlir::DenseI64ArrayAttr::get(&context, sizes));
      subview.getResult().setType(mlir::memref::SubViewOp::inferResultType(
          subview.getSourceType(), subview.getStaticOffsets(), sizes, subview.getStaticStrides()));
    }
    return true;
  });
  reject("fill repeated inside full N tiles", [&](mlir::ModuleOp module) {
    mlir::linalg::FillOp fill;
    module.walk([&](mlir::linalg::FillOp op) { fill = op; });
    if (!fill) return false;
    fill->moveBefore(roles(module).products[0]); return true;
  });
  reject("positive-zero fill removed", [&](mlir::ModuleOp module) {
    mlir::linalg::FillOp fill;
    module.walk([&](mlir::linalg::FillOp op) { fill = op; });
    if (!fill) return false;
    fill.erase(); return true;
  });
  reject("negative-zero fill", [&](mlir::ModuleOp module) {
    bool hit = false;
    module.walk([&](mlir::arith::ConstantOp op) {
      if (op.getType().isF32()) { op.setValueAttr(mlir::FloatAttr::get(op.getType(), -0.0)); hit = true; }
    });
    return hit;
  });
  reject("hidden allocation effect", [&](mlir::ModuleOp module) {
    auto function = roles(module).function;
    mlir::OpBuilder builder(&function.front(), function.front().begin());
    mlir::memref::AllocOp::create(builder, function.getLoc(),
        mlir::MemRefType::get({1}, builder.getF32Type()));
    return true;
  });
  reject("forged source authority", [&](mlir::ModuleOp module) {
    module->setAttr("mdsl.execution_authority", mlir::UnitAttr::get(&context)); return true;
  });
  reject("forged input noalias", [&](mlir::ModuleOp module) {
    roles(module).products[1]->setAttr("mdsl.noalias", mlir::UnitAttr::get(&context)); return true;
  });
  reject("return wrong destination", [&](mlir::ModuleOp module) {
    auto function = roles(module).function;
    mlir::cast<mlir::func::ReturnOp>(function.front().back())->setOperand(0, function.getArgument(0));
    return true;
  });
  // Harmless dead integer glue must pass the semantic envelope but still fail
  // exact recipe replay. This separates two obligations instead of allowing
  // replay to mask an incomplete independent graph proof above.
  mlir::OwningOpRef<mlir::ModuleOp> replayOnly = scheduled->clone();
  auto function = roles(*replayOnly).function;
  mlir::OpBuilder builder(&function.front(), function.front().begin());
  mlir::arith::ConstantIndexOp::create(builder, function.getLoc(), 42);
  check(mlir::succeeded(mlir::verify(*replayOnly)), "benign replay mismatch is valid MLIR");
  error.clear();
  const bool replayAccepted = gp::verifyStrictGemmOutputPeelNV1(*replayOnly, tiles, error);
  check(!replayAccepted &&
            error.find("peeled payload differs from parameter-bound upstream replay") != std::string::npos &&
            error.find("peeled scalar envelope:") == std::string::npos,
        "benign graph-equivalent change reaches only replay: " + error);
  mlir::OwningOpRef<mlir::ModuleOp> forged = canonical.bufferized->clone();
  (*forged)->setAttr("mdsl.execution_authority", mlir::UnitAttr::get(&context));
  const auto forgedText = print(*forged);
  check(!gp::deriveStrictGemmOutputPeelNV1(*forged, tiles, error) && !error.empty() &&
            print(*forged) == forgedText,
        "forged canonical input cannot acquire derivation authority");
  check(validMutants >= 35, "at least 35 independently valid semantic adversaries");
  std::printf("output peel inspection: %u checks, %u valid mutants, %u failures\n",
              checks, validMutants, failures);
  return failures != 0;
}
