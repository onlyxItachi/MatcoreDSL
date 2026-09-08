#include "MatcoreCpuGemmCandidate.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include <iostream>

namespace candidate = matcore::mdslc::cpu_candidate;
int checks = 0, failures = 0;
void check(bool condition, const std::string &label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << label << '\n';
  }
}
int main() {
  mlir::MLIRContext context;
  auto stages = candidate::buildStrictGemmStagesV1(context);
  check(bool(stages), "semantic-directed stages: " + stages.error);
  if (!stages)
    return 1;
  std::string error;
  check(candidate::verifyStrictGemmStructuredV1(*stages.structured, error),
        "structured verifies");
  check(candidate::verifyStrictGemmBufferizedV1(*stages.bufferized, error),
        "buffer verifies");
  check(!candidate::verifyStrictGemmStructuredV1(*stages.semantic, error),
        "inspection semantic witness is not a candidate");
  auto reject = [&](bool buffer, auto mutate, const std::string &label) {
    mlir::OwningOpRef<mlir::ModuleOp> bad =
        buffer ? stages.bufferized->clone() : stages.structured->clone();
    mutate(*bad);
    check(!(buffer ? candidate::verifyStrictGemmBufferizedV1(*bad, error)
                   : candidate::verifyStrictGemmStructuredV1(*bad, error)) &&
              !error.empty(),
          label);
  };
  for (bool buffer : {false, true}) {
    reject(
        buffer,
        [&](mlir::ModuleOp m) {
          m->setAttr("mdsl.execution_authority",
                     mlir::StringAttr::get(&context, "trusted"));
        },
        "forged authority");
    reject(
        buffer,
        [&](mlir::ModuleOp m) {
          m.walk([&](mlir::linalg::MatmulOp op) {
            op->setAttr("mdsl.noalias", mlir::UnitAttr::get(&context));
          });
        },
        "forged nested noalias");
    reject(
        buffer,
        [&](mlir::ModuleOp m) {
          m.walk([&](mlir::linalg::MatmulOp op) {
            auto a = op->getOperand(0);
            op->setOperand(0, op->getOperand(1));
            op->setOperand(1, a);
          });
        },
        "swapped lhs/rhs");
    reject(
        buffer,
        [&](mlir::ModuleOp m) {
          m.walk([&](mlir::arith::ConstantOp op) {
            op.setValueAttr(
                mlir::FloatAttr::get(mlir::Float32Type::get(&context), -0.0));
          });
        },
        "negative-zero seed");
    reject(
        buffer,
        [&](mlir::ModuleOp m) {
          m.walk([&](mlir::arith::MulFOp op) {
            op.setFastmath(mlir::arith::FastMathFlags::contract);
          });
        },
        "FMA permission");
    reject(
        buffer,
        [&](mlir::ModuleOp m) {
          m.walk([&](mlir::arith::AddFOp op) {
            op.setFastmath(mlir::arith::FastMathFlags::reassoc);
          });
        },
        "reduction reassociation");
    reject(
        buffer,
        [&](mlir::ModuleOp m) {
          m.walk([&](mlir::arith::AddFOp op) {
            op->setOperand(0, op->getOperand(1));
          });
        },
        "wrong reduction accumulator");
    reject(
        buffer,
        [&](mlir::ModuleOp m) {
          auto fn = *m.getOps<mlir::func::FuncOp>().begin();
          fn.setName("forged_kernel");
        },
        "forged symbol");
  }
  reject(
      true,
      [&](mlir::ModuleOp m) {
        auto fn = *m.getOps<mlir::func::FuncOp>().begin();
        mlir::cast<mlir::func::ReturnOp>(fn.getBody().front().back())
            ->setOperand(0, fn.getArgument(0));
      },
      "wrong returned physical buffer");
  reject(
      true,
      [&](mlir::ModuleOp m) {
        m.walk([&](mlir::linalg::MatmulOp op) {
          op->setOperand(2, op->getOperand(0));
        });
      },
      "contraction destination aliases input");
  reject(
      false,
      [&](mlir::ModuleOp m) {
        auto fn = *m.getOps<mlir::func::FuncOp>().begin();
        mlir::OpBuilder builder(&context);
        builder.setInsertionPointToStart(&fn.getBody().front());
        mlir::arith::ConstantOp::create(builder, fn.getLoc(),
                                        builder.getF32FloatAttr(42.0));
      },
      "extra structured operation");
  reject(
      false,
      [&](mlir::ModuleOp m) {
        auto encoded = mlir::RankedTensorType::get(
            {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic},
            mlir::Float32Type::get(&context),
            mlir::StringAttr::get(&context, "forged_target_layout"));
        auto fn = *m.getOps<mlir::func::FuncOp>().begin();
        fn.setType(mlir::FunctionType::get(
            &context, {encoded, encoded, encoded}, {encoded}));
        for (auto argument : fn.getArguments())
          argument.setType(encoded);
        m.walk([&](mlir::Operation *op) {
          for (auto result : op->getResults())
            if (mlir::isa<mlir::RankedTensorType>(result.getType()))
              result.setType(encoded);
        });
      },
      "independent adversary: forged target/layout tensor encoding");
  auto first = candidate::issueStrictGemmArtifactV1(context, false);
  auto second = candidate::issueStrictGemmArtifactV1(context, false);
  auto sanitized = candidate::issueStrictGemmArtifactV1(context, true);
  check(bool(first), "artifact issued: " + first.error);
  check(bool(second) && first.llvm_ir == second.llvm_ir &&
            first.manifest == second.manifest,
        "deterministic issuance");
  check(first.semantic_ir.find("inspection_only_no_execution") !=
            std::string::npos,
        "semantic witness authority unchanged");
  check(first.llvm_ir.find("fmul float") != std::string::npos &&
            first.llvm_ir.find("fadd float") != std::string::npos,
        "separate scalar LLVM arithmetic");
  check(first.llvm_ir.find("llvm.fma") == std::string::npos &&
            first.llvm_ir.find(" fast ") == std::string::npos,
        "no FMA/fast math");
  check(sanitized &&
            sanitized.llvm_ir.find("sanitize_address") != std::string::npos &&
            sanitized.manifest != first.manifest,
        "generated sanitizer attributes are bound into artifact");
  auto row = candidate::deriveStrictGemmRowContiguousV1(*stages.bufferized, error);
  check(bool(row), "row-contiguous upstream Transform derivation: " + error);
  if (row) {
    check(candidate::verifyStrictGemmRowContiguousV1(*row, error),
          "exact row-contiguous structure verifies");
    check(candidate::verifyStrictGemmBufferizedV1(*stages.bufferized, error),
          "upstream schedule preserves its input witness");
    check(!candidate::verifyStrictGemmBufferizedV1(*row, error),
          "scheduled witness does not impersonate original pairing");
    auto rejectRow = [&](auto mutate, const std::string &label) {
      mlir::OwningOpRef<mlir::ModuleOp> bad = row->clone();
      mutate(*bad);
      check(!candidate::verifyStrictGemmRowContiguousV1(*bad, error) &&
                !error.empty(), label);
    };
    rejectRow([&](mlir::ModuleOp m) {
      m.walk([&](mlir::linalg::GenericOp op) {
        op->setAttr("library_call", mlir::StringAttr::get(&context, "forged"));
      });
    }, "schedule cannot inject an external library call");
    rejectRow([&](mlir::ModuleOp m) {
      m.walk([&](mlir::arith::MulFOp op) {
        op.setFastmath(mlir::arith::FastMathFlags::contract);
      });
    }, "schedule cannot grant FMA");
    rejectRow([&](mlir::ModuleOp m) {
      m.walk([&](mlir::arith::AddFOp op) {
        op.setFastmath(mlir::arith::FastMathFlags::reassoc);
      });
    }, "row schedule cannot grant reduction reassociation");
    rejectRow([&](mlir::ModuleOp m) {
      m.walk([&](mlir::arith::AddFOp op) {
        op->setOperand(0, op->getOperand(1));
      });
    }, "row schedule cannot replace the prior accumulator");
    rejectRow([&](mlir::ModuleOp m) {
      m.walk([&](mlir::linalg::GenericOp op) {
        op.setIteratorTypesAttr(mlir::Builder(&context).getArrayAttr({
            mlir::linalg::IteratorTypeAttr::get(&context, mlir::utils::IteratorType::parallel),
            mlir::linalg::IteratorTypeAttr::get(&context, mlir::utils::IteratorType::parallel),
            mlir::linalg::IteratorTypeAttr::get(&context, mlir::utils::IteratorType::reduction)}));
      });
    }, "row schedule cannot change the reduction iterator");
    rejectRow([&](mlir::ModuleOp m) {
      m.walk([&](mlir::linalg::GenericOp op) {
        auto input = op->getOperand(0);
        op->setOperand(0, op->getOperand(1));
        op->setOperand(1, input);
      });
    }, "schedule cannot commute operands");
    rejectRow([&](mlir::ModuleOp m) {
      m.walk([&](mlir::linalg::GenericOp op) {
        op->setOperand(2, op->getOperand(0));
      });
    }, "schedule cannot change isolated destination");
    rejectRow([&](mlir::ModuleOp m) {
      m.walk([&](mlir::linalg::GenericOp op) {
        auto maps = op.getIndexingMapsArray();
        std::swap(maps[0], maps[1]);
        op.setIndexingMapsAttr(mlir::Builder(&context).getAffineMapArrayAttr(maps));
      });
    }, "schedule cannot change K or lane indexing");
  }
  auto rowArtifact = candidate::issueStrictGemmArtifactV1(
      context, false, candidate::StrictGemmScheduleV1::RowContiguousMKN);
  auto repeatedRow = candidate::issueStrictGemmArtifactV1(
      context, false, candidate::StrictGemmScheduleV1::RowContiguousMKN);
  check(bool(rowArtifact), "row-contiguous artifact issuance: " + rowArtifact.error);
  check(rowArtifact && repeatedRow && rowArtifact.manifest == repeatedRow.manifest &&
            rowArtifact.llvm_ir == repeatedRow.llvm_ir &&
            rowArtifact.semantic_ir == first.semantic_ir &&
            rowArtifact.llvm_ir != first.llvm_ir &&
            rowArtifact.manifest != first.manifest &&
            !rowArtifact.transform_ir.empty(),
        "deterministic schedule identity changes realization, not semantics");
  check(!candidate::issueStrictGemmArtifactV1(
      context, false, static_cast<candidate::StrictGemmScheduleV1>(99)),
      "unknown schedule cannot fall back or issue authority");

  auto tiled = candidate::deriveStrictGemmCacheTiledV1(*stages.bufferized, error);
  check(bool(tiled), "cache-tiled upstream derivation: " + error);
  check(!candidate::verifyStrictGemmCacheTiledV1({}, error) && !error.empty(),
        "null tiled evidence fails closed");
  if (tiled) {
    check(candidate::verifyStrictGemmCacheTiledV1(*tiled, error),
          "exact cache-tiled structure verifies");
    check(candidate::verifyStrictGemmBufferizedV1(*stages.bufferized, error),
          "cache tiling leaves original buffer witness untouched");
    check(!candidate::verifyStrictGemmBufferizedV1(*tiled, error) &&
              !candidate::verifyStrictGemmRowContiguousV1(*tiled, error),
          "tiled result cannot impersonate original or row witness");
    auto rejectTiled = [&](auto mutate, const std::string &label) {
      mlir::OwningOpRef<mlir::ModuleOp> bad = tiled->clone();
      mutate(*bad);
      check(!candidate::verifyStrictGemmCacheTiledV1(*bad, error) &&
                !error.empty(), label);
    };
    auto kLoop = [](mlir::ModuleOp m) {
      mlir::scf::ForOp result;
      m.walk([&](mlir::scf::ForOp op) {
        // Only the K loop directly contains the scheduled contraction.
        if (!op.getBody()->getOps<mlir::linalg::GenericOp>().empty()) result = op;
      });
      return result;
    };
    rejectTiled([&](mlir::ModuleOp m) {
      auto k = kLoop(m);
      auto n = mlir::cast<mlir::scf::ForOp>(k->getParentOp());
      k.setUpperBound(n.getUpperBound());
    }, "cache tiling cannot replace K bound by N");
    rejectTiled([&](mlir::ModuleOp m) {
      auto k = kLoop(m);
      mlir::OpBuilder b(k);
      auto one = mlir::arith::ConstantOp::create(b, k.getLoc(), b.getIndexAttr(1));
      k.setLowerBound(one);
    }, "cache tiling cannot skip the first K element");
    rejectTiled([&](mlir::ModuleOp m) {
      auto k = kLoop(m);
      mlir::OpBuilder b(k);
      auto step = mlir::arith::ConstantOp::create(b, k.getLoc(), b.getIndexAttr(64));
      k.setStep(step);
    }, "cache tiling cannot skip K chunks through a different step");
    rejectTiled([&](mlir::ModuleOp m) {
      auto k = kLoop(m);
      m.walk([&](mlir::affine::AffineMinOp op) {
        if (op.getOperand(0) == k.getInductionVar()) {
          auto d = mlir::getAffineDimExpr(0, &context);
          auto s = mlir::getAffineSymbolExpr(0, &context);
          op->setAttr("map", mlir::AffineMapAttr::get(
              mlir::AffineMap::get(1, 1, {s - d, mlir::getAffineConstantExpr(31, &context)}, &context)));
        }
      });
    }, "cache tiling cannot omit the last element of each K tile");
    rejectTiled([&](mlir::ModuleOp m) {
      auto fn = *m.getOps<mlir::func::FuncOp>().begin();
      m.walk([&](mlir::memref::SubViewOp op) {
        if (op.getSource() == fn.getArgument(2))
          op->setOperand(0, fn.getArgument(0));
      });
    }, "cache tiling cannot update an input instead of private C");
    rejectTiled([&](mlir::ModuleOp m) {
      auto fn = *m.getOps<mlir::func::FuncOp>().begin();
      auto k = kLoop(m);
      m.walk([&](mlir::memref::SubViewOp op) {
        if (op.getSource() == fn.getArgument(2))
          op.getOffsetsMutable()[0].set(k.getInductionVar());
      });
    }, "cache tiling cannot use a K offset for C rows");
    rejectTiled([&](mlir::ModuleOp m) {
      auto fn = *m.getOps<mlir::func::FuncOp>().begin();
      m.walk([&](mlir::memref::SubViewOp op) {
        if (op.getSource() == fn.getArgument(2))
          op.getSizesMutable()[0].set(op.getSizes()[1]);
      });
    }, "cache tiling cannot use the N tail size for C rows");
    rejectTiled([&](mlir::ModuleOp m) {
      mlir::linalg::FillOp fill;
      mlir::linalg::GenericOp generic;
      m.walk([&](mlir::linalg::FillOp op) { fill = op; });
      m.walk([&](mlir::linalg::GenericOp op) { generic = op; });
      fill->moveBefore(generic);
    }, "cache tiling cannot re-zero C inside each K chunk");
    rejectTiled([&](mlir::ModuleOp m) {
      auto fn = *m.getOps<mlir::func::FuncOp>().begin();
      mlir::OpBuilder b(&fn.getBody().front(), fn.getBody().front().begin());
      mlir::memref::AllocaOp::create(b, fn.getLoc(),
          mlir::MemRefType::get({4, 64}, b.getF32Type()));
    }, "cache tiling cannot introduce private tensor allocation");
    rejectTiled([&](mlir::ModuleOp m) {
      auto fn = *m.getOps<mlir::func::FuncOp>().begin();
      mlir::OpBuilder b(&fn.getBody().front(), fn.getBody().front().begin());
      mlir::memref::CopyOp::create(b, fn.getLoc(), fn.getArgument(0), fn.getArgument(2));
    }, "cache tiling cannot add a hidden copy");
    rejectTiled([&](mlir::ModuleOp m) {
      m.walk([&](mlir::arith::MulFOp op) {
        op.setFastmath(mlir::arith::FastMathFlags::contract);
      });
    }, "cache tiling cannot acquire FMA permission");
    rejectTiled([&](mlir::ModuleOp m) {
      m.walk([&](mlir::arith::AddFOp op) {
        op->setOperand(0, op->getOperand(1));
      });
    }, "cache tiling cannot replace prior rounded accumulator");
    rejectTiled([&](mlir::ModuleOp m) {
      m.walk([&](mlir::arith::AddFOp op) {
        auto previous = op.getLhs();
        op->setOperand(0, op.getRhs());
        op->setOperand(1, previous);
      });
    }, "exact replay retains scalar accumulator operand order");
    rejectTiled([&](mlir::ModuleOp m) {
      m.walk([&](mlir::arith::MulFOp op) {
        auto previous = op.getLhs();
        op->setOperand(0, op.getRhs());
        op->setOperand(1, previous);
      });
    }, "exact replay retains scalar multiplication operand order");
    rejectTiled([&](mlir::ModuleOp m) {
      m.walk([&](mlir::linalg::GenericOp op) {
        auto maps = op.getIndexingMapsArray();
        std::swap(maps[0], maps[1]);
        op.setIndexingMapsAttr(mlir::Builder(&context).getAffineMapArrayAttr(maps));
      });
    }, "cache tiling cannot commute contraction indexing");
    rejectTiled([&](mlir::ModuleOp m) {
      m.walk([&](mlir::linalg::GenericOp op) {
        op->setAttr("library_call", mlir::StringAttr::get(&context, "forged"));
      });
    }, "cache tiling cannot add a provider call");
    rejectTiled([&](mlir::ModuleOp m) {
      auto fn = *m.getOps<mlir::func::FuncOp>().begin();
      mlir::cast<mlir::func::ReturnOp>(fn.getBody().front().back())
          ->setOperand(0, fn.getArgument(0));
    }, "cache tiling cannot change returned destination identity");
  }
  auto tiledArtifact = candidate::issueStrictGemmArtifactV1(
      context, false, candidate::StrictGemmScheduleV1::CacheTiledMKN);
  auto repeatedTiled = candidate::issueStrictGemmArtifactV1(
      context, false, candidate::StrictGemmScheduleV1::CacheTiledMKN);
  auto sanitizedTiled = candidate::issueStrictGemmArtifactV1(
      context, true, candidate::StrictGemmScheduleV1::CacheTiledMKN);
  check(bool(tiledArtifact), "cache-tiled artifact issuance: " + tiledArtifact.error);
  check(tiledArtifact && repeatedTiled &&
            tiledArtifact.manifest == repeatedTiled.manifest &&
            tiledArtifact.llvm_ir == repeatedTiled.llvm_ir &&
            tiledArtifact.semantic_ir == first.semantic_ir &&
            tiledArtifact.structured_ir == first.structured_ir &&
            tiledArtifact.bufferized_ir == first.bufferized_ir &&
            tiledArtifact.llvm_ir != rowArtifact.llvm_ir &&
            tiledArtifact.manifest.find("schedule=cache-tiled-4x64x32-mkn") != std::string::npos,
        "tiled realization identity is deterministic without semantic drift");
  check(sanitizedTiled &&
            sanitizedTiled.llvm_ir.find("sanitize_address") != std::string::npos &&
            sanitizedTiled.semantic_ir == first.semantic_ir,
        "cache-tiled generated sanitizer attributes preserve semantic identity");
  std::cout << "strict CPU candidate: " << checks << " checks, " << failures
            << " failures\n";
  return failures != 0;
}
