#include "MatcoreCpuGemmCandidate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
  std::cout << "strict CPU candidate: " << checks << " checks, " << failures
            << " failures\n";
  return failures != 0;
}
