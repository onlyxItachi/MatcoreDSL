#include "MatcoreSpirvGemmCandidate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <limits>

int main() {
  namespace candidate = matcore::mdslc::spirv_candidate;
  namespace cpu = matcore::mdslc::cpu_candidate;
  using namespace mlir;
  MLIRContext context;
  unsigned checks = 0;
  std::string error;
  const auto require = [&](bool condition, llvm::StringRef name) {
    ++checks;
    if (!condition) llvm::errs() << "FAIL " << name << ": " << error << '\n';
    return condition;
  };
  using Shape = candidate::StaticGemmShapeV1;
  const Shape accepted[] = {{1, 1, 1}, {2, 3, 4}, {1, 7, 9}, {9, 7, 1},
      {3, 5, 7}, {17, 31, 13}, {65, 3, 67}, {65535, 1, 1}, {1, 1, 65535},
      {1, 33554432, 1}, {4096, 1, 8192}, {65535, 1, 512}};
  for (auto shape : accepted)
    if (!require(candidate::validateStaticGemmShapeV1(shape, error), "accepted family shape")) return 1;
  const Shape rejected[] = {{0, 3, 4}, {2, 0, 4}, {2, 3, 0}, {-1, 3, 4},
      {2, -1, 4}, {2, 3, -1}, {65536, 1, 1}, {1, 1, 65536},
      {1, 33554433, 1}, {4096, 1, 8193}, {65535, 1, 513},
      {1, INT32_MAX, 1}, {INT64_MAX, INT64_MAX, INT64_MAX},
      {INT64_MIN, 1, 1}};
  for (auto shape : rejected)
    if (!require(!candidate::validateStaticGemmShapeV1(shape, error) && !error.empty(),
                 "rejected family shape")) return 1;
  for (auto shape : {Shape{INT64_MAX, 1, 1}, Shape{1, INT64_MAX, 1},
                     Shape{1, 1, INT64_MAX}, Shape{INT64_MAX / 4, 5, 1}})
    if (!require(!cpu::buildStrictGemmStagesV1(context, shape),
                 "CPU static specialization rejects unrepresentable byte products")) return 1;

  auto artifact = candidate::issueStrictGemmArtifactV1(context, {2, 3, 4});
  error = artifact.error;
  if (!require(bool(artifact), "control closed issuance") ||
      !require(candidate::verifyStrictGemmArtifactV1(context, artifact, error),
               "complete control replay")) return 1;
  if (!require(artifact.fill_abi.buffers.size() == 1 &&
               artifact.fill_abi.buffers[0].bytes == 32 &&
               artifact.fill_abi.buffers[0].binding == 0 &&
               artifact.gemm_abi.buffers.size() == 3 &&
               artifact.gemm_abi.buffers[0].bytes == 24 &&
               artifact.gemm_abi.buffers[1].bytes == 48 &&
               artifact.gemm_abi.buffers[2].bytes == 32 &&
               artifact.gemm_abi.buffers[2].binding == 2 &&
               artifact.gemm_abi.dispatch == std::array<std::uint32_t, 3>{2, 4, 1} &&
               artifact.gemm_abi.local_size == std::array<std::uint32_t, 3>{1, 1, 1},
               "control exact independently expected ABI")) return 1;
  for (auto shape : accepted) {
    auto current = candidate::issueStrictGemmArtifactV1(context, shape);
    error = current.error;
    if (!require(bool(current), "family compiler issuance")) return 1;
    auto outlined = parseSourceString<ModuleOp>(current.outlined_ir, &context);
    auto fill = parseSourceString<spirv::ModuleOp>(current.fill_spirv_ir, &context);
    auto gemm = parseSourceString<spirv::ModuleOp>(current.gemm_spirv_ir, &context);
    if (!require(outlined && candidate::verifyStrictGemmOutlinedV1(*outlined, shape, error),
                 "independent outlined family inspection") ||
        !require(fill && candidate::verifyStrictGemmShaderV1(*fill, shape, true, error),
                 "independent fill family inspection") ||
        !require(gemm && candidate::verifyStrictGemmShaderV1(*gemm, shape, false, error),
                 "independent GEMM family inspection")) return 1;
  }
  const auto artifactAttack = [&](llvm::StringRef name,
      std::function<void(candidate::StrictGemmArtifactV1 &)> mutate) {
    auto changed = artifact;
    mutate(changed);
    return require(!candidate::verifyStrictGemmArtifactV1(context, changed, error), name);
  };
  if (!artifactAttack("shape binding", [](auto &a) { a.shape.k = 4; }) ||
      !artifactAttack("semantic identity", [](auto &a) { a.semantic_ir += "\n"; }) ||
      !artifactAttack("structured identity", [](auto &a) { a.structured_ir += "\n"; }) ||
      !artifactAttack("bufferized identity", [](auto &a) { a.bufferized_ir += "\n"; }) ||
      !artifactAttack("outlined identity", [](auto &a) { a.outlined_ir += "\n"; }) ||
      !artifactAttack("final fill graph identity", [](auto &a) { a.fill_spirv_ir += "\n"; }) ||
      !artifactAttack("final GEMM graph identity", [](auto &a) { a.gemm_spirv_ir += "\n"; }) ||
      !artifactAttack("fill dispatch binding", [](auto &a) { a.fill_abi.dispatch[0]++; }) ||
      !artifactAttack("GEMM local geometry binding", [](auto &a) { a.gemm_abi.local_size[0]++; }) ||
      !artifactAttack("GEMM storage range binding", [](auto &a) { a.gemm_abi.buffers[0].bytes += 4; }) ||
      !artifactAttack("GEMM storage stride binding", [](auto &a) { a.gemm_abi.buffers[0].element_stride++; }) ||
      !artifactAttack("GEMM entry identity", [](auto &a) { a.gemm_abi.entry_point += "_foreign"; }) ||
      !artifactAttack("GEMM output role binding", [](auto &a) { a.gemm_abi.buffers[2].role = candidate::BufferRoleV1::Lhs; }) ||
      !artifactAttack("manifest cannot replace issuer", [](auto &a) { a.manifest += "execution-authority=true\n"; })) return 1;

  const auto binaryAttack = [&](llvm::StringRef name, spirv::Opcode opcode,
      std::function<bool(std::vector<std::uint32_t> &, std::size_t)> mutate) {
    auto changed = artifact;
    bool mutated = false;
    for (std::size_t i = 5; i < changed.gemm_words.size();) {
      const auto count = changed.gemm_words[i] >> 16;
      if (!count || count > changed.gemm_words.size() - i) return false;
      if ((changed.gemm_words[i] & 65535) == std::uint32_t(opcode) && mutate(changed.gemm_words, i)) {
        mutated = true;
        break;
      }
      i += count;
    }
    return require(mutated && !candidate::verifyStrictGemmArtifactV1(context, changed, error), name);
  };
  if (!binaryAttack("binary integer K constant mutation preserves opcode counts/modes",
      spirv::Opcode::OpConstant, [](auto &w, auto i) {
        if ((w[i] >> 16) != 4 || w[i + 3] != 3) return false;
        w[i + 3] = 2; return true;
      }) ||
      !binaryAttack("binary AccessChain mutation preserves opcode counts/modes",
      spirv::Opcode::OpAccessChain, [](auto &w, auto i) {
        if ((w[i] >> 16) != 6) return false;
        w[i + 5] = w[i + 4]; return true;
      }) ||
      !binaryAttack("binary binding mutation preserves opcode counts/modes",
      spirv::Opcode::OpDecorate, [](auto &w, auto i) {
        if ((w[i] >> 16) != 4 || w[i + 2] != std::uint32_t(spirv::Decoration::Binding)) return false;
        ++w[i + 3]; return true;
      }) ||
      !binaryAttack("binary stride mutation preserves opcode counts/modes",
      spirv::Opcode::OpDecorate, [](auto &w, auto i) {
        if ((w[i] >> 16) != 4 || w[i + 2] != std::uint32_t(spirv::Decoration::ArrayStride)) return false;
        w[i + 3] = 8; return true;
      }) ||
      !binaryAttack("binary branch edge mutation preserves opcode counts/modes",
      spirv::Opcode::OpBranchConditional, [](auto &w, auto i) {
        std::swap(w[i + 2], w[i + 3]); return true;
      }) ||
      !binaryAttack("binary strict float mode mutation",
      spirv::Opcode::OpExecutionMode, [](auto &w, auto i) {
        if (w[i + 2] != std::uint32_t(spirv::ExecutionMode::RoundingModeRTE)) return false;
        w[i + 2] = std::uint32_t(spirv::ExecutionMode::RoundingModeRTZ); return true;
      }) ||
      !binaryAttack("binary NoContraction mutation",
      spirv::Opcode::OpDecorate, [](auto &w, auto i) {
        if (w[i + 2] != std::uint32_t(spirv::Decoration::NoContraction)) return false;
        w[i + 2] = std::uint32_t(spirv::Decoration::RelaxedPrecision); return true;
      })) return 1;

  const auto outlinedAttack = [&](llvm::StringRef name, std::function<void(ModuleOp)> mutate) {
    auto changed = parseSourceString<ModuleOp>(artifact.outlined_ir, &context);
    if (!changed) return require(false, "parse outlined control");
    mutate(*changed);
    if (!require(succeeded(verify(*changed)), "mutated outlined module remains valid MLIR")) return false;
    return require(!candidate::verifyStrictGemmOutlinedV1(*changed, {2, 3, 4}, error), name);
  };
  if (!outlinedAttack("outlined reduction starts at one", [](auto module) {
        module.walk([](scf::ForOp op) { op.getLowerBoundMutable().assign(op.getStep()); });
      }) ||
      !outlinedAttack("outlined operand permutation", [](auto module) {
        module.walk([](arith::MulFOp op) { auto lhs = op.getLhs(); op.getLhsMutable().assign(op.getRhs()); op.getRhsMutable().assign(lhs); });
      }) ||
      !outlinedAttack("outlined contract permission", [](auto module) {
        module.walk([](arith::AddFOp op) { op.setFastmath(arith::FastMathFlags::contract); });
      }) ||
      !outlinedAttack("outlined untrusted argument permission", [&](auto module) {
        module.walk([&](gpu::GPUFuncOp op) { op.setArgAttr(0, "llvm.noalias", UnitAttr::get(&context)); });
      })) return 1;

  const auto shaderAttack = [&](llvm::StringRef name,
      std::function<void(spirv::ModuleOp)> mutate, bool fill = false) {
    auto changed = parseSourceString<spirv::ModuleOp>(
        fill ? artifact.fill_spirv_ir : artifact.gemm_spirv_ir, &context);
    if (!changed) return require(false, "parse shader control");
    mutate(*changed);
    if (!require(succeeded(verify(*changed)), "mutated shader remains valid MLIR")) return false;
    return require(!candidate::verifyStrictGemmShaderV1(*changed, {2, 3, 4}, fill, error), name);
  };
  if (!shaderAttack("final shader K-bound mutation keeps op counts/float modes", [](auto module) {
        module.walk([](spirv::SLessThanOp op) { op->setOperand(1, op->getOperand(0)); });
      }) ||
      !shaderAttack("final shader AccessChain mutation keeps op counts/float modes", [](auto module) {
        module.walk([](spirv::AccessChainOp op) { op->setOperand(2, op->getOperand(1)); });
      }) ||
      !shaderAttack("final shader branch edge mutation keeps op counts/float modes", [](auto module) {
        module.walk([](spirv::BranchConditionalOp op) { auto *first = op->getSuccessor(0); op->setSuccessor(op->getSuccessor(1), 0); op->setSuccessor(first, 1); });
      }) ||
      !shaderAttack("final shader binding mutation keeps op counts/float modes", [](auto module) {
        for (auto global : module.template getOps<spirv::GlobalVariableOp>())
          if (global.getBinding()) global.setBinding(0);
      }) ||
      !shaderAttack("final shader removed NoContraction", [](auto module) {
        module.walk([](spirv::FMulOp op) { op->removeAttr("no_contraction"); });
      }) ||
      !shaderAttack("final shader valid load alignment overclaim", [](auto module) {
        module.walk([](spirv::LoadOp op) {
          if (!op->getResult(0).getType().isF32()) return;
          op.setMemoryAccess(spirv::MemoryAccess::Aligned);
          op.setAlignment(8);
        });
      }) ||
      !shaderAttack("final shader valid store memory modifier", [](auto module) {
        module.walk([](spirv::StoreOp op) { op.setMemoryAccess(spirv::MemoryAccess::Volatile); });
      }) ||
      !shaderAttack("final shader nested index coefficient overflow is safely refused", [&](auto module) {
        spirv::AccessChainOp target;
        module.walk([&](spirv::AccessChainOp op) { if (!target) target = op; });
        OpBuilder builder(target);
        auto large = spirv::ConstantOp::create(builder, target.getLoc(),
            builder.getI32Type(), builder.getI32IntegerAttr(INT32_MAX));
        Value scaled = target->getOperand(2);
        for (unsigned i = 0; i != 6; ++i)
          scaled = spirv::IMulOp::create(builder, target.getLoc(), scaled, large);
        target->setOperand(2, scaled);
      }) ||
      !shaderAttack("final shader valid static array stride mutation", [](auto module) {
        const auto changedType = [](Type type) -> Type {
          auto pointer = dyn_cast<spirv::PointerType>(type);
          if (!pointer || pointer.getStorageClass() != spirv::StorageClass::StorageBuffer) return type;
          auto structure = dyn_cast<spirv::StructType>(pointer.getPointeeType());
          if (!structure || structure.getNumElements() != 1) return type;
          auto array = dyn_cast<spirv::ArrayType>(structure.getElementType(0));
          if (!array) return type;
          auto changed = spirv::ArrayType::get(array.getElementType(), array.getNumElements(), 8);
          return spirv::PointerType::get(spirv::StructType::get({changed}, {0}), pointer.getStorageClass());
        };
        module.walk([&](Operation *op) {
          if (auto global = dyn_cast<spirv::GlobalVariableOp>(op)) global.setType(changedType(global.getType()));
          for (auto result : op->getResults()) result.setType(changedType(result.getType()));
        });
      }) ||
      !shaderAttack("final shader missing denormal mode", [](auto module) {
        for (auto mode : llvm::make_early_inc_range(module.template getOps<spirv::ExecutionModeOp>()))
          if (mode.getExecutionMode() == spirv::ExecutionMode::DenormPreserve) mode.erase();
      }) ||
      !shaderAttack("final shader signed zero seed mutation", [&](auto module) {
        module.walk([&](spirv::ConstantOp op) {
          if (op->getResult(0).getType().isF32()) op.setValueAttr(FloatAttr::get(op->getResult(0).getType(), -0.0));
        });
      }, true)) return 1;
  llvm::outs() << "PASS " << checks << " closed static strict GEMM SPIR-V issuer checks; no execution authority\n";
}
