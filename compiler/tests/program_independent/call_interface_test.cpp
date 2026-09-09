#include "AuthenticatedHostThunk.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace cg = matcore::mdslc::codegen;
namespace {
constexpr auto symbol = "_Z5firstN7matcore4mdsl7StorageES1_S1_yyy";
struct Calls { llvm::CallBase *clean = nullptr, *effect_altered = nullptr; };
Calls calls(llvm::Module &module) {
  Calls found;
  for (auto &function : module)
    for (auto &block : function)
      for (auto &instruction : block)
        if (auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
          if (auto *callee = call->getCalledFunction(); callee && callee->getName() == symbol) {
            auto *&slot = call->hasFnAttr(llvm::Attribute::Memory) ? found.effect_altered : found.clean;
            if (slot) throw std::runtime_error("fixture has duplicate call classification");
            slot = call;
          }
  if (!found.clean || !found.effect_altered)
    throw std::runtime_error("real Clang weakref fixture lacks clean/altered calls");
  return found;
}
std::unique_ptr<llvm::Module> parse(const llvm::MemoryBuffer &source, llvm::LLVMContext &context) {
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseAssembly(source.getMemBufferRef(), diagnostic, context);
  if (!module || llvm::verifyModule(*module, &llvm::errs()))
    throw std::runtime_error("fixture must be valid real Clang LLVM");
  return module;
}
llvm::StructType *alterRecord(llvm::Type *type, llvm::LLVMContext &context) {
  auto *record = llvm::cast<llvm::StructType>(type);
  std::vector<llvm::Type *> fields(record->elements().begin(), record->elements().end());
  fields.push_back(llvm::Type::getInt64Ty(context));
  return llvm::StructType::get(context, fields, record->isPacked());
}
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  unsigned checks = 0, failures = 0;
  auto check = [&](bool good, const std::string &name) {
    ++checks;
    if (!good) { ++failures; std::cerr << "FAIL " << name << '\n'; }
  };
  try {
    auto bytes = llvm::MemoryBuffer::getFile(argv[1]);
    if (!bytes) return 2;
    llvm::LLVMContext reference_context;
    auto reference = parse(**bytes, reference_context);
    const auto original = calls(*reference);
    check(!original.clean->getCalledFunction()->hasFnAttribute(llvm::Attribute::Memory),
          "target function remains clean despite alias call promise");
    check(cg::sameAuthenticatedHostCallInterface(*original.clean, *original.clean), "clean reflexivity");
    check(!cg::sameAuthenticatedHostCallInterface(*original.clean, *original.effect_altered),
          "actual call-only weakref effect rejected");
    for (unsigned mutation = 0; mutation != 11; ++mutation) {
      // Like the real program compiler, both modules belong to one owned
      // context. Attribute identity from unrelated contexts is not this API's
      // input contract; identified record types may still differ here.
      auto &context = reference_context;
      auto module = parse(**bytes, context);
      auto *call = calls(*module).clean;
      check(cg::sameAuthenticatedHostCallInterface(*original.clean, *call),
            "separately parsed sret/byval calls remain equivalent");
      switch (mutation) {
      case 0: call->setCallingConv(llvm::CallingConv::Fast); break;
      case 1: call->addFnAttr(llvm::Attribute::NoFree); break;
      case 2: call->setMemoryEffects(llvm::MemoryEffects::argMemOnly()); break;
      case 3: call->removeFnAttr(llvm::Attribute::NoUnwind); break;
      case 4: call->addParamAttr(0, llvm::Attribute::getWithAlignment(context, llvm::Align(16))); break;
      case 5: call->addParamAttr(1, llvm::Attribute::getWithAlignment(context, llvm::Align(16))); break;
      case 6: call->addParamAttr(1, llvm::Attribute::getWithByValType(context,
        alterRecord(call->getParamByValType(1), context))); break;
      case 7: call->addParamAttr(0, llvm::Attribute::getWithStructRetType(context,
        alterRecord(call->getParamStructRetType(0), context))); break;
      case 8: call->addParamAttr(1, llvm::Attribute::NoAlias); break;
      case 9: call->removeParamAttr(0, llvm::Attribute::DeadOnUnwind); break;
      case 10:
        call->setCalledOperand(llvm::ConstantExpr::getAddrSpaceCast(call->getCalledFunction(),
            llvm::PointerType::get(context, 1)));
        break;
      }
      check(!llvm::verifyModule(*module, &llvm::errs()), "mutation remains valid LLVM");
      check(!cg::sameAuthenticatedHostCallInterface(*original.clean, *call),
            "changed call contract rejected (mutation " + std::to_string(mutation) + ")");
    }
  } catch (const std::exception &error) {
    std::cerr << "FAIL exception: " << error.what() << '\n';
    return 1;
  }
  std::cout << "Call-interface checks " << checks << ", failures " << failures << '\n';
  return failures ? 1 : 0;
}
