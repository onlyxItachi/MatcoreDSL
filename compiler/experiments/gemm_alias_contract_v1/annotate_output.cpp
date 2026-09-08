// Research instrument, not a production issuer or importer. The caller of this
// exact specimen guarantees that C's data is disjoint from A/B for the call.
// Annotate the lowered aligned-data pointer, not the C-wrapper descriptor.
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIRFile(argv[1], diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 2;
  }
  auto *function = module->getFunction("research_gemm");
  if (!function || function->isDeclaration() || function->isVarArg() ||
      function->arg_size() != 21 || !function->getReturnType()->isVoidTy())
    return 3;
  for (unsigned index = 0; index != 21; ++index) {
    auto *type = function->getArg(index)->getType();
    if ((index % 7 < 2 ? !type->isPointerTy() : !type->isIntegerTy(64)) ||
        function->hasParamAttribute(index, llvm::Attribute::NoAlias))
      return 3;
  }
  // Each pinned unpacked rank-2 descriptor is
  // allocated, aligned, offset, sizes[2], strides[2]. Only aligned C is used
  // to access C's data. Neither A nor B gets a noalias attribute.
  function->addParamAttr(15, llvm::Attribute::NoAlias);
  if (llvm::verifyModule(*module, &llvm::errs())) return 4;
  std::error_code error;
  llvm::raw_fd_ostream output(argv[2], error);
  if (error) return 5;
  module->print(output, nullptr);
  return 0;
}
