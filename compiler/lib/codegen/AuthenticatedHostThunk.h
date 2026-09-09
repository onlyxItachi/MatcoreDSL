#ifndef MATCORE_MDSLC_AUTHENTICATED_HOST_THUNK_H
#define MATCORE_MDSLC_AUTHENTICATED_HOST_THUNK_H

#include "llvm/IR/Module.h"
#include <memory>
#include <string>
#include <vector>
namespace llvm { class CallBase; }

namespace matcore::mdslc::codegen {

// Compiler-private reuse of the thunk's existing structural ABI comparator.
// Equality of LLVM types is not source admission, an effect summary, or
// permission to import external LLVM. Declaration witnesses are compared to
// declarations, because Clang attaches additional facts to definitions.
bool sameAuthenticatedHostABI(const llvm::Function &, const llvm::Function &,
                              bool remapped = false);
bool sameAuthenticatedHostFunctionAttributes(const llvm::Function &,
                                            const llvm::Function &);
// Exact callsite ABI/effect comparison against a compiler-issued call witness.
// Reuses the same type/attribute algorithms; does not admit caller LLVM.
bool sameAuthenticatedHostCallInterface(const llvm::CallBase &, const llvm::CallBase &);

struct HostThunkRequest {
  std::string host_symbol;
  std::string helper_symbol;
  // Exact compiler-owned Value helper symbols selected by the frontend seal.
  // Never a prefix, namespace sweep, or instruction to delete Shape helpers.
  std::vector<std::string> retired_value_functions;
};

struct HostThunkResult {
  // LLVM Linker can rename context-owned types even through a cloned Module.
  // Own an isolated context; destroy its Module before its context, including
  // during move assignment of a previously populated result.
  std::unique_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
  std::string error;
  HostThunkResult()=default;
  HostThunkResult(HostThunkResult &&)=default;
  HostThunkResult &operator=(HostThunkResult &&other) noexcept {
    if(this!=&other) {
      module.reset(); context=std::move(other.context);
      module=std::move(other.module); error=std::move(other.error);
    }
    return *this;
  }
  explicit operator bool() const { return static_cast<bool>(module); }
};

// Internal transformation, NOT an admission or execution-authority issuer.
// Both modules must be freshly produced by the trusted compiler, before LLVM
// optimizations can consume facts about the old region body, in one context.
// The caller authenticates source, ABI record semantics and the exact symbols.
// All non-entry helper definitions are compiler-private and internalized before
// linking; ordinary host weak/strong symbols never select their implementation.
// This requires a compiler-issued implementation with no user-visible helper
// definition identity contract. External runtime declarations stay external.
// LLVM opaque-pointer layout equality cannot authenticate C++ record meaning.
// Input modules remain unchanged on success/failure; the isolated result keeps
// the original function symbol, linkage, visibility and all ordinary host uses.
HostThunkResult linkAuthenticatedHostThunk(const llvm::Module &host,
                                           const llvm::Module &helper,
                                           const HostThunkRequest &request);

} // namespace matcore::mdslc::codegen
#endif
