#ifndef MATCORE_MDSLC_AUTHENTICATED_HOST_THUNK_H
#define MATCORE_MDSLC_AUTHENTICATED_HOST_THUNK_H

#include "llvm/IR/Module.h"
#include <memory>
#include <string>
#include <vector>

namespace matcore::mdslc::codegen {

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
// LLVM opaque-pointer layout equality cannot authenticate C++ record meaning.
// Input modules remain unchanged on success/failure; the isolated result keeps
// the original function symbol, linkage, visibility and all ordinary host uses.
HostThunkResult linkAuthenticatedHostThunk(const llvm::Module &host,
                                           const llvm::Module &helper,
                                           const HostThunkRequest &request);

} // namespace matcore::mdslc::codegen
#endif
