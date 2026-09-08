#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBufferRef.h"
#include <cstddef>
#include <string>

namespace matcore::mdslc::codegen {

enum class SymbolArtifactOwner { MatcoreRuntime, ExternalProvider, PrivateCandidates };

// Compiler-owned immutable bytes, already authenticated against the driver's
// built-in artifact identity. This utility does not open source-selected paths,
// establish a library hash, or issue source/execution authority. The backing
// buffers must outlive the call. The driver rechecks its link-input closure.
struct TrustedSymbolArtifact {
  SymbolArtifactOwner owner;
  llvm::MemoryBufferRef bytes;
};

struct ArtifactSymbolOwnershipReport {
  std::size_t runtime_exports = 0;
  std::size_t provider_exports = 0;
  std::size_t candidate_exports = 0;
};

// Inspect the ORIGINAL freshly Clang-produced host module before introducing
// compiler-generated helper definitions. Exactly one trusted runtime ELF DSO is
// required; an isolated private candidate DSO and explicitly selected provider
// ELF DSOs may follow. The execution compiler requires its candidate DSO too.
// Every defined nonlocal dynamic symbol in every trusted DSO is reserved,
// uniformly, including weak/data/alias/IFUNC symbols. The canonical runtime
// already hides its implementation/STL symbols. A future export leak must be
// corrected at that DSO's visibility boundary, not guessed from demangled names.
//
// Nonempty module and instruction inline assembly is outside this bounded link
// contract because it can define symbols absent from LLVM's global-value graph.
// Empty instruction barriers remain allowed. The exact libstdc++ iostream
// declaration `.globl _ZSt21ios_base_library_initv` is also allowed only when
// that symbol is not owned by a trusted artifact; no general assembler grammar
// or header-path exemption is admitted.
//
// Compiler helper definitions are separately internalized, while the candidate
// DSO localizes implementation definitions. Archive whole-linking is not an
// equivalent ownership proof. Pinning does not authenticate future loading.
// Deployment requires stable trusted runtime/provider/standard-library loading,
// conforming allocator hooks and no foreign interposition. No sandbox or
// executable self-authentication claim follows from a successful check.
bool verifyHostArtifactSymbolOwnership(
    const llvm::Module &original_host,
    llvm::ArrayRef<TrustedSymbolArtifact> artifacts,
    ArtifactSymbolOwnershipReport &report, std::string &error);

} // namespace matcore::mdslc::codegen
