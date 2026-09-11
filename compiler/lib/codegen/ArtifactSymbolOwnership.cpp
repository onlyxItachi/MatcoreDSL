#include "ArtifactSymbolOwnership.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <map>

namespace matcore::mdslc::codegen {
namespace {

std::string symbolKey(llvm::StringRef name) {
  // Clang's explicit assembler-label spelling may suppress target mangling with
  // a leading '\1'. ELF symbol versions still own the underlying symbol name.
  if (name.starts_with("\1")) name = name.drop_front();
  return name.split('@').first.str();
}

using ReservedSymbols = std::map<std::string, std::string>;
constexpr const char *thread_start =
    "_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE";
constexpr const char *thread_join = "_ZNSt6thread4joinEv";
// Qualified libstdc++ C++20 std::thread -> POSIX create/join closure. Unlike
// reserving every libstdc++ export, this does not reject normal inline STL
// instantiations. Unknown worker ABIs require an explicit new qualification.
constexpr const char *worker_symbols[] = {
    thread_start, thread_join, "pthread_create", "pthread_join",
    "_ZNSt6thread6_StateD0Ev", "_ZNSt6thread6_StateD1Ev", "_ZNSt6thread6_StateD2Ev",
    "_ZTINSt6thread6_StateE", "_ZTVNSt6thread6_StateE", "_ZTSNSt6thread6_StateE",
    "_ZSt9terminatev", "_ZSt20__throw_system_errori"};

bool collect(const TrustedSymbolArtifact &artifact, ReservedSymbols &symbols,
             unsigned expected_machine, ArtifactSymbolOwnershipReport &report,
             std::string &error) {
  const auto label = artifact.bytes.getBufferIdentifier().str();
  auto parsed = llvm::object::ObjectFile::createObjectFile(artifact.bytes);
  if (!parsed) {
    error = "cannot inspect trusted symbol artifact " + label + ": " +
            llvm::toString(parsed.takeError());
    return false;
  }
  const auto *elf = llvm::dyn_cast<llvm::object::ELFObjectFileBase>(parsed->get());
  if (!elf || elf->getEType() != llvm::ELF::ET_DYN ||
      elf->getEMachine() != expected_machine ||
      elf->getBytesInAddress() != 8 || !elf->isLittleEndian()) {
    error = "trusted symbol artifact is not a little-endian Linux ELF64 DSO matching the authenticated host target: " + label;
    return false;
  }
  if (artifact.owner != SymbolArtifactOwner::MatcoreRuntime &&
      artifact.owner != SymbolArtifactOwner::ExternalProvider &&
      artifact.owner != SymbolArtifactOwner::PrivateCandidates) {
    error = "unknown trusted symbol artifact owner";
    return false;
  }
  if (artifact.requires_worker_threads && artifact.owner != SymbolArtifactOwner::PrivateCandidates) {
    error = "only the private candidate DSO may require the worker-thread contract";
    return false;
  }
  bool imports_start = false, imports_join = false;
  std::size_t count = 0;
  for (const auto symbol : elf->getDynamicSymbolIterators()) {
    auto flags = symbol.getFlags();
    if (!flags) {
      error = "invalid trusted artifact symbol flags: " + llvm::toString(flags.takeError());
      return false;
    }
    auto name = symbol.getName();
    if (!name) {
      error = "invalid trusted artifact symbol name: " + llvm::toString(name.takeError());
      return false;
    }
    const auto key = symbolKey(*name);
    if (key.empty()) continue;
    if (*flags & llvm::object::SymbolRef::SF_Undefined) {
      imports_start |= key == thread_start;
      imports_join |= key == thread_join;
      continue;
    }
    if (symbol.getBinding() == llvm::ELF::STB_LOCAL) continue;
    symbols.try_emplace(key, label);
    ++count;
  }
  if (count == 0) {
    error = "trusted symbol artifact has no exports for its declared owner: " + label;
    return false;
  }
  if (artifact.requires_worker_threads) {
    if (!imports_start || !imports_join) {
      error = "private candidate DSO does not match the qualified std::thread worker ABI";
      return false;
    }
    for (const auto *symbol : worker_symbols)
      symbols.try_emplace(symbol, "trusted worker thread runtime");
  }
  if (artifact.owner == SymbolArtifactOwner::MatcoreRuntime) report.runtime_exports += count;
  else if (artifact.owner == SymbolArtifactOwner::ExternalProvider) report.provider_exports += count;
  else report.candidate_exports += count;
  return true;
}

} // namespace

bool verifyHostArtifactSymbolOwnership(
    const llvm::Module &host, llvm::ArrayRef<TrustedSymbolArtifact> artifacts,
    ArtifactSymbolOwnershipReport &report, std::string &error) {
  report = {};
  error.clear();
  const llvm::Triple target(host.getTargetTriple());
  const auto machine = target.getArch() == llvm::Triple::x86_64
                           ? llvm::ELF::EM_X86_64
                       : target.getArch() == llvm::Triple::aarch64
                           ? llvm::ELF::EM_AARCH64 : llvm::ELF::EM_NONE;
  if (machine == llvm::ELF::EM_NONE || !target.isOSLinux() ||
      host.getDataLayout().getPointerSizeInBits() != 64 ||
      !host.getDataLayout().isLittleEndian()) {
    error = "artifact symbol ownership requires a little-endian Linux x86-64 or AArch64 host contract";
    return false;
  }
  for (const auto &function : host)
    for (const auto &block : function)
      for (const auto &instruction : block)
        if (const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
          if (const auto *assembly = llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand()))
            if (!assembly->getAsmString().empty()) {
              error = "host instruction assembly can define symbols outside the authenticated ownership graph";
              return false;
            }

  std::size_t runtimes = 0, candidates = 0;
  for (const auto &artifact : artifacts) {
    runtimes += artifact.owner == SymbolArtifactOwner::MatcoreRuntime;
    candidates += artifact.owner == SymbolArtifactOwner::PrivateCandidates;
  }
  if (runtimes != 1 || candidates > 1) {
    error = "artifact ownership requires exactly one Matcore runtime and at most one private candidate DSO";
    return false;
  }

  ReservedSymbols symbols;
  ArtifactSymbolOwnershipReport candidate;
  for (const auto &artifact : artifacts)
    if (!collect(artifact, symbols, machine, candidate, error)) return false;

  // GNU libstdc++ <iostream> emits this declaration to retain its initialization
  // dependency. It defines no symbol. Do not interpret arbitrary assembly or
  // trust a source/header location: only these exact declaration bytes qualify.
  // Even .globl may promote an existing local assembler-label definition, so
  // the declared name must not belong to either authenticated artifact owner.
  const auto module_asm = llvm::StringRef(host.getModuleInlineAsm()).trim();
  if (!module_asm.empty() &&
      (module_asm != ".globl _ZSt21ios_base_library_initv" ||
       symbols.contains("_ZSt21ios_base_library_initv"))) {
    error = "host module assembly can define symbols outside the authenticated ownership graph";
    return false;
  }

  for (const auto &value : host.global_values()) {
    if (value.isDeclaration() || value.hasLocalLinkage()) continue;
    const auto symbol = symbolKey(value.getName());
    const auto found = symbols.find(symbol);
    if (found != symbols.end()) {
      error = "original host defines symbol owned by trusted artifact: " + symbol +
              " (" + found->second + ")";
      return false;
    }
  }
  report = candidate;
  return true;
}

} // namespace matcore::mdslc::codegen
