#include "ProgramHostInterface.h"
#include "ClosedHostInspection.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/SourceManager.h"
#include <algorithm>

namespace matcore::mdslc::frontend::detail {
namespace {
namespace host = closed_region_host;
using ParameterKind = ClosedRegionParameterBinding::Kind;
std::string symbolKey(llvm::StringRef name) {
  if (name.starts_with("\1")) name = name.drop_front();
  return name.split('@').first.str();
}
std::string symbol(const clang::FunctionDecl *function, clang::ASTContext &context) {
  auto mangler = std::unique_ptr<clang::MangleContext>(context.createMangleContext());
  std::string result;
  llvm::raw_string_ostream out(result);
  if (mangler->shouldMangleDeclName(function)) mangler->mangleName(function, out);
  else out << function->getNameAsString();
  return result;
}
// Upstream's out-of-line visitor keeps Clang's lazy/template allocation inside
// its owning prebuilt library. The declaration validation below stays ASan/
// UBSan-instrumented; no general sanitizer or declaration check is disabled.
class InterfaceVisitor : public clang::DynamicRecursiveASTVisitor {
public:
  InterfaceVisitor(clang::ASTContext &context, const ProgramRegionOwners &entries,
      std::size_t unit, ClosedRegionASTPolicy &policy, clang::FileManager &files,
      const ExperimentalRegionHeaders &headers, ProgramHostInterface &proof, std::string &error)
      : context_(context), entries_(entries), unit_(unit), policy_(policy),
        proof_(proof), error_(error), files_(files), headers_(headers) {}
  bool VisitFunctionDecl(clang::FunctionDecl *function) override {
    // Weak references may disappear into their target during Clang CodeGen:
    // effects on the alias can survive only at call sites while the target's
    // clean declaration remains unchanged. Inspect every alias edge before
    // name matching. Every chain ending at a protected symbol has an incoming
    // edge here, even if the earlier aliases or templates have unrelated names.
    auto protected_target = [&](llvm::StringRef target) {
      return !target.empty() && entries_.contains(symbolKey(target));
    };
    if (const auto *alias = function->getAttr<clang::AliasAttr>();
        alias && protected_target(alias->getAliasee()))
      return fail(function, "foreign region source alias targets an issued symbol without its canonical interface");
    if (const auto *weak = function->getAttr<clang::WeakRefAttr>();
        weak && protected_target(weak->getAliasee()))
      return fail(function, "foreign region source alias weakref targets an issued symbol without its canonical interface");
    if (const auto *resolver = function->getAttr<clang::IFuncAttr>();
        resolver && protected_target(resolver->getResolver()))
      return fail(function, "source ifunc resolver targets an issued region");
    if (function->isMain() && function->doesThisDeclarationHaveABody())
      ++proof_.main_definitions;
    const auto qualified = std::find_if(entries_.begin(), entries_.end(), [&](const auto &item) {
      return item.second.binding.qualified_name == function->getQualifiedNameAsString();
    });
    // Ownership is of an issued linker symbol, not every ordinary overload of
    // the same C++ name. Alternate source spellings still cannot borrow the
    // interface proof when an asm label or literal C name targets that symbol.
    const bool can_name_entry = qualified != entries_.end() ||
        function->hasAttr<clang::AsmLabelAttr>() || function->isExternC();
    const auto name = can_name_entry ? symbol(function, context_) : std::string{};
    const auto found = entries_.find(symbolKey(name));
    if (found == entries_.end()) {
      for (const auto *attribute : function->specific_attrs<clang::AnnotateAttr>())
        if (attribute->getAnnotation() == "matcore.experimental.region.v1" && function->doesThisDeclarationHaveABody())
          return fail(function, "program contains an unselected region definition");
      return true;
    }
    if (function->getQualifiedNameAsString() != found->second.binding.qualified_name)
      return fail(function, "foreign region source spelling differs from its issued symbol");
    if (policy_.owned_header.isInvalid() &&
        !authenticateExperimentalRegionHeaders(context_, files_, headers_, policy_, error_)) return false;
    const auto &entry = found->second;
    if (name != found->first) return fail(function, "foreign region symbol/signature differs from its issuer");
    proof_.declarations.push_back(name + (function->doesThisDeclarationHaveABody() ? ":definition" : ":declaration"));
    if (unit_ == entry.owner) return true; // Existing admission owns this complete redeclaration chain.
    // Competing definitions are diagnosed by the global LLVM ownership guard,
    // including weak/COMDAT/alias cases before ordinary linker selection.
    if (function->getDefinition()) {
      if (function->doesThisDeclarationHaveABody()) proof_.foreign_definitions.push_back(name);
      return true;
    }
    if (function->hasAttrs()) return fail(function, "foreign region declaration carries unadmitted function attributes");
    const auto *prototype = function->getType()->getAs<clang::FunctionProtoType>();
    if (!prototype || prototype->getCallConv() != clang::CC_C || function->isVariadic() ||
        prototype->getExceptionSpecType() != clang::EST_BasicNoexcept ||
        !record(function->getReturnType(), "matcore::mdsl::Result") ||
        function->getNumParams() != entry.binding.parameters.size())
      return fail(function, "foreign region declaration lacks its canonical source ABI");
    for (unsigned i = 0; i < function->getNumParams(); ++i) {
      const auto *parameter = function->getParamDecl(i);
      const auto type = parameter->getType();
      const bool valid = entry.binding.parameters[i].kind == ParameterKind::Storage
        ? record(type, "matcore::mdsl::Storage")
        : !type.isNull() && context_.hasSameType(type, context_.UnsignedLongLongTy);
      if (!valid || parameter->hasAttrs() || parameter->hasDefaultArg())
        return fail(function, "foreign region parameter lacks its canonical source ABI");
    }
    return true;
  }
private:
  bool record(clang::QualType type, llvm::StringRef name) {
    if (type.isNull() || type.hasLocalQualifiers() || type->isReferenceType()) return false;
    const auto *declaration = type->getAsCXXRecordDecl();
    // CXXRecordDecl's inline definition query also instantiates lazy allocator
    // templates. TagDecl's out-of-line query resolves the same actual body in
    // the owning Clang library; never weaken this into a spelling/name check.
    const auto *record = declaration ? llvm::dyn_cast_or_null<clang::CXXRecordDecl>(
      static_cast<const clang::TagDecl *>(declaration)->getDefinition()) : nullptr;
    if (!record) return false;
    const auto location = record->getLocation();
    return record->getQualifiedNameAsString() == name && !location.isMacroID() &&
      context_.getSourceManager().getFileID(location) == policy_.owned_header;
  }
  bool fail(const clang::FunctionDecl *function, const std::string &message) {
    auto &sm = context_.getSourceManager();
    const auto location = function->getLocation();
    error_ = sm.getFilename(location).str() + ":" +
      std::to_string(sm.getSpellingLineNumber(location)) + ": " + message;
    return false;
  }
  clang::ASTContext &context_;
  const ProgramRegionOwners &entries_;
  std::size_t unit_;
  ClosedRegionASTPolicy &policy_;
  ProgramHostInterface &proof_;
  std::string &error_;
  clang::FileManager &files_;
  const ExperimentalRegionHeaders &headers_;
};

bool scan(const std::vector<std::string> &arguments, const std::string &cwd,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> filesystem,
    const std::string &source, const std::string &input,
    const ExperimentalRegionHeaders &headers, const ProgramRegionOwners &entries,
    std::size_t index, ProgramHostInterface &proof, std::string &error) {
  return inspectClosedHost(arguments, cwd, filesystem, source, input,
    [&](clang::ASTContext &context, const clang::Decl *anchor, clang::FileManager &files,
        ClosedRegionASTPolicy &policy, std::string &diagnostic) {
      InterfaceVisitor visitor(context, entries, index, policy, files, headers, proof, diagnostic);
      // Use the out-of-line owner query: ASTContext's inline getter can emit
      // incompatible weak lazy-allocator instantiations with prebuilt Clang.
      return !anchor || visitor.TraverseDecl(const_cast<clang::TranslationUnitDecl *>(
          anchor->getTranslationUnitDecl()));
    }, proof.preprocessing, error);
}
} // namespace

ProgramHostInterfaceResult inspectProgramHost(
    const Options &options, const std::string &working_directory,
    const ExperimentalRegionHeaders &headers, const ProgramRegionOwners &entries, std::size_t index,
    std::shared_ptr<const closed_region_host::HostInputSnapshot> snapshot) {
  ProgramHostInterfaceResult result;
  std::optional<ProgramHostInterface> initial;
  if (!snapshot) {
    auto capture = host::prepareHostInputs(options, working_directory, {}, result.error,
        host::HostInputPrelude::None);
    if (!capture) return result;
    initial.emplace();
    if (!scan(capture->arguments(), capture->workingDirectory(), capture->fileSystem(),
        capture->sourceSnapshot(), capture->inputPath(), headers, entries, index, *initial, result.error))
      return result;
    snapshot = capture->freeze(result.error);
    if (!snapshot) return result;
  }
  auto replay = snapshot->replay();
  if (!replay.ok(result.error)) return result;
  if (!scan(snapshot->arguments(), snapshot->workingDirectory(), replay.filesystem,
      snapshot->sourceSnapshot(), snapshot->inputPath(), headers, entries, index, result.interface, result.error))
    return result;
  if (!replay.ok(result.error)) return result;
  if (initial && *initial != result.interface) {
    result.error = "frozen host interface replay differs";
    return result;
  }
  if (!snapshot->unchanged(result.error)) return result;
  result.snapshot = std::move(snapshot);
  return result;
}
} // namespace matcore::mdslc::frontend::detail
