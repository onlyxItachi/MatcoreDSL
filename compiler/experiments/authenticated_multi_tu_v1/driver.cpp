// Research only. Reuse internal implementations directly instead of copying
// the ABI algorithm, preprocessor transcript, or artifact publication checks.
// This is intentionally not a proposed installed API or production build target.
#include "../../lib/frontend/ClosedRegionHostAdmission.cpp"
#include "../../lib/codegen/AuthenticatedHostThunk.cpp"
#define main single_tu_driver_reference_main
#include "../../tools/mdslc-region/main.cpp"
#undef main
#include "../../lib/codegen/FrozenHostCodegen.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/Frontend/FrontendActions.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

namespace experiment {
namespace fe = matcore::mdslc::frontend;
namespace cg = matcore::mdslc::codegen;
namespace host = fe::closed_region_host;
using ParameterKind = fe::ClosedRegionParameterBinding::Kind;

struct Unit {
  fs::path path;
  std::string region;
  std::optional<fe::AuthenticatedClosedRegionEvidence> evidence;
  std::optional<cg::ExperimentalRegionEmission> emission;
  std::shared_ptr<const host::HostInputSnapshot> snapshot;
  std::optional<cg::ExperimentalLLVMCompilation> compilation;
  std::unique_ptr<llvm::Module> raw;
  std::vector<std::string> foreign_definitions;
};
struct Entry {
  std::size_t owner;
  fe::ClosedRegionEntryBinding binding;
};
using Entries = std::map<std::string, Entry>;

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

struct InterfaceProof {
  unsigned main_definitions = 0;
  std::vector<std::string> declarations;
  std::vector<std::string> foreign_definitions;
  std::string preprocessing;
  bool operator==(const InterfaceProof &) const = default;
};
// Upstream's out-of-line visitor keeps Clang's lazy/template allocation inside
// its owning prebuilt library. The declaration validation below stays ASan/
// UBSan-instrumented; no general sanitizer or declaration check is disabled.
class InterfaceVisitor : public clang::DynamicRecursiveASTVisitor {
public:
  InterfaceVisitor(clang::ASTContext &context, const Entries &entries,
      std::size_t unit, const fe::detail::ClosedRegionASTPolicy &policy,
      InterfaceProof &proof, std::string &error)
      : context_(context), entries_(entries), unit_(unit), policy_(policy),
        proof_(proof), error_(error) {}
  bool VisitFunctionDecl(clang::FunctionDecl *function) override {
    if (function->isMain() && function->doesThisDeclarationHaveABody())
      ++proof_.main_definitions;
    const auto found = std::find_if(entries_.begin(), entries_.end(), [&](const auto &item) {
      return item.second.binding.qualified_name == function->getQualifiedNameAsString();
    });
    if (found == entries_.end()) {
      // An asm label or literal extern-C spelling can name the same ELF
      // symbol without naming the source interface. Do not let it borrow the
      // proof of an unrelated clean declaration already present in this TU.
      if ((function->hasAttr<clang::AsmLabelAttr>() || function->isExternC()) &&
          entries_.contains(symbolKey(symbol(function, context_))))
        return fail(function, "foreign region source spelling differs from its issued symbol");
      return true;
    }
    const auto &entry = found->second;
    const auto name = symbol(function, context_);
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
  const Entries &entries_;
  std::size_t unit_;
  const fe::detail::ClosedRegionASTPolicy &policy_;
  InterfaceProof &proof_;
  std::string &error_;
};

class InterfaceConsumer : public clang::ASTConsumer {
public:
  InterfaceConsumer(clang::CompilerInstance &compiler, fe::ParseState &state,
      const Entries &entries, std::size_t index, InterfaceProof &proof)
      : compiler_(compiler), state_(state), entries_(entries), index_(index), proof_(proof) {}
  bool HandleTopLevelDecl(clang::DeclGroupRef group) override {
    if (!anchor_ && !group.isNull() && group.begin() != group.end()) anchor_ = *group.begin();
    return true;
  }
  void HandleTranslationUnit(clang::ASTContext &context) override {
    state_.visited = true;
    if (compiler_.getDiagnostics().hasErrorOccurred()) return;
    auto main = compiler_.getFileManager().getFileRef(state_.program.source_identity);
    if (!main) { llvm::consumeError(main.takeError()); state_.error = "host main identity unavailable"; return; }
    auto &sm = context.getSourceManager();
    bool invalid = false;
    if (sm.translateFile(&main->getFileEntry()) != sm.getMainFileID() ||
        fe::detail::closedRegionDigest(sm.getBufferData(sm.getMainFileID(), &invalid).str()) != state_.program.source_sha256 || invalid) {
      state_.error = "host main FileID/source capture differs"; return;
    }
    if (state_.volatile_preprocessing || state_.preprocessing_budget_exceeded) {
      state_.error = "host preprocessing exceeds existing frozen-input contract"; return;
    }
    if (!fe::bindPreprocessing(context, state_) ||
        !fe::detail::authenticateExperimentalRegionHeaders(context, compiler_.getFileManager(),
          *state_.public_headers, state_.policy, state_.error)) return;
    InterfaceVisitor visitor(context, entries_, index_, state_.policy, proof_, state_.error);
    // Reuse existing admission's out-of-line TU query. ASTContext's inline
    // getter can instantiate an ASan-poisoning lazy allocator protocol into
    // the supported unsanitized Clang/LLVM library through weak definitions.
    if (!anchor_ || !visitor.TraverseDecl(anchor_->getTranslationUnitDecl())) return;
    proof_.preprocessing = state_.program.compiler_identity;
    state_.admitted = true;
  }
private:
  clang::CompilerInstance &compiler_;
  fe::ParseState &state_;
  const Entries &entries_;
  std::size_t index_;
  InterfaceProof &proof_;
  clang::Decl *anchor_ = nullptr;
};
class InterfaceAction : public clang::ASTFrontendAction {
public:
  InterfaceAction(fe::ParseState &state, const Entries &entries, std::size_t index, InterfaceProof &proof)
      : state_(state), entries_(entries), index_(index), proof_(proof) {}
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &compiler, llvm::StringRef) override {
    compiler.getPreprocessor().addPPCallbacks(std::make_unique<fe::Preprocessing>(
      compiler.getSourceManager(), compiler.getLangOpts(), state_));
    return std::make_unique<InterfaceConsumer>(compiler, state_, entries_, index_, proof_);
  }
private:
  fe::ParseState &state_;
  const Entries &entries_;
  std::size_t index_;
  InterfaceProof &proof_;
};

InterfaceProof scan(const std::vector<std::string> &arguments, const std::string &cwd,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> filesystem,
    const std::string &source, const std::string &input,
    const fe::ExperimentalRegionHeaders &headers, const Entries &entries, std::size_t index) {
  fe::ParseState state;
  fe::initialize(state, source, input, "", &headers);
  InterfaceProof proof;
  clang::FileSystemOptions file_options;
  file_options.WorkingDir = cwd;
  auto files = llvm::makeIntrusiveRefCnt<clang::FileManager>(file_options, filesystem);
  fe::Diagnostics diagnostics;
  clang::tooling::ToolInvocation invocation(arguments,
      std::make_unique<InterfaceAction>(state, entries, index, proof), files.get());
  invocation.setDiagnosticConsumer(&diagnostics);
  if (!invocation.run() || diagnostics.failed || !state.visited || !state.admitted)
    reject("INTERFACE: " + state.error + " " + diagnostics.text);
  return proof;
}

std::unique_ptr<llvm::Module> parseIr(const std::string &ir, llvm::LLVMContext &context) {
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseAssemblyString(ir, diagnostic, context);
  if (!module) reject("frozen Clang LLVM could not be parsed");
  return module;
}

int execute(int argc, char **argv) {
  // All retained original modules must be destroyed before their LLVMContext,
  // including when a pre-link rejection unwinds this research invocation.
  llvm::LLVMContext context;
  std::vector<Unit> units;
  fs::path output;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    auto next = [&]() { if (++i >= argc) reject("missing argument for " + option); return std::string(argv[i]); };
    if (option == "-o") { if (!output.empty()) reject("duplicate output"); output = next(); }
    else if (option == "--host") units.push_back({fs::absolute(next()), ""});
    else if (option == "--region") { auto path = fs::absolute(next()); units.push_back({path, next()}); }
    else reject("research driver accepts only --host SOURCE, --region SOURCE NAME, -o NEW_OUTPUT");
  }
  if (units.size() < 2 || units.size() > 8 || output.empty()) reject("bounded experiment requires 2..8 source TUs and new output");
  output = fs::absolute(output);
  std::string error;
  if (!support::prospective_output_path_supported_v1(output, error) ||
      !fs::is_directory(output.parent_path()) || fs::symlink_status(output).type() != fs::file_type::not_found)
    reject("output must be a supported new path: " + error);
  std::set<std::string> paths;
  for (const auto &unit : units) if (!paths.insert(unit.path.string()).second) reject("duplicate source TU");
  Staging staging(output.parent_path());
  const auto ir_path = staging.path / "program.ll", binary_path = staging.path / "program";
  write(ir_path, ""); write(binary_path, "");
  std::vector<fs::path> helper_directories;
  for (std::size_t i = 0; i < units.size(); ++i) {
    auto directory = staging.path / ("helper-" + std::to_string(i));
    fs::create_directory(directory); helper_directories.push_back(directory);
  }
  const auto witness_directory = staging.path / "interface-witness";
  fs::create_directory(witness_directory);
  const auto cwd = fs::current_path().string();
  const Layout installed{REGION_BUILD_INCLUDE, REGION_BUILD_PRIVATE_HEADER,
    REGION_BUILD_CANDIDATES, REGION_BUILD_RUNTIME, true};
  const auto clang = Artifact::capture(REGION_CLANG, REGION_CLANG_SHA);
  const auto linker = Artifact::capture(REGION_LINKER, REGION_LINKER_SHA);
  const auto candidates = Artifact::capture(installed.candidates, REGION_CANDIDATES_SHA);
  const auto runtime = Artifact::capture(installed.runtime, REGION_RUNTIME_SHA);
  const auto public_header = Artifact::capture(installed.include / "matcore/region.h");
  const auto storage_header = Artifact::capture(installed.include / "matcore/detail/region_storage.h");
  const auto private_header = Artifact::capture(installed.header);
  std::optional<Artifact> provider;
  if (std::strlen(REGION_PROVIDER_PATH)) provider = Artifact::capture(REGION_PROVIDER_PATH, REGION_PROVIDER_SHA);
  std::vector<cg::TrustedSymbolArtifact> artifacts{
    {cg::SymbolArtifactOwner::MatcoreRuntime, llvm::MemoryBufferRef(runtime.bytes, "pinned canonical runtime")},
    {cg::SymbolArtifactOwner::PrivateCandidates, llvm::MemoryBufferRef(candidates.bytes, "pinned private candidates")}};
  if (provider) artifacts.push_back({cg::SymbolArtifactOwner::ExternalProvider,
    llvm::MemoryBufferRef(provider->bytes, "pinned provider")});
  const fe::ExperimentalRegionHeaders headers{public_header.path.string(), storage_header.path.string()};
  auto optionsFor = [&](const Unit &unit) {
    fe::Options options;
    options.input_path = unit.path.string(); options.clang_path = REGION_CLANG;
    options.clang_resource_directory = REGION_RESOURCE_DIR;
    options.compiler_arguments = {"-I" + installed.include.string()};
    if (REGION_SANITIZED) options.compiler_arguments.push_back("-fsanitize=address,undefined");
    return options;
  };
  Entries entries;
  std::map<std::string, std::set<std::size_t>> retired;
  std::set<std::string> helpers;
  for (std::size_t i = 0; i < units.size(); ++i) {
    auto &unit = units[i];
    if (unit.region.empty()) continue;
    auto admitted = fe::admitExperimentalRegionHost(optionsFor(unit), cwd, headers, unit.region);
    if (!admitted) reject("ADMISSION: " + admitted.error);
    unit.evidence = std::move(*admitted.evidence);
    const auto &binding = *unit.evidence->entryBinding();
    if (!entries.emplace(binding.mangled_name, Entry{i, binding}).second)
      reject("OWNERSHIP: multiple admitted owners for one region symbol");
    auto emitted = cg::emitExperimentalRegion(*unit.evidence, cg::ClosedCpuPolicy::GeneratedStrict);
    if (!emitted) reject(emitted.error);
    unit.emission = std::move(*emitted.emission);
    if (!helpers.insert(unit.emission->helper_symbol).second) reject("OWNERSHIP: duplicate issued helper");
    for (const auto &name : unit.emission->retired_value_helpers) retired[name].insert(i);
    unit.snapshot = fe::detail::ClosedRegionCompilationAccess::host(*unit.evidence);
  }
  if (entries.empty()) reject("program requires at least one admitted region");
  unsigned mains = 0;
  std::optional<std::size_t> main_owner;
  for (std::size_t i = 0; i < units.size(); ++i) {
    auto &unit = units[i];
    std::optional<InterfaceProof> initial;
    if (!unit.snapshot) {
      auto capture = host::prepareHostInputs(optionsFor(unit), cwd,
        {{fe::detail::closedRegionOwnedHeaderPath(), fe::detail::closedRegionOwnedHeaderSource()}}, error);
      if (!capture) reject(error);
      initial = scan(capture->arguments(), capture->workingDirectory(), capture->fileSystem(),
        capture->sourceSnapshot(), capture->inputPath(), headers, entries, i);
      unit.snapshot = capture->freeze(error);
      if (!unit.snapshot) reject(error);
    }
    auto replay = unit.snapshot->replay();
    if (!replay.ok(error)) reject(error);
    auto proof = scan(unit.snapshot->arguments(), unit.snapshot->workingDirectory(), replay.filesystem,
      unit.snapshot->sourceSnapshot(), unit.snapshot->inputPath(), headers, entries, i);
    if (!replay.ok(error) || (initial && *initial != proof)) reject("INTERFACE: frozen host interface replay differs: " + error);
    mains += proof.main_definitions;
    if (proof.main_definitions) main_owner = i;
    unit.foreign_definitions = std::move(proof.foreign_definitions);
    std::string ir;
    if (!cg::detail::compileFrozenHostToLLVM(*unit.snapshot, ir, error)) reject(error);
    unit.raw = parseIr(ir, context);
    cg::ArtifactSymbolOwnershipReport ownership;
    if (!cg::verifyHostArtifactSymbolOwnership(*unit.raw, artifacts, ownership, error)) reject(error);
    std::cout << "FROZEN " << i << " " << unit.snapshot->identity() << " " << unit.path.string() << "\n";
  }
  if (mains != 1) reject("ENTRY: program requires exactly one ordinary C++ main definition");
  // sameAbi deliberately compares complete parameter/target attributes. Clang
  // gives a definition extra sret-noalias and CPU facts absent on its ordinary
  // declaration. Generate a clean declaration counterpart from the already
  // sealed canonical parameter kinds; do not delete facts until a comparison
  // passes or implement a competing ABI comparator.
  std::string witness_source = "#include <matcore/region.h>\nnamespace md = matcore::mdsl;\n"
    "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"\n";
  unsigned ordinal = 0;
  for (const auto &[name, entry] : entries) {
    if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos)
      reject("research declaration witness requires ordinary ASCII linker spelling");
    const auto declaration = "__research_decl_" + std::to_string(ordinal);
    witness_source += "extern \"C\" md::Result " + declaration + "(";
    for (std::size_t i = 0; i < entry.binding.parameters.size(); ++i) {
      if (i) witness_source += ", ";
      witness_source += entry.binding.parameters[i].kind == ParameterKind::Storage ? "md::Storage" : "md::Shape";
    }
    witness_source += ") noexcept asm(\"" + name + "\");\n";
    witness_source += "extern \"C\" { auto *__research_use_" + std::to_string(ordinal++) +
      " = &" + declaration + "; }\n";
  }
  const auto witness_path = witness_directory / "interface.cpp";
  write(witness_path, witness_source);
  fe::Options witness_options;
  witness_options.input_path = witness_path.string();
  witness_options.clang_path = REGION_CLANG;
  witness_options.clang_resource_directory = REGION_RESOURCE_DIR;
  witness_options.compiler_arguments = {"-I/__mdsl_private__"};
  if (REGION_SANITIZED) witness_options.compiler_arguments.push_back("-fsanitize=address,undefined");
  auto witness_capture = host::prepareHostInputs(witness_options, witness_directory.string(),
    {{"/__mdsl_private__/fixture.h", "#pragma once\n"},
     {"/__mdsl_private__/matcore/region.h", fe::detail::experimentalRegionHeaderSource()},
     {"/__mdsl_private__/matcore/detail/region_storage.h", fe::detail::experimentalRegionStorageHeaderSource()}}, error);
  if (!witness_capture || witness_capture->sourceSnapshot() != witness_source)
    reject("compiler-issued interface witness differs from its source: " + error);
  clang::FileSystemOptions witness_file_options;
  witness_file_options.WorkingDir = witness_directory.string();
  auto witness_files = llvm::makeIntrusiveRefCnt<clang::FileManager>(witness_file_options, witness_capture->fileSystem());
  clang::tooling::ToolInvocation witness_syntax(witness_capture->arguments(),
    std::make_unique<clang::SyntaxOnlyAction>(), witness_files.get());
  if (!witness_syntax.run()) reject("compiler-issued declaration witness failed Sema");
  auto witness_snapshot = witness_capture->freeze(error);
  if (!witness_snapshot) reject(error);
  std::string witness_ir;
  if (!cg::detail::compileFrozenHostToLLVM(*witness_snapshot, witness_ir, error)) reject(error);
  auto witness = parseIr(witness_ir, context);
  std::map<std::string, const llvm::Function *> witness_declarations;
  for (const auto &function : *witness)
    if (!function.isIntrinsic()) witness_declarations.emplace(symbolKey(function.getName()), &function);
  std::cout << "DECLARATION_WITNESS " << witness_snapshot->identity() << "\n";
  for (std::size_t i = 0; i < units.size(); ++i) {
    const auto &module = *units[i].raw;
    if (module.getTargetTriple() != witness->getTargetTriple() || module.getDataLayoutStr() != witness->getDataLayoutStr())
      reject("ABI: source TU target or data layout differs from canonical declaration witness");
    for (const auto &global : module.global_values()) {
      const auto name = symbolKey(global.getName());
      if (name == "main" && !global.isDeclaration() && i != *main_owner)
        reject("ENTRY: foreign source symbol competes with main");
      if (helpers.contains(name)) reject("OWNERSHIP: source names an issued private helper");
      if (auto entry = entries.find(name); entry != entries.end()) {
        if (i != entry->second.owner && !global.isDeclaration()) {
          const auto *object = llvm::dyn_cast<llvm::GlobalObject>(&global);
          reject("OWNERSHIP: competing region definition before linker resolution: " + name +
            " linkage=" + std::to_string(global.getLinkage()) +
            " comdat=" + std::to_string(object && object->hasComdat()));
        }
        const auto *function = llvm::dyn_cast<llvm::Function>(&global);
        const auto *owner = i == entry->second.owner ? units[i].raw->getFunction(name) :
          (witness_declarations.contains(name) ? witness_declarations.at(name) : nullptr);
        if (!function || !owner || !cg::sameAbi(*function, *owner, false)) {
          std::string detail;
          llvm::raw_string_ostream out(detail);
          if (function) { out << "\nconsumer: "; function->print(out); }
          if (owner) { out << "\nproducer attributes: "; owner->getAttributes().print(out); }
          reject("ABI: foreign region declaration differs from sealed producer: " + name + detail);
        }
        // sameAbi intentionally is not an effect-summary comparator. Reject
        // additional declaration-side promises (e.g. memory(read/none)) too;
        // a clean compiler witness does not authorize source pure/const facts.
        if (i != entry->second.owner &&
            !cg::sameAttributes(function->getAttributes().getFnAttrs(), owner->getAttributes().getFnAttrs()))
          reject("DECLARATION: foreign LLVM function promises differ from compiler witness: " + name);
      }
      if (auto helper = retired.find(name); helper != retired.end() && !helper->second.contains(i) &&
          (!global.isDeclaration() || !global.use_empty()))
        reject("RETIREMENT: foreign TU uses a retired source-only Value helper: " + name);
    }
    if (!units[i].foreign_definitions.empty())
      reject("OWNERSHIP: foreign source defines an issued region even when Clang emits no definition: " +
        units[i].foreign_definitions.front());
  }
  std::cout << "VALIDATED program ownership and ABI before cross-TU link/optimization\n";
  auto unchanged = [&] {
    if (!witness_snapshot->unchanged(error)) reject(error);
    for (const auto &unit : units) {
      if (!unit.snapshot->unchanged(error)) reject(error);
      if (unit.compilation && !unit.compilation->inputsUnchanged(error)) reject(error);
    }
    for (const auto *artifact : {&clang, &linker, &candidates, &runtime, &public_header, &storage_header, &private_header})
      artifact->unchanged();
    if (provider) provider->unchanged();
  };
  unchanged();
  std::unique_ptr<llvm::Module> combined;
  for (std::size_t i = 0; i < units.size(); ++i) {
    auto &unit = units[i];
    std::unique_ptr<llvm::Module> module;
    if (unit.evidence) {
      auto compiled = cg::compileExperimentalRegionToLLVM(*unit.evidence,
        {REGION_CLANG, REGION_RESOURCE_DIR, installed.include, installed.header,
         helper_directories[i], bool(REGION_SANITIZED), artifacts}, cg::ClosedCpuPolicy::GeneratedStrict);
      if (!compiled) reject(compiled.error);
      unit.compilation = std::move(*compiled.compilation);
      module = parseIr(unit.compilation->llvm_ir, context);
    } else module = std::move(unit.raw);
    if (!combined) combined = std::move(module);
    else {
      std::cout << "CROSS_TU_LINK " << i << "\n";
      if (llvm::Linker::linkModules(*combined, std::move(module))) reject("ordinary LLVM program link failed");
    }
    unchanged();
  }
  for (const auto &[name, entry] : entries) {
    auto *function = combined->getFunction(name);
    const auto *original = units[entry.owner].raw->getFunction(name);
    if (!function || function->isDeclaration() || !cg::sameAbi(*function, *original, false))
      reject("ABI: linked region no longer matches its issued producer");
  }
  for (const auto &name : helpers) {
    auto *helper = combined->getFunction(name);
    if (!helper || helper->isDeclaration()) reject("OWNERSHIP: issued implementation missing after link");
    helper->setLinkage(llvm::GlobalValue::InternalLinkage);
  }
  std::string verification;
  llvm::raw_string_ostream diagnostics(verification);
  if (llvm::verifyModule(*combined, &diagnostics)) reject("invalid combined LLVM: " + verification);
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  combined->print(stream, nullptr);
  write(ir_path, ir);
  const auto issued_ir = Artifact::capture(ir_path);
  support::ProcessRequestV1 process;
  process.working_directory = staging.path;
  process.environment = support::compiler_environment_sanitization_v1();
  process.environment.push_back({"TMPDIR", (staging.path / "tool-tmp").string()});
  for (const auto *name : {"COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH", "LD_RUN_PATH",
       "LDEMULATION", "GNUTARGET", "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"})
    process.environment.push_back({name, std::nullopt});
  process.argv = {REGION_CLANG, "--no-default-config", "-resource-dir=" REGION_RESOURCE_DIR,
    "-x", "ir", ir_path.string(), "-O2", "-o", binary_path.string(), "--ld-path=" REGION_LINKER,
    "-x", "none", "-Xlinker", "--no-as-needed", installed.candidates.string(), installed.runtime.string(),
    "-lm", "-pthread", "-Xlinker", "-rpath", "-Xlinker", installed.runtime.parent_path().string(),
    "-Xlinker", "-rpath", "-Xlinker", installed.candidates.parent_path().string()};
  if (REGION_SANITIZED) process.argv.push_back("-fsanitize=address,undefined");
  if (provider) process.argv.insert(process.argv.end(), {provider->path.string(),
    "-Xlinker", "-rpath", "-Xlinker", provider->path.parent_path().string()});
  unchanged();
  std::cout << "NATIVE_LINK authenticated program\n";
  const auto linked = support::run_process_v1(process);
  if (!linked.launched || linked.exit_code != 0) reject("native program link failed: " + linked.stderr_text);
  issued_ir.unchanged(); unchanged();
  const auto final = Artifact::capture(binary_path);
  verifyOutput(final, false);
  fs::create_hard_link(binary_path, output);
  std::cout << "PUBLISHED " << final.sha << " " << output.string() << "\n";
  return 0;
}
} // namespace experiment

int main(int argc, char **argv) {
  try { return experiment::execute(argc, argv); }
  catch (const std::exception &error) {
    std::cerr << "multi-tu research: " << error.what() << '\n'; return 1;
  }
}
