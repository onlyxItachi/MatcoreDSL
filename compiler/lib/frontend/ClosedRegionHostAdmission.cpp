#include "ClosedRegionAdmissionInternal.h"

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/SmallString.h>
#include <cstdint>
#include <map>

namespace matcore::mdslc::frontend {
namespace {
namespace cr = closed_region;
namespace host = closed_region_host;

struct ParseState {
  cr::Program program;
  detail::ClosedRegionASTPolicy policy;
  std::string region_name;
  std::string error;
  bool visited = false;
  bool admitted = false;
  bool volatile_preprocessing = false;
  bool preprocessing_budget_exceeded = false;
};

class Diagnostics final : public clang::DiagnosticConsumer {
public:
  bool failed = false;
  std::string text;
  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic &diagnostic) override {
    clang::DiagnosticConsumer::HandleDiagnostic(level, diagnostic);
    if (level < clang::DiagnosticsEngine::Error) return;
    failed = true;
    llvm::SmallString<128> message;
    diagnostic.FormatDiagnostic(message);
    if (!text.empty()) text += "\n";
    text += message.str();
  }
};

// Clang owns preprocessing. These observations only fence the source ranges
// admitted below; they never expand macros or choose conditional branches.
class Preprocessing final : public clang::PPCallbacks {
public:
  Preprocessing(clang::SourceManager &sources, const clang::LangOptions &language,
                ParseState &state)
      : sources_(sources), language_(language), state_(state) {}
  void MacroExpands(const clang::Token &token,
                    const clang::MacroDefinition &, clang::SourceRange range,
                    const clang::MacroArgs *) override {
    const std::string name = token.getIdentifierInfo()
                                 ? token.getIdentifierInfo()->getName().str()
                                 : "<unnamed>";
    add(range, "macro expansion " + name);
    if (name == "__DATE__" || name == "__TIME__" || name == "__TIMESTAMP__")
      state_.volatile_preprocessing = true;
  }
  void SourceRangeSkipped(clang::SourceRange range,
                          clang::SourceLocation) override {
    add(range, "inactive conditional source");
  }
  void PragmaDirective(clang::SourceLocation location,
                       clang::PragmaIntroducerKind) override {
    // Clang's raw lexer supplies the logical directive extent, including
    // escaped newlines. A one-physical-line fence would miss annotations
    // injected from a continued #pragma clang attribute directive.
    auto begin = sources_.getExpansionLoc(location);
    auto end = begin;
    bool invalid = false, invalid_buffer = false;
    const char *bytes = sources_.getCharacterData(begin, &invalid);
    const auto buffer = sources_.getBufferData(sources_.getFileID(begin), &invalid_buffer);
    const auto base = reinterpret_cast<std::uintptr_t>(buffer.data());
    const auto position = reinterpret_cast<std::uintptr_t>(bytes);
    if (!invalid && !invalid_buffer && bytes && buffer.data() &&
        position >= base && position - base <= buffer.size()) {
      clang::Lexer lexer(begin, language_, bytes, bytes, buffer.end());
      clang::Token token;
      bool first = true;
      for (;;) {
        lexer.LexFromRawLexer(token);
        if (token.is(clang::tok::eof) || (!first && token.isAtStartOfLine())) break;
        end = token.getLocation();
        first = false;
      }
    } else {
      state_.preprocessing_budget_exceeded = true;
    }
    add({begin, end}, "pragma directive");
  }
  void If(clang::SourceLocation location, clang::SourceRange condition,
          ConditionValueKind) override { add({location, condition.getEnd()}, "if directive"); }
  void Ifdef(clang::SourceLocation location, const clang::Token &token,
             const clang::MacroDefinition &) override {
    add({location, token.getLocation()}, "ifdef directive");
  }
  void Ifndef(clang::SourceLocation location, const clang::Token &token,
              const clang::MacroDefinition &) override {
    add({location, token.getLocation()}, "ifndef directive");
  }
  void Elif(clang::SourceLocation location, clang::SourceRange condition,
            ConditionValueKind, clang::SourceLocation) override {
    add({location, condition.getEnd()}, "elif directive");
  }
  void Else(clang::SourceLocation location, clang::SourceLocation) override {
    add({location, location}, "else directive");
  }
  void Endif(clang::SourceLocation location, clang::SourceLocation) override {
    add({location, location}, "endif directive");
  }
private:
  void add(clang::SourceRange range, std::string description) {
    if (state_.policy.events.size() >= 250000) {
      state_.preprocessing_budget_exceeded = true;
      return;
    }
    state_.policy.events.push_back({range, std::move(description)});
  }
  clang::SourceManager &sources_;
  const clang::LangOptions &language_;
  ParseState &state_;
};

// Normalize callback evidence by immutable file bytes and actual source
// offsets, not SourceLocation raw encodings or presumed #line coordinates.
// This is a transcript, not a second preprocessor or a public interchange.
bool bindPreprocessing(clang::ASTContext &context, ParseState &state) {
  auto &sources = context.getSourceManager();
  std::map<unsigned, std::string> file_digests;
  std::string transcript;
  auto field = [&](const std::string &text) {
    transcript += std::to_string(text.size()) + ":" + text;
  };
  auto location = [&](clang::SourceLocation raw) {
    const auto point = sources.getExpansionLoc(raw);
    if (point.isInvalid() || point.isMacroID()) return false;
    const auto id = sources.getFileID(point);
    bool invalid_buffer = false, invalid_point = false;
    const auto buffer = sources.getBufferData(id, &invalid_buffer);
    const char *character = sources.getCharacterData(point, &invalid_point);
    const auto base = reinterpret_cast<std::uintptr_t>(buffer.data());
    const auto position = reinterpret_cast<std::uintptr_t>(character);
    if (invalid_buffer || invalid_point || !buffer.data() || !character ||
        position < base || position - base > buffer.size()) return false;
    auto digest = file_digests.find(id.getHashValue());
    if (digest == file_digests.end())
      digest = file_digests.emplace(id.getHashValue(),
          detail::closedRegionDigest(buffer.str())).first;
    field(sources.getFilename(point).str());
    field(digest->second);
    field(std::to_string(position - base));
    return true;
  };
  for (const auto &event : state.policy.events) {
    field(event.description);
    if (!location(event.range.getBegin()) || !location(event.range.getEnd())) {
      state.error = "preprocessor callback source range could not be authenticated";
      return false;
    }
  }
  state.program.compiler_identity += "; closed-host-pp-sha256=" +
                                     detail::closedRegionDigest(transcript);
  return true;
}

class Consumer final : public clang::ASTConsumer {
public:
  Consumer(clang::CompilerInstance &compiler, ParseState &state)
      : compiler_(compiler), state_(state) {}
  bool HandleTopLevelDecl(clang::DeclGroupRef group) override {
    if (!anchor_ && !group.isNull() && group.begin() != group.end())
      anchor_ = *group.begin();
    return true;
  }
  void HandleTranslationUnit(clang::ASTContext &context) override {
    state_.visited = true;
    if (compiler_.getDiagnostics().hasErrorOccurred()) return;
    auto main_entry = compiler_.getFileManager().getFileRef(state_.program.source_identity);
    if (!main_entry) {
      llvm::consumeError(main_entry.takeError());
      state_.error = "captured host main file was not resolved";
      return;
    }
    const auto main_id = context.getSourceManager().getMainFileID();
    bool invalid = false;
    const auto main_bytes = context.getSourceManager().getBufferData(main_id, &invalid);
    if (context.getSourceManager().translateFile(&main_entry->getFileEntry()) != main_id ||
        invalid || detail::closedRegionDigest(main_bytes.str()) != state_.program.source_sha256) {
      state_.error = "parsed host main FileID or immutable source bytes disagree with capture";
      return;
    }
    if (state_.volatile_preprocessing) {
      state_.error = "closed host context rejects volatile date/time/timestamp preprocessing";
      return;
    }
    if (state_.preprocessing_budget_exceeded) {
      state_.error = "closed host preprocessor evidence budget exceeded";
      return;
    }
    if (!bindPreprocessing(context, state_)) return;
    state_.policy.owned_header = detail::authenticateClosedRegionHeader(
        context, compiler_.getFileManager(), state_.error);
    if (!state_.error.empty()) return;
    state_.admitted = detail::admitClosedRegionParsedAST(
        context, anchor_, state_.region_name, state_.program,
        state_.policy, state_.error);
  }
private:
  clang::CompilerInstance &compiler_;
  ParseState &state_;
  const clang::Decl *anchor_ = nullptr;
};

class Action final : public clang::ASTFrontendAction {
public:
  explicit Action(ParseState &state) : state_(state) {}
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance &compiler, llvm::StringRef) override {
    compiler.getPreprocessor().addPPCallbacks(
        std::make_unique<Preprocessing>(compiler.getSourceManager(), compiler.getLangOpts(), state_));
    return std::make_unique<Consumer>(compiler, state_);
  }
private:
  ParseState &state_;
};

bool parse(const std::vector<std::string> &arguments,
           const std::string &working_directory,
           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> filesystem,
           ParseState &state, bool &syntax_valid) {
  clang::FileSystemOptions file_options;
  file_options.WorkingDir = working_directory;
  // ToolInvocation neither overlays mutable files nor adjusts these already
  // authenticated full driver arguments. FileManager and all driver lookups
  // share the recording/replay filesystem, including failed queries.
  auto files = llvm::makeIntrusiveRefCnt<clang::FileManager>(file_options, filesystem);
  Diagnostics diagnostics;
  clang::tooling::ToolInvocation invocation(
      arguments, std::make_unique<Action>(state), files.get());
  invocation.setDiagnosticConsumer(&diagnostics);
  const bool ran = invocation.run();
  syntax_valid = ran && state.visited && !diagnostics.failed;
  if (!syntax_valid) {
    state.error = "closed host Clang syntax/type validation failed: " + diagnostics.text;
    return false;
  }
  return state.admitted;
}

void initialize(ParseState &state, const std::string &source,
                const std::string &input, const std::string &region) {
  state.region_name = region;
  state.program.source_identity = input;
  state.program.source_sha256 = detail::closedRegionDigest(source);
  state.program.header_sha256 = detail::closedRegionDigest(detail::closedRegionOwnedHeaderSource());
  state.program.compiler_identity = clang::getClangFullVersion();
}
void bindContext(cr::Program &program, const std::string &identity) {
  program.compiler_identity += "; closed-host-context-sha256=" + identity;
}
} // namespace

namespace detail {
bool replayClosedRegionHost(const host::HostInputSnapshot &snapshot,
                           const std::string &region_name, cr::Program &program,
                           std::string &error) {
  auto replay = snapshot.replay();
  if (!replay.ok(error)) return false;
  ParseState state;
  initialize(state, snapshot.sourceSnapshot(), snapshot.inputPath(), region_name);
  bool syntax_valid = false;
  const bool admitted = parse(snapshot.arguments(), snapshot.workingDirectory(),
                             replay.filesystem, state, syntax_valid);
  if (!replay.ok(error)) return false;
  if (!admitted) {
    error = "immutable host-context replay rejected: " + state.error;
    return false;
  }
  bindContext(state.program, snapshot.identity());
  program = std::move(state.program);
  return true;
}

ClosedRegionAdmissionResult admitClosedRegionHostForTesting(
    const Options &options, const std::string &working_directory,
    const std::string &region_name, const std::function<void()> &after_initial_parse) {
  ClosedRegionAdmissionResult result;
  if (region_name.empty() || region_name.size() > 4096) {
    result.error = "bounded closed host admission requires a selected region";
    return result;
  }
  auto capture = host::prepareHostInputs(options, working_directory,
      {{closedRegionOwnedHeaderPath(), closedRegionOwnedHeaderSource()}}, result.error);
  if (!capture) return result;
  ParseState state;
  initialize(state, capture->sourceSnapshot(), capture->inputPath(), region_name);
  if (!parse(capture->arguments(), capture->workingDirectory(),
             capture->fileSystem(), state, result.syntax_valid)) {
    result.error = state.error;
    return result;
  }
  if (after_initial_parse) after_initial_parse();
  auto snapshot = capture->freeze(result.error);
  if (!snapshot) return result;
  bindContext(state.program, snapshot->identity());
  cr::Program replay;
  if (!replayClosedRegionHost(*snapshot, region_name, replay, result.error)) return result;
  // Reuse the existing exact semantic verifier; no second graph comparator or
  // private serialization is introduced to assert replay equivalence.
  mlir::MLIRContext context;
  auto module = cr::buildModule(state.program, context);
  if (!module || !cr::verifyModuleMatchesProgram(replay, *module.module, result.error)) {
    if (result.error.empty()) result.error = module.error;
    return result;
  }
  if (!snapshot->unchanged(result.error)) return result;
  return ClosedRegionEvidenceIssuer::host(std::move(state.program), std::move(snapshot), region_name);
}
} // namespace detail

ClosedRegionAdmissionResult admitClosedRegionHost(
    const Options &options, const std::string &working_directory,
    const std::string &region_name) {
  return detail::admitClosedRegionHostForTesting(options, working_directory, region_name, {});
}
} // namespace matcore::mdslc::frontend
