#include "frontend.h"
#include "native_frontend_evidence_internal.h"
#include "recovered_gemm_recognizer.h"
#include "../support/platform_support.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/ExprConcepts.h>
#include <clang/AST/ParentMapContext.h>
#include <clang/AST/RecordLayout.h>
#include <clang/AST/StmtCXX.h>
#include <clang/AST/TypeLoc.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Basic/CodeGenOptions.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileEntry.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/PPCallbacks.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <link.h>
#endif

namespace matcore::mdslc::frontend {
namespace {

namespace support = matcore::mdslc::support;

constexpr std::string_view kGemmName = "matcore::mdsl::gemm";
constexpr std::string_view kOutName = "matcore::mdsl::out";
constexpr std::string_view kGemmAnnotation = "matcore.op.gemm";
constexpr std::string_view kOutAnnotation = "matcore.wrapper.out";

std::string normalizeDisplayPath(const std::string &path) {
  std::string error;
  const std::optional<std::filesystem::path> native =
      support::path_from_utf8_v1(path, error);
  if (!native) return path;
  const std::optional<std::string> encoded =
      support::path_to_utf8_v1(native->lexically_normal(), error);
  return encoded.value_or(path);
}

std::string canonicalPath(const std::string &path) {
  std::string error;
  const std::optional<std::filesystem::path> native =
      support::path_from_utf8_v1(path, error);
  if (!native) return path;
  const std::filesystem::path canonical =
      support::normalize_path_v1(*native, true, error);
  const std::optional<std::string> encoded = support::path_to_utf8_v1(
      error.empty() ? canonical : native->lexically_normal(), error);
  return encoded.value_or(path);
}

std::optional<std::string> readFile(const std::string &path) {
  std::string error;
  const std::optional<std::filesystem::path> native =
      support::path_from_utf8_v1(path, error);
  if (!native) return std::nullopt;
  std::ifstream input(*native, std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

class CountingDiagnosticConsumer final : public clang::DiagnosticConsumer {
public:
  CountingDiagnosticConsumer()
      : printer_(llvm::errs(), diagnostic_options_) {}

  void BeginSourceFile(const clang::LangOptions &language_options,
                       const clang::Preprocessor *preprocessor) override {
    printer_.BeginSourceFile(language_options, preprocessor);
  }

  void EndSourceFile() override { printer_.EndSourceFile(); }

  void finish() override { printer_.finish(); }

  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic &diagnostic) override {
    clang::DiagnosticConsumer::HandleDiagnostic(level, diagnostic);
    printer_.HandleDiagnostic(level, diagnostic);
  }

private:
  clang::DiagnosticOptions diagnostic_options_;
  clang::TextDiagnosticPrinter printer_;
};

bool makeToolArguments(const Options &options,
                       std::vector<std::string> &arguments,
                       std::string &error) {
  const bool clang_cl = std::find(options.compiler_arguments.begin(),
                                  options.compiler_arguments.end(),
                                  "--driver-mode=cl") !=
                        options.compiler_arguments.end();
  if (options.clang_resource_directory.empty()) {
    error = "authenticated Clang resource directory is unavailable";
    return false;
  }
  arguments.push_back("-resource-dir=" + options.clang_resource_directory);
  for (std::size_t index = 0; index < options.compiler_arguments.size();
       ++index) {
    const std::string &argument = options.compiler_arguments[index];
    if (argument == options.input_path ||
        (!argument.starts_with('-') &&
         (!clang_cl || !argument.starts_with('/')) &&
         canonicalPath(argument) == canonicalPath(options.input_path))) {
      continue;
    }
    if (clang_cl && argument == "--driver-mode=cl") {
      if (std::find(arguments.begin(), arguments.end(), argument) ==
          arguments.end()) {
        arguments.push_back(argument);
      }
      continue;
    }
    if (clang_cl && argument == "--no-default-config") {
      continue;
    }
    if (clang_cl &&
        (support::windows_option_body_v1(argument) == "TP" ||
         argument == "-fno-color-diagnostics")) {
      continue;
    }
    if (argument == "-fsyntax-only" || argument == "-fno-color-diagnostics") {
      continue;
    }
    if (argument == "-x") {
      if (++index == options.compiler_arguments.size()) {
        error = "-x requires a language argument";
        return false;
      }
      if (options.compiler_arguments[index] != "c++") {
        error = ".mdsl extraction requires the exact language selection "
                "-x c++";
        return false;
      }
      continue;
    }
    const auto risk =
        support::classify_untrusted_compiler_argument_v1(argument, clang_cl);
    if (risk != support::CompilerArgumentRiskV1::none) {
      error = std::string(support::compiler_argument_risk_message_v1(risk)) +
              " is not valid for extraction: " + argument;
      return false;
    }
    const bool clang_cl_value_option =
        clang_cl && support::clang_cl_option_consumes_next_v1(argument);
    if (clang_cl_value_option) {
      if (++index == options.compiler_arguments.size()) {
        error = argument + " requires a value";
        return false;
      }
      if (!support::compiler_consumed_value_is_safe_v1(
              options.compiler_arguments[index])) {
        error = "nested response-file expansion is forbidden in the value "
                "for " +
                argument;
        return false;
      }
      arguments.push_back(argument);
      arguments.push_back(options.compiler_arguments[index]);
      continue;
    }
    arguments.push_back(argument);
  }
  if (clang_cl) {
    if (std::find(arguments.begin(), arguments.end(), "--driver-mode=cl") ==
        arguments.end()) {
      arguments.insert(arguments.begin(), "--driver-mode=cl");
    }
    arguments.insert(arguments.begin(), "--no-default-config");
    arguments.insert(arguments.end(),
                     {"/TP", "/Zs", "-fno-color-diagnostics"});
  } else {
    arguments.insert(arguments.end(), {"-x", "c++", "-fsyntax-only",
                                       "-fno-color-diagnostics"});
  }
  return true;
}

std::string shellQuoted(std::string_view value) {
  std::string result = "'";
  for (const char character : value) {
    if (character == '\'') {
      result += "'\\''";
    } else {
      result += character;
    }
  }
  result += '\'';
  return result;
}

using FileIdentity = std::pair<std::uint64_t, std::uint64_t>;

FileIdentity identityOf(const llvm::sys::fs::UniqueID &identity) {
  return {identity.getDevice(), identity.getFile()};
}

std::string snapshotHash(llvm::StringRef bytes) {
  const auto digest = llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes));
  return "sha256:" + llvm::toHex(llvm::ArrayRef(digest), true);
}

void bindIdentityField(std::string &identity, std::string_view value) {
  identity += std::to_string(value.size()) + ":";
  identity.append(value);
}

struct NativeState {
  NativeState(Result &frontend_result, std::string source_display_path,
              std::string source_canonical_path,
              std::string compilation_identity_value,
              std::string source_snapshot)
      : result(frontend_result), display_path(std::move(source_display_path)),
        canonical_input(std::move(source_canonical_path)),
        compilation_identity(std::move(compilation_identity_value)),
        source(std::move(source_snapshot)) {}

  Result &result;
  std::string display_path;
  std::string canonical_input;
  std::string compilation_identity;
  std::string source;
  std::set<FileIdentity> trusted_header_ids;
  std::map<FileIdentity, std::string> trusted_header_contents;
  std::set<std::string> trusted_header_external_macros;
  bool saw_direct_expected_include = false;
  bool saw_direct_trusted_include = false;
  bool saw_candidate = false;
  clang::SourceLocation first_candidate_location;
  std::vector<const clang::CallExpr *> calls;
  std::vector<const clang::CallExpr *> captured_calls;
  std::vector<const clang::DeclRefExpr *> function_references;
  bool inspect_recovered_cpp_gemm = false;
  bool inspect_two_gemm_regions = false;
  std::map<std::string, FileIdentity> region_dependency_ids;
  unsigned optimization_level = 0;
  std::string denormal_mode;
  std::string fp32_denormal_mode;
  bool denormal_mode_is_ieee = false;
  bool fp32_denormal_mode_is_ieee = false;
  std::vector<const clang::ForStmt *> recovered_outer_loops;
};

bool isTrustedFile(const clang::FileEntry *entry, const NativeState &state) {
  return entry != nullptr &&
         state.trusted_header_ids.contains(identityOf(entry->getUniqueID()));
}

bool declarationIsTrusted(const clang::Decl &declaration,
                          const clang::SourceManager &source_manager,
                          const NativeState &state) {
  const clang::SourceLocation location =
      source_manager.getSpellingLoc(declaration.getLocation());
  if (location.isInvalid()) {
    return false;
  }
  const clang::FileID file_id = source_manager.getFileID(location);
  const clang::FileEntry *entry = source_manager.getFileEntryForID(file_id);
  if (!isTrustedFile(entry, state)) {
    return false;
  }
  const auto snapshot =
      state.trusted_header_contents.find(identityOf(entry->getUniqueID()));
  if (snapshot == state.trusted_header_contents.end()) {
    return false;
  }
  bool invalid = false;
  const llvm::StringRef parsed = source_manager.getBufferData(file_id, &invalid);
  return !invalid && parsed == llvm::StringRef(snapshot->second);
}

Diagnostic diagnosticAt(const clang::SourceManager &source_manager,
                        clang::SourceLocation location,
                        const NativeState &state, std::string message) {
  if (location.isMacroID()) {
    location = source_manager.getExpansionLoc(location);
  } else {
    location = source_manager.getSpellingLoc(location);
  }
  Diagnostic diagnostic{.file = {},
                        .line = 0,
                        .column = 0,
                        .message = std::move(message)};
  if (location.isInvalid()) {
    diagnostic.file = state.display_path;
    return diagnostic;
  }
  if (source_manager.isWrittenInMainFile(location)) {
    diagnostic.file = state.display_path;
  } else {
    const clang::PresumedLoc presumed = source_manager.getPresumedLoc(location);
    if (presumed.isValid()) {
      diagnostic.file = presumed.getFilename();
    }
  }
  if (diagnostic.file.empty()) {
    diagnostic.file = source_manager.getFilename(location).str();
  }
  diagnostic.line = source_manager.getSpellingLineNumber(location);
  diagnostic.column = source_manager.getSpellingColumnNumber(location);
  return diagnostic;
}

class TrustedHeaderCallbacks final : public clang::PPCallbacks {
public:
  TrustedHeaderCallbacks(const clang::SourceManager &source_manager,
                         NativeState &state)
      : source_manager_(source_manager), state_(state) {}

  void InclusionDirective(
      clang::SourceLocation hash_location, const clang::Token &,
      llvm::StringRef file_name, bool,
      clang::CharSourceRange,
      clang::OptionalFileEntryRef file, llvm::StringRef, llvm::StringRef,
      const clang::Module *, bool,
      clang::SrcMgr::CharacteristicKind) override {
    const clang::SourceLocation spelling =
        source_manager_.getSpellingLoc(hash_location);
    if (!source_manager_.isWrittenInMainFile(spelling) ||
        file_name != "matcore/mdsl.h") {
      return;
    }
    state_.saw_direct_expected_include = true;
    // PPCallbacks has already resolved macro-expanded include operands here.
    // Authenticate the resulting FileEntry rather than the token spelling so
    // an active main-file include macro remains valid without trusting a
    // shadow or copied header.
    if (file && isTrustedFile(&file->getFileEntry(), state_)) {
      state_.saw_direct_trusted_include = true;
    }
  }

  void MacroExpands(const clang::Token &macro_name,
                    const clang::MacroDefinition &definition,
                    clang::SourceRange range,
                    const clang::MacroArgs *) override {
    const clang::SourceLocation use =
        source_manager_.getExpansionLoc(range.getBegin());
    if (use.isInvalid()) {
      return;
    }
    const clang::FileID file_id = source_manager_.getFileID(use);
    if (!isTrustedFile(source_manager_.getFileEntryForID(file_id), state_)) {
      return;
    }
    const clang::IdentifierInfo *identifier = macro_name.getIdentifierInfo();
    const clang::MacroInfo *macro = definition.getMacroInfo();
    if (identifier == nullptr || macro == nullptr || macro->isBuiltinMacro() ||
        identifier->getName() == "MATCORE_MDSL_ANNOTATE") {
      return;
    }
    const clang::SourceLocation definition_location =
        source_manager_.getExpansionLoc(macro->getDefinitionLoc());
    if (definition_location.isValid() &&
        isTrustedFile(source_manager_.getFileEntryForID(
                          source_manager_.getFileID(definition_location)),
                      state_)) {
      return;
    }
    state_.trusted_header_external_macros.insert(identifier->getName().str());
  }

private:
  const clang::SourceManager &source_manager_;
  NativeState &state_;
};

const clang::RecordDecl *recordDeclaration(clang::QualType type) {
  type = type.getCanonicalType().getUnqualifiedType();
  const auto *record_type = type->getAs<clang::RecordType>();
  return record_type == nullptr ? nullptr : record_type->getDecl();
}

bool isNamedRecord(clang::QualType type, std::string_view name) {
  const clang::RecordDecl *record = recordDeclaration(type);
  return record != nullptr && record->getCanonicalDecl()
                                  ->getQualifiedNameAsString() == name;
}

bool isConstLvalueReferenceTo(clang::QualType type, std::string_view name) {
  type = type.getCanonicalType();
  const auto *reference = type->getAs<clang::LValueReferenceType>();
  if (reference == nullptr) {
    return false;
  }
  const clang::QualType pointee = reference->getPointeeType();
  return pointee.isConstQualified() && isNamedRecord(pointee, name);
}

bool isMutableLvalueReferenceTo(clang::QualType type, std::string_view name) {
  type = type.getCanonicalType();
  const auto *reference = type->getAs<clang::LValueReferenceType>();
  if (reference == nullptr) {
    return false;
  }
  const clang::QualType pointee = reference->getPointeeType();
  return !pointee.isConstQualified() && isNamedRecord(pointee, name);
}

bool hasExpectedGemmSignature(const clang::FunctionDecl &declaration) {
  return declaration.getReturnType()->isVoidType() &&
         !declaration.isVariadic() && declaration.getNumParams() == 4 &&
         isNamedRecord(declaration.getParamDecl(0)->getType(),
                       "matcore::mdsl::out_arg") &&
         isConstLvalueReferenceTo(declaration.getParamDecl(1)->getType(),
                                  "matcore::mdsl::matrix_view") &&
         isConstLvalueReferenceTo(declaration.getParamDecl(2)->getType(),
                                  "matcore::mdsl::matrix_view") &&
         isNamedRecord(declaration.getParamDecl(3)->getType(),
                       "matcore::mdsl::policy") &&
         declaration.getParamDecl(3)->hasDefaultArg();
}

bool hasExpectedOutSignature(const clang::FunctionDecl &declaration) {
  return !declaration.isVariadic() && declaration.getNumParams() == 1 &&
         isNamedRecord(declaration.getReturnType(),
                       "matcore::mdsl::out_arg") &&
         isMutableLvalueReferenceTo(declaration.getParamDecl(0)->getType(),
                                    "matcore::mdsl::matrix_view");
}

const clang::RecordDecl *recordDefinition(const clang::RecordDecl *record) {
  if (record == nullptr) {
    return nullptr;
  }
  const clang::RecordDecl *definition = record->getDefinition();
  return definition == nullptr ? record : definition;
}

std::vector<const clang::FieldDecl *>
recordFields(const clang::RecordDecl &record) {
  std::vector<const clang::FieldDecl *> fields;
  for (const clang::FieldDecl *field : record.fields()) {
    fields.push_back(field);
  }
  return fields;
}

std::uint64_t alignTo(std::uint64_t value, std::uint64_t alignment) {
  return alignment == 0 ? value
                        : ((value + alignment - 1) / alignment) * alignment;
}

bool hasNaturalRecordLayout(const clang::RecordDecl &record,
                            const std::vector<const clang::FieldDecl *> &fields,
                            const clang::ASTContext &context) {
  const clang::ASTRecordLayout &layout = context.getASTRecordLayout(&record);
  std::uint64_t expected_offset = 0;
  std::uint64_t expected_alignment = context.getCharWidth();
  for (std::size_t index = 0; index < fields.size(); ++index) {
    const std::uint64_t field_alignment =
        context.getTypeAlign(fields[index]->getType());
    expected_alignment = std::max(expected_alignment, field_alignment);
    expected_offset = alignTo(expected_offset, field_alignment);
    if (layout.getFieldOffset(static_cast<unsigned>(index)) !=
        expected_offset) {
      return false;
    }
    expected_offset += context.getTypeSize(fields[index]->getType());
  }
  const std::uint64_t expected_size =
      alignTo(expected_offset, expected_alignment);
  return static_cast<std::uint64_t>(context.toBits(layout.getAlignment())) ==
             expected_alignment &&
         static_cast<std::uint64_t>(context.toBits(layout.getSize())) ==
             expected_size;
}

bool isFloatPointer(clang::QualType type) {
  type = type.getCanonicalType().getUnqualifiedType();
  const auto *pointer = type->getAs<clang::PointerType>();
  return pointer != nullptr &&
         pointer->getPointeeType()
             ->isSpecificBuiltinType(clang::BuiltinType::Float);
}

bool isSignedIntegerWithWidth(clang::QualType type, unsigned width,
                              const clang::ASTContext &context) {
  type = type.getCanonicalType().getUnqualifiedType();
  return type->isSignedIntegerType() && context.getTypeSize(type) == width;
}

bool isPointerToRecord(clang::QualType type, std::string_view name) {
  type = type.getCanonicalType().getUnqualifiedType();
  const auto *pointer = type->getAs<clang::PointerType>();
  return pointer != nullptr && isNamedRecord(pointer->getPointeeType(), name);
}

const clang::EnumDecl *enumDefinition(clang::QualType type) {
  type = type.getCanonicalType().getUnqualifiedType();
  const auto *enumeration_type = type->getAs<clang::EnumType>();
  if (enumeration_type == nullptr) {
    return nullptr;
  }
  const clang::EnumDecl *enumeration = enumeration_type->getDecl();
  const clang::EnumDecl *definition = enumeration->getDefinition();
  return definition == nullptr ? enumeration : definition;
}

bool hasExpectedEnum(const clang::EnumDecl &enumeration,
                     std::string_view qualified_name,
                     const std::vector<std::pair<std::string_view, int>> &values,
                     const clang::ASTContext &context,
                     const clang::SourceManager &source_manager,
                     const NativeState &state) {
  if (enumeration.getCanonicalDecl()->getQualifiedNameAsString() !=
          qualified_name ||
      !declarationIsTrusted(enumeration, source_manager, state)) {
    return false;
  }
  clang::QualType integer_type = enumeration.getIntegerType();
  if (integer_type.isNull()) {
    return false;
  }
  integer_type = integer_type.getCanonicalType().getUnqualifiedType();
  if (!integer_type->isUnsignedIntegerType() ||
      context.getTypeSize(integer_type) != 32) {
    return false;
  }
  auto expected = values.begin();
  for (const clang::EnumConstantDecl *constant : enumeration.enumerators()) {
    if (expected == values.end() ||
        constant->getName() !=
            llvm::StringRef(expected->first.data(), expected->first.size()) ||
        constant->getInitVal() != expected->second) {
      return false;
    }
    ++expected;
  }
  return expected == values.end();
}

bool validatePublicAbi(const clang::FunctionDecl &gemm,
                       const clang::ASTContext &context,
                       const NativeState &state, std::string &error) {
  const clang::SourceManager &source_manager = context.getSourceManager();
  const clang::RecordDecl *out =
      recordDefinition(recordDeclaration(gemm.getParamDecl(0)->getType()));
  clang::QualType lhs_type = gemm.getParamDecl(1)->getType().getCanonicalType();
  const auto *lhs_reference = lhs_type->getAs<clang::LValueReferenceType>();
  const clang::RecordDecl *matrix =
      lhs_reference == nullptr
          ? nullptr
          : recordDefinition(recordDeclaration(lhs_reference->getPointeeType()));
  const clang::RecordDecl *policy =
      recordDefinition(recordDeclaration(gemm.getParamDecl(3)->getType()));
  if (out == nullptr || matrix == nullptr || policy == nullptr ||
      !out->isCompleteDefinition() || !matrix->isCompleteDefinition() ||
      !policy->isCompleteDefinition()) {
    error = "public matrix descriptor records are incomplete";
    return false;
  }
  for (const auto &[record, name] :
       std::vector<std::pair<const clang::RecordDecl *, std::string_view>>{
           {matrix, "matcore::mdsl::matrix_view"},
           {out, "matcore::mdsl::out_arg"},
           {policy, "matcore::mdsl::policy"}}) {
    const auto *cxx = llvm::dyn_cast<clang::CXXRecordDecl>(record);
    if (record->getCanonicalDecl()->getQualifiedNameAsString() != name ||
        !declarationIsTrusted(*record, source_manager, state) || cxx == nullptr ||
        cxx->getNumBases() != 0 || !cxx->isAggregate() ||
        !cxx->isStandardLayout()) {
      error = "public record semantics differ from <matcore/mdsl.h>";
      return false;
    }
  }

  const std::vector<const clang::FieldDecl *> matrix_fields =
      recordFields(*matrix);
  const std::vector<const clang::FieldDecl *> out_fields = recordFields(*out);
  const std::vector<const clang::FieldDecl *> policy_fields =
      recordFields(*policy);
  if (matrix_fields.size() != 3 ||
      matrix_fields[0]->getName() != "data" ||
      !isFloatPointer(matrix_fields[0]->getType()) ||
      matrix_fields[1]->getName() != "rows" ||
      !isSignedIntegerWithWidth(matrix_fields[1]->getType(), 64, context) ||
      matrix_fields[2]->getName() != "columns" ||
      !isSignedIntegerWithWidth(matrix_fields[2]->getType(), 64, context) ||
      out_fields.size() != 1 || out_fields[0]->getName() != "value" ||
      !isPointerToRecord(out_fields[0]->getType(),
                         "matcore::mdsl::matrix_view") ||
      policy_fields.size() != 2 ||
      policy_fields[0]->getName() != "target" ||
      policy_fields[1]->getName() != "fallback" ||
      !policy_fields[0]->hasInClassInitializer() ||
      !policy_fields[1]->hasInClassInitializer()) {
    error = "public descriptor fields or types differ from the f32 ABI";
    return false;
  }
  if (!hasNaturalRecordLayout(*matrix, matrix_fields, context) ||
      !hasNaturalRecordLayout(*out, out_fields, context) ||
      !hasNaturalRecordLayout(*policy, policy_fields, context)) {
    error = "public descriptor layout was changed by compiler state";
    return false;
  }

  const clang::EnumDecl *target =
      enumDefinition(policy_fields[0]->getType());
  const clang::EnumDecl *fallback =
      enumDefinition(policy_fields[1]->getType());
  if (target == nullptr || fallback == nullptr ||
      !hasExpectedEnum(*target, "matcore::mdsl::target",
                       {{"cpu", 0}, {"cuda", 1}}, context, source_manager,
                       state) ||
      !hasExpectedEnum(*fallback, "matcore::mdsl::fallback", {{"error", 0}},
                       context, source_manager, state)) {
    error = "public policy enum semantics differ from <matcore/mdsl.h>";
    return false;
  }
  return true;
}

enum class Authentication { valid, untrusted, wrong_signature, annotation };

Authentication authenticateDeclaration(
    const clang::FunctionDecl &declaration, std::string_view expected_name,
    std::string_view expected_annotation,
    const clang::SourceManager &source_manager, const NativeState &state,
    std::string &annotation_error) {
  const clang::FunctionDecl *canonical = declaration.getCanonicalDecl();
  if (canonical->getQualifiedNameAsString() != expected_name ||
      !declarationIsTrusted(*canonical, source_manager, state)) {
    return Authentication::untrusted;
  }
  const bool signature_valid =
      expected_name == kGemmName ? hasExpectedGemmSignature(*canonical)
                                 : hasExpectedOutSignature(*canonical);
  if (!signature_valid) {
    return Authentication::wrong_signature;
  }

  unsigned annotations = 0;
  for (const clang::FunctionDecl *redeclaration : canonical->redecls()) {
    for (const clang::AnnotateAttr *attribute :
         redeclaration->specific_attrs<clang::AnnotateAttr>()) {
      if (attribute->isInherited()) {
        continue;
      }
      ++annotations;
      if (attribute->getAnnotation() !=
          llvm::StringRef(expected_annotation.data(),
                          expected_annotation.size())) {
        annotation_error = "conflicting or unsupported annotation payload '" +
                           attribute->getAnnotation().str() + "'";
      }
    }
  }
  if (annotations != 1 || !annotation_error.empty()) {
    if (annotation_error.empty()) {
      annotation_error = annotations == 0
                             ? "required annotation payload is missing"
                             : "duplicate annotation payloads are not supported";
    }
    return Authentication::annotation;
  }
  return Authentication::valid;
}

bool hasReservedOperationAnnotation(const clang::FunctionDecl &declaration) {
  const clang::FunctionDecl *canonical = declaration.getCanonicalDecl();
  for (const clang::FunctionDecl *redeclaration : canonical->redecls()) {
    for (const clang::AnnotateAttr *attribute :
         redeclaration->specific_attrs<clang::AnnotateAttr>()) {
      if (!attribute->isInherited() &&
          attribute->getAnnotation().starts_with("matcore.op.")) {
        return true;
      }
    }
  }
  return false;
}

const clang::DeclRefExpr *directCalleeReference(const clang::CallExpr &call) {
  const clang::Expr *callee = call.getCallee()->IgnoreParenImpCasts();
  return llvm::dyn_cast<clang::DeclRefExpr>(callee);
}

bool hasCanonicalMdslQualifier(const clang::DeclRefExpr &reference) {
  if (!reference.hasQualifier() ||
      llvm::isa<clang::UsingShadowDecl>(reference.getFoundDecl())) {
    return false;
  }
#if LLVM_VERSION_MAJOR >= 22
  const clang::NestedNameSpecifier qualifier = reference.getQualifier();
  if (qualifier.getKind() != clang::NestedNameSpecifier::Kind::Namespace)
    return false;
  const clang::NamespaceBaseDecl *namespace_base =
      qualifier.getAsNamespaceAndPrefix().Namespace;
  const clang::NamespaceDecl *resolved =
      namespace_base == nullptr ? nullptr : namespace_base->getNamespace();
#else
  const clang::NestedNameSpecifier *qualifier = reference.getQualifier();
  const clang::NamespaceDecl *resolved = nullptr;
  if (qualifier->getKind() == clang::NestedNameSpecifier::Namespace) {
    resolved = qualifier->getAsNamespace();
  } else if (qualifier->getKind() ==
             clang::NestedNameSpecifier::NamespaceAlias) {
    const clang::NamespaceAliasDecl *alias = qualifier->getAsNamespaceAlias();
    resolved = alias == nullptr ? nullptr : alias->getNamespace();
  }
#endif
  return resolved != nullptr &&
         resolved->getCanonicalDecl()->getQualifiedNameAsString() ==
             "matcore::mdsl";
}

bool simpleLvalue(const clang::Expr &expression, std::string &name) {
  const clang::Expr *current = expression.IgnoreParenImpCasts();
  if (const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(current)) {
    if (!llvm::isa<clang::VarDecl>(reference->getDecl()) &&
        !llvm::isa<clang::ParmVarDecl>(reference->getDecl())) {
      return false;
    }
    name = reference->getDecl()->getNameAsString();
    return !name.empty();
  }
  if (const auto *member = llvm::dyn_cast<clang::MemberExpr>(current)) {
    std::string base_name;
    const clang::Expr *base = member->getBase()->IgnoreParenImpCasts();
    if (!llvm::isa<clang::CXXThisExpr>(base) &&
        !simpleLvalue(*base, base_name)) {
      return false;
    }
    name = member->getMemberDecl()->getNameAsString();
    if (name.empty()) {
      return false;
    }
    if (!base_name.empty()) {
      name = base_name + "." + name;
    }
    return true;
  }
  return false;
}

const clang::EnumConstantDecl *
directEnumConstant(const clang::Expr &expression) {
  const clang::Expr *current = &expression;
  while (true) {
    current = current->IgnoreParenImpCasts();
    if (const auto *default_init =
            llvm::dyn_cast<clang::CXXDefaultInitExpr>(current)) {
      current = default_init->getExpr();
      continue;
    }
    if (const auto *constant = llvm::dyn_cast<clang::ConstantExpr>(current)) {
      current = constant->getSubExpr();
      continue;
    }
    break;
  }
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(current);
  return reference == nullptr
             ? nullptr
             : llvm::dyn_cast<clang::EnumConstantDecl>(reference->getDecl());
}

bool directEnumValue(const clang::Expr &expression,
                     std::string_view expected_type, std::string &value) {
  const clang::EnumConstantDecl *constant = directEnumConstant(expression);
  const auto *enumeration =
      constant == nullptr
          ? nullptr
          : llvm::dyn_cast<clang::EnumDecl>(constant->getDeclContext());
  if (enumeration == nullptr ||
      enumeration->getCanonicalDecl()->getQualifiedNameAsString() !=
          expected_type) {
    return false;
  }
  value = constant->getNameAsString();
  return !value.empty();
}

const clang::InitListExpr *policyInitializer(const clang::Expr &expression) {
  const clang::Expr *current = expression.IgnoreParenImpCasts();
  if (const auto *cast =
          llvm::dyn_cast<clang::CXXFunctionalCastExpr>(current)) {
    current = cast->getSubExpr()->IgnoreParenImpCasts();
  } else if (const auto *temporary =
                 llvm::dyn_cast<clang::CXXTemporaryObjectExpr>(current)) {
    if (temporary->getNumArgs() != 1) {
      return nullptr;
    }
    current = temporary->getArg(0)->IgnoreParenImpCasts();
  }
  return llvm::dyn_cast<clang::InitListExpr>(current);
}

bool parsePolicy(const clang::Expr &expression, std::string &target,
                 std::string &fallback) {
  target = "cpu";
  fallback = "error";
  const clang::Expr *current = expression.IgnoreParenImpCasts();
  if (llvm::isa<clang::CXXDefaultArgExpr>(current)) {
    return true;
  }
  const clang::InitListExpr *initializer = policyInitializer(expression);
  if (initializer == nullptr || initializer->getNumInits() != 2) {
    return false;
  }
  return directEnumValue(*initializer->getInit(0), "matcore::mdsl::target",
                         target) &&
         directEnumValue(*initializer->getInit(1), "matcore::mdsl::fallback",
                         fallback);
}

std::optional<ir::SourceRange>
mainFileTokenRange(const clang::Expr &expression,
                   const clang::SourceManager &source_manager,
                   const clang::LangOptions &language_options) {
  clang::SourceLocation begin = expression.getBeginLoc();
  clang::SourceLocation end = expression.getEndLoc();
  if (begin.isInvalid() || end.isInvalid() || begin.isMacroID() ||
      end.isMacroID()) {
    return std::nullopt;
  }
  begin = source_manager.getSpellingLoc(begin);
  end = source_manager.getSpellingLoc(end);
  if (!source_manager.isWrittenInMainFile(begin) ||
      !source_manager.isWrittenInMainFile(end)) {
    return std::nullopt;
  }
  const clang::SourceLocation after_end = clang::Lexer::getLocForEndOfToken(
      end, 0, source_manager, language_options);
  if (after_end.isInvalid() ||
      source_manager.getFileID(begin) != source_manager.getFileID(after_end)) {
    return std::nullopt;
  }
  return ir::SourceRange{
      .begin = source_manager.getFileOffset(begin),
      .end = source_manager.getFileOffset(after_end),
  };
}

bool hasMacroOrigin(const clang::Stmt &statement) {
  if (statement.getBeginLoc().isMacroID() ||
      statement.getEndLoc().isMacroID()) {
    return true;
  }
  for (const clang::Stmt *child : statement.children()) {
    if (child != nullptr && hasMacroOrigin(*child)) {
      return true;
    }
  }
  return false;
}

enum class ContextViolation {
  none,
  template_context,
  lambda,
  constexpr_context,
  unevaluated_context
};

ContextViolation forbiddenContext(const clang::DynTypedNode &node,
                                  clang::ASTContext &context,
                                  unsigned depth = 0) {
  if (depth > 64) {
    return ContextViolation::template_context;
  }
  for (const clang::DynTypedNode &parent : context.getParents(node)) {
    if (const auto *parent_call = parent.get<clang::CallExpr>()) {
      if (parent_call->isUnevaluatedBuiltinCall(context)) {
        return ContextViolation::unevaluated_context;
      }
    }
    if (parent.get<clang::CXXNoexceptExpr>() != nullptr ||
        parent.get<clang::UnaryExprOrTypeTraitExpr>() != nullptr ||
        parent.get<clang::TypeTraitExpr>() != nullptr ||
        parent.get<clang::ExpressionTraitExpr>() != nullptr ||
        parent.get<clang::RequiresExpr>() != nullptr ||
        parent.get<clang::CXXTypeidExpr>() != nullptr ||
        parent.get<clang::ChooseExpr>() != nullptr ||
        parent.get<clang::GenericSelectionExpr>() != nullptr ||
        parent.get<clang::CXXUuidofExpr>() != nullptr) {
      return ContextViolation::unevaluated_context;
    }
    if (const clang::TypeLoc *type_location = parent.get<clang::TypeLoc>()) {
      if (!type_location->isNull() &&
          llvm::isa<clang::DecltypeType>(type_location->getTypePtr())) {
        return ContextViolation::unevaluated_context;
      }
    }
    if (const clang::Type *type = parent.get<clang::Type>()) {
      if (llvm::isa<clang::DecltypeType>(type)) {
        return ContextViolation::unevaluated_context;
      }
    }
    if (parent.get<clang::LambdaExpr>() != nullptr) {
      return ContextViolation::lambda;
    }
    if (parent.get<clang::FunctionTemplateDecl>() != nullptr ||
        parent.get<clang::ClassTemplateDecl>() != nullptr ||
        parent.get<clang::ClassTemplateSpecializationDecl>() != nullptr) {
      return ContextViolation::template_context;
    }
    if (parent.get<clang::ConstantExpr>() != nullptr ||
        parent.get<clang::StaticAssertDecl>() != nullptr) {
      return ContextViolation::constexpr_context;
    }
    if (const auto *function = parent.get<clang::FunctionDecl>()) {
      if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(function)) {
        if (method->getParent()->isLambda()) {
          return ContextViolation::lambda;
        }
      }
      if (function->isConstexpr()) {
        return ContextViolation::constexpr_context;
      }
      if (function->getTemplatedKind() !=
          clang::FunctionDecl::TK_NonTemplate) {
        return ContextViolation::template_context;
      }
    }
    const ContextViolation nested = forbiddenContext(parent, context, depth + 1);
    if (nested != ContextViolation::none) {
      return nested;
    }
  }
  return ContextViolation::none;
}

class MatchCollector final
    : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  explicit MatchCollector(NativeState &state) : state_(state) {}

  void run(const clang::ast_matchers::MatchFinder::MatchResult &result) override {
    if (const auto *call =
            result.Nodes.getNodeAs<clang::CallExpr>("matcore-call")) {
      state_.calls.push_back(call);
    }
    if (const auto *reference =
            result.Nodes.getNodeAs<clang::DeclRefExpr>("function-reference")) {
      state_.function_references.push_back(reference);
    }
    if (const auto *outer =
            result.Nodes.getNodeAs<clang::ForStmt>("recovered-outer-loop")) {
      state_.recovered_outer_loops.push_back(outer);
    }
  }

private:
  NativeState &state_;
};

class NativeConsumer final : public clang::ASTConsumer {
public:
  explicit NativeConsumer(NativeState &state)
      : state_(state), collector_(state) {
    using namespace clang::ast_matchers;
    finder_.addMatcher(callExpr().bind("matcore-call"), &collector_);
    finder_.addMatcher(declRefExpr(to(functionDecl())).bind("function-reference"),
                       &collector_);
    if (state_.inspect_recovered_cpp_gemm) {
      finder_.addMatcher(
          forStmt(unless(hasAncestor(forStmt())),
                  unless(isExpansionInSystemHeader()))
              .bind("recovered-outer-loop"),
          &collector_);
    }
  }

  void HandleTranslationUnit(clang::ASTContext &context) override {
    finder_.matchAST(context);
    finalize(context);
  }

private:
  // A binding identity is deliberately not a runtime pointer identity. Two
  // distinct parameters (even references) may still denote the same storage.
  std::string declarationIdentity(const clang::Decl &declaration,
                                  clang::ASTContext &context) {
    const auto &manager = context.getSourceManager();
    const auto declared_location = declaration.getCanonicalDecl()->getLocation();
    if (declared_location.isInvalid() || declared_location.isMacroID()) return {};
    const auto location = manager.getSpellingLoc(declared_location);
    const auto file_id = manager.getFileID(location);
    bool invalid = false;
    const auto buffer = manager.getBufferData(file_id, &invalid);
    if (invalid) return {};
    std::string identity = "mdsl-declaration-v1";
    bindIdentityField(identity, canonicalPath(manager.getFilename(location).str()));
    bindIdentityField(identity, snapshotHash(buffer));
    bindIdentityField(identity, std::to_string(manager.getFileOffset(location)));
    bindIdentityField(identity, declaration.getDeclKindName());
    return snapshotHash(identity);
  }

  const clang::VarDecl *descriptorRoot(const clang::VarDecl *declaration) {
    std::set<const clang::VarDecl *> visited;
    while (declaration != nullptr && visited.size() < 64) {
      declaration = declaration->getCanonicalDecl();
      if (!visited.insert(declaration).second) return nullptr;
      if (!declaration->getType()->isReferenceType() ||
          llvm::isa<clang::ParmVarDecl>(declaration)) return declaration;
      // Do not chase nonlocal references or initializer expressions with any
      // operation. Unsupported reference bindings remain an admission barrier.
      if (!declaration->isLocalVarDecl() || !declaration->hasLocalStorage() ||
          !declaration->hasInit()) return nullptr;
      const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(
          declaration->getInit()->IgnoreParens());
      if (reference == nullptr) return nullptr;
      declaration = llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
    }
    return nullptr;
  }

  bool descriptorBinding(const clang::Expr &expression, unsigned stage,
                         clang::ASTContext &context,
                         TwoGemmDescriptorBindingV1 &binding) {
    const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(
        expression.IgnoreParenImpCasts());
    if (reference == nullptr || expression.HasSideEffects(context)) return false;
    const auto *declaration = llvm::dyn_cast<clang::VarDecl>(reference->getDecl());
    const auto *root = descriptorRoot(declaration);
    if (declaration == nullptr || root == nullptr) return false;
    const auto range = mainFileTokenRange(expression, context.getSourceManager(),
                                          context.getLangOpts());
    if (!range || range->end > state_.source.size()) return false;
    binding.declaration_id = declarationIdentity(*declaration, context);
    binding.descriptor_id = declarationIdentity(*root, context);
    binding.source_expression = state_.source.substr(range->begin,
                                                    range->end - range->begin);
    binding.snapshot_stage = stage;
    return !binding.declaration_id.empty() && !binding.descriptor_id.empty();
  }

  const clang::CompoundStmt *directCompound(const clang::Stmt &statement,
                                           clang::ASTContext &context) {
    const auto parents = context.getParents(statement);
    return parents.size() == 1 ? parents[0].get<clang::CompoundStmt>() : nullptr;
  }

  bool supportedCompound(const clang::CompoundStmt &compound,
                         const clang::FunctionDecl &function,
                         clang::ASTContext &context) {
    if (function.getBody() == &compound) return true;
    const auto parents = context.getParents(compound);
    if (parents.size() != 1) return false;
    const auto *try_statement = parents[0].get<clang::CXXTryStmt>();
    return try_statement != nullptr && try_statement->getTryBlock() == &compound &&
           directCompound(*try_statement, context) == function.getBody();
  }

  bool regionDeclarationUncontaminated(const clang::CallExpr &call,
                                       clang::ASTContext &context) {
    const auto *canonical = call.getDirectCallee()->getCanonicalDecl();
    if (canonical->getDefinition() != nullptr) return false;
    const auto &manager = context.getSourceManager();
    for (const auto *redeclaration : canonical->redecls()) {
      for (const auto *attribute : redeclaration->attrs()) {
        if (attribute->isInherited()) continue;
        // Attribute merging can attach a user attribute to an earlier Decl;
        // authenticate its own source location, not just the owning Decl.
        const auto location = attribute->getLocation();
        if (location.isInvalid() ||
            !declarationIsTrusted(*redeclaration, manager, state_) ||
            !isTrustedFile(manager.getFileEntryForID(
                manager.getFileID(manager.getSpellingLoc(location))), state_) ||
            !isTrustedFile(manager.getFileEntryForID(
                manager.getFileID(manager.getExpansionLoc(location))), state_))
          return false;
      }
    }
    return true;
  }

  void snapshotRegionDependencies(clang::ASTContext &context) {
    const auto &manager = context.getSourceManager();
    std::map<std::string, std::string> snapshots;
    for (auto iterator = manager.fileinfo_begin(); iterator != manager.fileinfo_end();
         ++iterator) {
      const auto file_id = manager.translateFile(iterator->first);
      if (file_id.isInvalid()) continue; // Lookup-only, never parsed.
      bool invalid = false;
      const auto parsed = manager.getBufferData(file_id, &invalid);
      const auto entry = manager.getFileEntryRefForID(file_id);
      if (invalid || !entry) {
        reject(manager, {}, "cannot authenticate a parsed region dependency");
        continue;
      }
      const std::string path = canonicalPath(entry->getName().str());
      const auto physical = readFile(path);
      llvm::sys::fs::UniqueID identity;
      const bool main_file = file_id == manager.getMainFileID();
      if (!physical || llvm::StringRef(*physical) != parsed ||
          llvm::sys::fs::getUniqueID(path, identity) ||
          (!main_file && identityOf(identity) != identityOf(entry->getUniqueID()))) {
        reject(manager, {}, "parsed region dependency changed or is not a stable "
                            "physical file: " + path);
        continue;
      }
      snapshots.insert_or_assign(path, snapshotHash(parsed));
      state_.region_dependency_ids.insert_or_assign(path, identityOf(identity));
    }
    for (const auto &[path, digest] : snapshots)
      state_.result.region_dependencies.push_back({path, digest});
  }

  void inspectTwoGemmRegions(clang::ASTContext &context) {
    if (!state_.inspect_two_gemm_regions) return;
    snapshotRegionDependencies(context);
    const auto &operations = state_.result.module.operations;
    // Greedy nonoverlapping admitted pairs. Rejected neighbouring candidates
    // remain diagnostics; a later independent pair can still be admitted.
    for (std::size_t index = 0; index < state_.captured_calls.size();) {
      TwoGemmRegionCandidateV1 candidate;
      candidate.source_snapshot_sha256 = snapshotHash(state_.source);
      const auto *first = state_.captured_calls[index];
      const auto *function = enclosingFunction(*first, context);
      if (function != nullptr)
        candidate.function_identity = declarationIdentity(*function, context);
      if (candidate.function_identity.empty())
        candidate.rejection_reasons.push_back(
            "region requires a direct nonmacro function declaration identity");
      candidate.source_range = operations[index].call_range;
      candidate.region_id = makeStableSiteId(
          stableSourceIdentity(state_.canonical_input), state_.compilation_identity,
          state_.source, operations[index].source.offset, "two-gemm-region");
      const std::size_t count = std::min<std::size_t>(2, state_.captured_calls.size() - index);
      for (std::size_t stage = 0; stage < count; ++stage) {
        const auto *call = state_.captured_calls[index + stage];
        const auto *out = llvm::cast<clang::CallExpr>(call->getArg(0)->IgnoreParenImpCasts());
        TwoGemmRegionSiteV1 site;
        site.site_id = operations[index + stage].site_id;
        site.capture_ordinal = index + stage;
        const clang::Expr *arguments[] = {out->getArg(0), call->getArg(1), call->getArg(2)};
        for (unsigned operand = 0; operand < 3; ++operand) {
          if (!descriptorBinding(*arguments[operand], static_cast<unsigned>(stage),
                                 context, site.bindings[operand]))
            candidate.rejection_reasons.push_back(
                "stage " + std::to_string(stage) + " operand " + std::to_string(operand) +
                " requires a direct descriptor declaration or transparent local reference");
        }
        if (!regionDeclarationUncontaminated(*call, context))
          candidate.rejection_reasons.push_back(
              "region admission rejects competing GEMM definitions or user redeclaration attributes");
        candidate.sites.push_back(std::move(site));
        candidate.source_range.end = operations[index + stage].call_range.end;
      }
      if (count != 2) {
        candidate.rejection_reasons.push_back("no following GEMM call for a two-call region");
      } else {
        const auto *second = state_.captured_calls[index + 1];
        const auto *compound = directCompound(*first, context);
        if (function == nullptr || compound == nullptr ||
            compound != directCompound(*second, context) ||
            function != enclosingFunction(*second, context) ||
            !supportedCompound(*compound, *function, context)) {
          candidate.rejection_reasons.push_back(
              "two GEMMs must be direct statements in the same function-body or direct try-body compound; control and scope are barriers");
        } else {
          bool consecutive = false;
          for (auto current = compound->body_begin(); current != compound->body_end(); ++current) {
            auto next = current;
            ++next;
            if (*current == first && next != compound->body_end() && *next == second)
              consecutive = true;
          }
          if (!consecutive) candidate.rejection_reasons.push_back(
              "intervening host statement, descriptor mutation or observer splits the region");
        }
        const auto &a = candidate.sites[0].bindings;
        const auto &b = candidate.sites[1].bindings;
        if (a[0].descriptor_id.empty() || a[0].descriptor_id != b[1].descriptor_id)
          candidate.rejection_reasons.push_back(
              "bounded region requires first output to be the second GEMM lhs descriptor");
        if (!a[0].descriptor_id.empty() && a[0].descriptor_id == b[0].descriptor_id)
          candidate.rejection_reasons.push_back("the two outputs require distinct descriptor bindings (not a noalias proof)");
        for (const auto &site : candidate.sites)
          if (!site.bindings[0].descriptor_id.empty() &&
              (site.bindings[0].descriptor_id == site.bindings[1].descriptor_id ||
               site.bindings[0].descriptor_id == site.bindings[2].descriptor_id))
            candidate.rejection_reasons.push_back("an output is proven to use its own input descriptor");
      }
      candidate.admitted = candidate.rejection_reasons.empty();
      index += candidate.admitted ? 2 : 1;
      state_.result.two_gemm_regions.push_back(std::move(candidate));
    }
  }

  const clang::FunctionDecl *
  enclosingFunction(const clang::Stmt &statement, clang::ASTContext &context,
                    unsigned depth = 0) {
    if (depth > 64) {
      return nullptr;
    }
    for (const clang::DynTypedNode &parent :
         context.getParents(clang::DynTypedNode::create(statement))) {
      if (const auto *function = parent.get<clang::FunctionDecl>()) {
        return function;
      }
      if (const auto *parent_statement = parent.get<clang::Stmt>()) {
        if (const clang::FunctionDecl *function =
                enclosingFunction(*parent_statement, context, depth + 1)) {
          return function;
        }
      }
      if (const auto *declaration = parent.get<clang::Decl>()) {
        for (const clang::DynTypedNode &grandparent :
             context.getParents(clang::DynTypedNode::create(*declaration))) {
          if (const auto *function = grandparent.get<clang::FunctionDecl>()) {
            return function;
          }
        }
      }
    }
    return nullptr;
  }

  void inspectRecoveredLoops(clang::ASTContext &context) {
    if (!state_.inspect_recovered_cpp_gemm) {
      return;
    }
    const clang::SourceManager &source_manager = context.getSourceManager();
    std::sort(state_.recovered_outer_loops.begin(),
              state_.recovered_outer_loops.end(),
              [&source_manager](const clang::ForStmt *left,
                                const clang::ForStmt *right) {
                return source_manager.isBeforeInTranslationUnit(
                    left->getBeginLoc(), right->getBeginLoc());
              });
    state_.recovered_outer_loops.erase(
        std::unique(state_.recovered_outer_loops.begin(),
                    state_.recovered_outer_loops.end()),
        state_.recovered_outer_loops.end());
    const RecoveredGemmInspectionInputs inputs{
        .main_display_path = state_.display_path,
        .main_canonical_path = state_.canonical_input,
        .compilation_identity = state_.compilation_identity,
        .optimization_level = state_.optimization_level,
        .denormal_mode = state_.denormal_mode,
        .fp32_denormal_mode = state_.fp32_denormal_mode,
        .denormal_mode_is_ieee = state_.denormal_mode_is_ieee,
        .fp32_denormal_mode_is_ieee = state_.fp32_denormal_mode_is_ieee,
    };
    for (const clang::ForStmt *outer : state_.recovered_outer_loops) {
      const clang::FunctionDecl *function = enclosingFunction(*outer, context);
      if (function == nullptr) {
        continue;
      }
      state_.result.recovered_gemm_candidates.push_back(
          inspectRecoveredGemmLoop(*outer, *function, context, inputs));
    }
    std::sort(state_.result.recovered_gemm_candidates.begin(),
              state_.result.recovered_gemm_candidates.end(),
              [](const RecoveredGemmCandidate &left,
                 const RecoveredGemmCandidate &right) {
                if (left.source_identity != right.source_identity) {
                  return left.source_identity < right.source_identity;
                }
                if (left.offset != right.offset) {
                  return left.offset < right.offset;
                }
                return left.site_id < right.site_id;
              });
    state_.result.recovered_gemm_candidates.erase(
        std::unique(state_.result.recovered_gemm_candidates.begin(),
                    state_.result.recovered_gemm_candidates.end(),
                    [](const RecoveredGemmCandidate &left,
                       const RecoveredGemmCandidate &right) {
                      return left.site_id == right.site_id;
                    }),
        state_.result.recovered_gemm_candidates.end());
  }

  void reject(const clang::SourceManager &source_manager,
              clang::SourceLocation location, std::string message) {
    state_.result.diagnostics.push_back(diagnosticAt(
        source_manager, location, state_, std::move(message)));
  }

  bool authenticate(const clang::FunctionDecl &declaration,
                    std::string_view expected_name,
                    std::string_view expected_annotation,
                    const clang::SourceManager &source_manager,
                    clang::SourceLocation call_location) {
    std::string annotation_error;
    switch (authenticateDeclaration(declaration, expected_name,
                                    expected_annotation, source_manager, state_,
                                    annotation_error)) {
    case Authentication::valid:
      return true;
    case Authentication::untrusted:
      reject(source_manager, call_location,
             "Matcore operations must resolve to the canonical declaration in "
             "the compiler's trusted <matcore/mdsl.h> header");
      return false;
    case Authentication::wrong_signature:
      reject(source_manager, call_location,
             "Matcore declaration has an unsupported canonical signature");
      return false;
    case Authentication::annotation:
      reject(source_manager, call_location,
             "Matcore declaration annotation authentication failed: " +
                 annotation_error + "; expected '" +
                 std::string(expected_annotation) + "'");
      return false;
    }
    return false;
  }

  void processCall(const clang::CallExpr &call, clang::ASTContext &context,
                   std::unordered_set<const clang::DeclRefExpr *> &allowed) {
    const clang::FunctionDecl *direct = call.getDirectCallee();
    if (direct == nullptr ||
        (direct->getCanonicalDecl()->getQualifiedNameAsString() != kGemmName &&
         !hasReservedOperationAnnotation(*direct))) {
      return;
    }
    state_.saw_candidate = true;
    if (state_.first_candidate_location.isInvalid()) {
      state_.first_candidate_location = call.getExprLoc();
    }
    const clang::DeclRefExpr *callee_reference = directCalleeReference(call);
    if (callee_reference != nullptr) {
      allowed.insert(callee_reference);
    }
    const clang::SourceManager &source_manager = context.getSourceManager();
    bool valid = true;
    auto fail = [&](std::string message) {
      reject(source_manager, call.getExprLoc(), std::move(message));
      valid = false;
    };

    if (!authenticate(*direct, kGemmName, kGemmAnnotation, source_manager,
                      call.getExprLoc())) {
      return;
    }
    if (!state_.trusted_header_external_macros.empty()) {
      fail("the trusted <matcore/mdsl.h> ABI was altered by macro expansion '" +
           *state_.trusted_header_external_macros.begin() +
           "'; source macros may not rewrite public Matcore declarations");
      return;
    }
    std::string public_abi_error;
    if (!validatePublicAbi(*direct, context, state_, public_abi_error)) {
      fail("the parsed <matcore/mdsl.h> ABI is incompatible with runtime v0: " +
           public_abi_error);
      return;
    }
    if (callee_reference == nullptr) {
      fail("indirect or function-pointer calls to Matcore operations are not "
           "supported");
    } else if (!hasCanonicalMdslQualifier(*callee_reference)) {
      fail("Matcore operations must be directly qualified through "
           "matcore::mdsl or a namespace alias to it; unqualified, ADL, and "
           "using-declaration re-exports are rejected");
    }
    if (hasMacroOrigin(call)) {
      fail("Matcore calls generated by macros or unsafe source ranges are not "
           "supported in native v1");
    } else if (!source_manager.isWrittenInMainFile(
                   source_manager.getSpellingLoc(call.getBeginLoc()))) {
      fail("Matcore call sites must be written directly in the input .mdsl "
           "file");
    }
    switch (forbiddenContext(clang::DynTypedNode::create(call), context)) {
    case ContextViolation::template_context:
      fail("Matcore calls inside templates are not supported in native v1");
      return;
    case ContextViolation::lambda:
      fail("Matcore calls inside lambdas are not supported in native v1");
      return;
    case ContextViolation::constexpr_context:
      fail("constexpr Matcore execution is not supported in native v1");
      return;
    case ContextViolation::unevaluated_context:
      fail("Matcore calls in unevaluated contexts are not supported in native "
           "v1");
      return;
    case ContextViolation::none:
      break;
    }
    if (call.getNumArgs() != 4) {
      fail("native gemm requires explicit output, lhs, rhs, and policy "
           "arguments after Sema");
      return;
    }

    std::string output_name;
    const clang::Expr *output_expression = call.getArg(0)->IgnoreParenImpCasts();
    const auto *output_call = llvm::dyn_cast<clang::CallExpr>(output_expression);
    if (output_call == nullptr || output_call->getDirectCallee() == nullptr ||
        output_call->getDirectCallee()
                ->getCanonicalDecl()
                ->getQualifiedNameAsString() != kOutName) {
      fail("gemm output must use the canonical matcore::mdsl::out wrapper");
    } else {
      const clang::DeclRefExpr *out_reference =
          directCalleeReference(*output_call);
      if (!authenticate(*output_call->getDirectCallee(), kOutName,
                        kOutAnnotation, source_manager,
                        output_call->getExprLoc())) {
        valid = false;
      } else if (out_reference == nullptr ||
                 !hasCanonicalMdslQualifier(*out_reference)) {
        fail("the out wrapper must be directly qualified through "
             "matcore::mdsl or a namespace alias to it");
      }
      if (output_call->getNumArgs() != 1 ||
          !simpleLvalue(*output_call->getArg(0), output_name)) {
        fail("gemm output must be a stable mutable lvalue with no side effects");
      }
    }

    std::string lhs_name;
    std::string rhs_name;
    if (!simpleLvalue(*call.getArg(1), lhs_name)) {
      fail("gemm lhs must be a stable matrix lvalue with no side effects");
    }
    if (!simpleLvalue(*call.getArg(2), rhs_name)) {
      fail("gemm rhs must be a stable matrix lvalue with no side effects");
    }
    // Region inspection retains the actual qualified expression spelling for
    // the compatibility report; declaration identity, not an unqualified name,
    // supplies its alias facts. Leave ordinary extraction byte-for-byte alone.
    if (state_.inspect_two_gemm_regions) {
      auto retain_spelling = [&](const clang::Expr &expression, std::string &name) {
        const auto range = mainFileTokenRange(expression, source_manager,
                                              context.getLangOpts());
        if (range && range->end <= state_.source.size())
          name = state_.source.substr(range->begin, range->end - range->begin);
      };
      if (output_call != nullptr && output_call->getNumArgs() == 1)
        retain_spelling(*output_call->getArg(0), output_name);
      retain_spelling(*call.getArg(1), lhs_name);
      retain_spelling(*call.getArg(2), rhs_name);
    }
    if (!output_name.empty() &&
        (output_name == lhs_name || output_name == rhs_name)) {
      fail("gemm output must not alias lhs or rhs");
    }

    std::string target;
    std::string fallback;
    if (call.getArg(3)->HasSideEffects(context)) {
      fail("gemm policy must not contain function calls or other side effects");
    } else if (!parsePolicy(*call.getArg(3), target, fallback)) {
      fail("gemm policy must be the default or an inline policy aggregate so "
           "target and fallback are compile-time visible");
    } else if (target != "cpu" || fallback != "error") {
      fail("native v1 supports only target=cpu with fallback=error; no silent "
           "fallback is permitted");
    }

    const auto call_range = mainFileTokenRange(
        call, source_manager, context.getLangOpts());
    if (!call_range) {
      fail("Matcore call does not have a safe main-file SourceManager range");
      return;
    }
    std::vector<ir::SourceRange> argument_ranges;
    for (unsigned index = 0; index != 3; ++index) {
      const auto range = mainFileTokenRange(
          *call.getArg(index), source_manager, context.getLangOpts());
      if (!range || range->begin < call_range->begin ||
          range->end > call_range->end) {
        fail("gemm argument does not have a safe main-file source range");
        break;
      }
      argument_ranges.push_back(*range);
    }
    if (!llvm::isa<clang::CXXDefaultArgExpr>(
            call.getArg(3)->IgnoreParenImpCasts())) {
      const auto range = mainFileTokenRange(
          *call.getArg(3), source_manager, context.getLangOpts());
      if (!range || range->begin < call_range->begin ||
          range->end > call_range->end) {
        fail("explicit gemm policy does not have a safe main-file source "
             "range");
      } else {
        argument_ranges.push_back(*range);
      }
    }
    if (!valid) {
      return;
    }

    const clang::SourceLocation begin =
        source_manager.getSpellingLoc(call.getBeginLoc());
    const std::uint64_t offset = source_manager.getFileOffset(begin);
    ir::Operation operation;
    operation.site_id = makeStableSiteId(
        stableSourceIdentity(state_.canonical_input),
        state_.compilation_identity, state_.source, offset, "gemm");
    operation.kind = "gemm";
    operation.canonical_callee = std::string(kGemmName);
    operation.source = ir::SourceLocation{
        .file = state_.display_path,
        .offset = offset,
        .line = source_manager.getSpellingLineNumber(begin),
        .column = source_manager.getSpellingColumnNumber(begin),
    };
    operation.call_range = *call_range;
    operation.argument_ranges = std::move(argument_ranges);
    operation.output = ir::MatrixValue{
        .role = "output", .expression = output_name, .mutability = "write"};
    operation.operands = {
        ir::MatrixValue{.role = "lhs",
                        .expression = lhs_name,
                        .mutability = "read"},
        ir::MatrixValue{.role = "rhs",
                        .expression = rhs_name,
                        .mutability = "read"},
    };
    operation.target = target;
    operation.fallback = fallback;
    state_.result.module.operations.push_back(std::move(operation));
    state_.captured_calls.push_back(&call);
  }

  void finalize(clang::ASTContext &context) {
    std::sort(state_.calls.begin(), state_.calls.end(),
              [&context](const clang::CallExpr *left,
                         const clang::CallExpr *right) {
                return context.getSourceManager().isBeforeInTranslationUnit(
                    left->getBeginLoc(), right->getBeginLoc());
              });
    std::unordered_set<const clang::DeclRefExpr *> allowed_references;
    for (const clang::CallExpr *call : state_.calls) {
      processCall(*call, context, allowed_references);
    }

    const clang::SourceManager &source_manager = context.getSourceManager();
    for (const clang::DeclRefExpr *reference : state_.function_references) {
      const auto *function =
          llvm::dyn_cast<clang::FunctionDecl>(reference->getDecl());
      if (function == nullptr ||
          (function->getCanonicalDecl()->getQualifiedNameAsString() !=
               kGemmName &&
           !hasReservedOperationAnnotation(*function)) ||
          allowed_references.contains(reference)) {
        continue;
      }
      state_.saw_candidate = true;
      if (state_.first_candidate_location.isInvalid()) {
        state_.first_candidate_location = reference->getExprLoc();
      }
      reject(source_manager, reference->getExprLoc(),
             "indirect or function-pointer references to Matcore operations "
             "are not supported");
    }

    inspectRecoveredLoops(context);
    inspectTwoGemmRegions(context);

    if (state_.saw_candidate && !state_.saw_direct_trusted_include) {
      reject(source_manager, state_.first_candidate_location,
             state_.saw_direct_expected_include
                 ? "the main .mdsl <matcore/mdsl.h> include did not resolve "
                   "to the compiler's trusted public header"
                 : "the main .mdsl file must directly include the compiler's "
                   "trusted <matcore/mdsl.h> public header");
    }

    std::sort(state_.result.module.operations.begin(),
              state_.result.module.operations.end(),
              [](const ir::Operation &left, const ir::Operation &right) {
                return left.source.offset < right.source.offset;
              });
    std::sort(state_.result.diagnostics.begin(),
              state_.result.diagnostics.end(),
              [](const Diagnostic &left, const Diagnostic &right) {
                if (left.file != right.file) {
                  return left.file < right.file;
                }
                if (left.line != right.line) {
                  return left.line < right.line;
                }
                if (left.column != right.column) {
                  return left.column < right.column;
                }
                return left.message < right.message;
              });
    state_.result.diagnostics.erase(
        std::unique(state_.result.diagnostics.begin(),
                    state_.result.diagnostics.end(),
                    [](const Diagnostic &left, const Diagnostic &right) {
                      return left.file == right.file &&
                             left.line == right.line &&
                             left.column == right.column &&
                             left.message == right.message;
                    }),
        state_.result.diagnostics.end());
  }

  NativeState &state_;
  MatchCollector collector_;
  clang::ast_matchers::MatchFinder finder_;
};

class NativeAction final : public clang::ASTFrontendAction {
public:
  explicit NativeAction(NativeState &state) : state_(state) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &compiler,
                    llvm::StringRef) override {
    const clang::CodeGenOptions &codegen = compiler.getCodeGenOpts();
    state_.optimization_level = codegen.OptimizationLevel;
    state_.denormal_mode = codegen.FPDenormalMode.str();
    state_.fp32_denormal_mode = codegen.FP32DenormalMode.str();
    state_.denormal_mode_is_ieee =
        codegen.FPDenormalMode == llvm::DenormalMode::getIEEE();
    state_.fp32_denormal_mode_is_ieee =
        codegen.FP32DenormalMode == llvm::DenormalMode::getIEEE();
    compiler.getPreprocessor().addPPCallbacks(
        std::make_unique<TrustedHeaderCallbacks>(compiler.getSourceManager(),
                                                 state_));
    return std::make_unique<NativeConsumer>(state_);
  }

private:
  NativeState &state_;
};

class NativeActionFactory final : public clang::tooling::FrontendActionFactory {
public:
  explicit NativeActionFactory(NativeState &state) : state_(state) {}

  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<NativeAction>(state_);
  }

private:
  NativeState &state_;
};

class ClangLibToolingFrontend final : public Frontend {
public:
  bool extract(const Options &options, Result &result) override {
    result = Result{};
    const std::string display_path = normalizeDisplayPath(options.input_path);
    result.module.translation_unit = display_path;
    result.module.source_file = display_path;
    result.module.producer = "clang-libtooling-v1";
    if (!std::string_view(display_path).ends_with(".mdsl")) {
      result.diagnostics.push_back(
          Diagnostic{.file = display_path,
                     .message = "input source must use the .mdsl extension"});
      return false;
    }
    const std::optional<std::string> source = readFile(options.input_path);
    if (!source) {
      result.diagnostics.push_back(
          Diagnostic{.file = display_path,
                     .message = "unable to open input .mdsl source"});
      return false;
    }
    result.source_snapshot = *source;

    NativeState state(result, display_path, canonicalPath(options.input_path),
                      stableCompilationIdentity(options), *source);
    state.inspect_recovered_cpp_gemm = options.inspect_recovered_cpp_gemm;
    state.inspect_two_gemm_regions = options.inspect_two_gemm_regions;
    for (const std::string &trusted_header : options.trusted_public_headers) {
      llvm::sys::fs::UniqueID identity;
      const std::optional<std::string> contents = readFile(trusted_header);
      if (!trusted_header.empty() && contents &&
          !llvm::sys::fs::getUniqueID(trusted_header, identity)) {
        const FileIdentity file_identity = identityOf(identity);
        state.trusted_header_ids.insert(file_identity);
        state.trusted_header_contents.insert_or_assign(file_identity,
                                                       *contents);
      }
    }
    if (state.trusted_header_ids.empty()) {
      result.diagnostics.push_back(
          Diagnostic{.file = display_path,
                     .message =
                         "compiler has no trusted <matcore/mdsl.h> identity"});
      return false;
    }

    std::vector<std::string> tool_arguments;
    std::string error;
    if (!makeToolArguments(options, tool_arguments, error)) {
      result.diagnostics.push_back(
          Diagnostic{.file = display_path, .message = std::move(error)});
      return false;
    }
    if (options.verbose) {
      std::cerr << "matcore-extract native Clang arguments:";
      for (const std::string &argument : tool_arguments) {
        std::cerr << ' ' << shellQuoted(argument);
      }
      std::cerr << ' ' << shellQuoted(options.input_path) << '\n';
    }

    std::string current_path_error;
    const std::optional<std::string> current_path =
        support::path_to_utf8_v1(std::filesystem::current_path(),
                                 current_path_error);
    if (!current_path) {
      result.diagnostics.push_back(
          Diagnostic{.file = display_path,
                     .message = "cannot encode frontend working directory as "
                                "UTF-8: " + current_path_error});
      return false;
    }
    clang::tooling::FixedCompilationDatabase compilations(*current_path,
                                                           tool_arguments);
    clang::tooling::ClangTool tool(compilations, {options.input_path});
    tool.mapVirtualFile(options.input_path, *source);
    CountingDiagnosticConsumer clang_diagnostics;
    tool.setDiagnosticConsumer(&clang_diagnostics);
    NativeActionFactory factory(state);
    const int tool_status = tool.run(&factory);

    if (options.inspect_two_gemm_regions) {
      result.region_native_clang_version = nativeClangRuntimeVersionV1();
      std::string context_identity = "mdsl-two-gemm-capture-context-v1";
      bindIdentityField(context_identity, result.region_native_clang_version);
      bindIdentityField(context_identity, options.clang_path);
      bindIdentityField(context_identity, options.clang_resource_directory);
      bindIdentityField(context_identity, state.canonical_input);
      bindIdentityField(context_identity, snapshotHash(*source));
      for (const auto &argument : tool_arguments)
        bindIdentityField(context_identity, argument);
      for (const auto &dependency : result.region_dependencies) {
        llvm::sys::fs::UniqueID identity;
        const auto physical = readFile(dependency.path);
        if (!physical || snapshotHash(*physical) != dependency.sha256 ||
            llvm::sys::fs::getUniqueID(dependency.path, identity) ||
            identityOf(identity) != state.region_dependency_ids.at(dependency.path)) {
          result.diagnostics.push_back(Diagnostic{
              .file = dependency.path,
              .message = "region dependency changed after native parsing; retry from a stable snapshot"});
        }
        bindIdentityField(context_identity, dependency.path);
        bindIdentityField(context_identity, dependency.sha256);
      }
      result.region_capture_identity = snapshotHash(context_identity);
    }

    const std::optional<std::string> verified_source = readFile(options.input_path);
    if (!verified_source || *verified_source != *source) {
      result.diagnostics.push_back(Diagnostic{
          .file = display_path,
          .message = "input .mdsl changed while native Clang parsed it; retry "
                     "from a stable source snapshot"});
    }
    if (tool_status != 0) {
      result.diagnostics.push_back(
          Diagnostic{.file = display_path,
                     .message = "Clang parsing/Sema failed with exit code " +
                                std::to_string(tool_status)});
    } else if (clang_diagnostics.getNumErrors() != 0) {
      result.diagnostics.push_back(Diagnostic{
          .file = display_path,
          .message = "Clang parsing/Sema emitted " +
                     std::to_string(clang_diagnostics.getNumErrors()) +
                     " error diagnostic(s)"});
    }
    if (!result.diagnostics.empty()) {
      result.module.operations.clear();
      return false;
    }
    if (!ir::verify(result.module, error)) {
      result.diagnostics.push_back(
          Diagnostic{.file = display_path,
                     .message = "Matcore IR verifier rejected extraction: " +
                                error});
      result.module.operations.clear();
      return false;
    }
    if (options.inspect_recovered_cpp_gemm || options.inspect_two_gemm_regions) {
      result.native_evidence =
          detail::NativeFrontendEvidenceIssuerV1::issue(result, options);
    }
    return true;
  }
};

} // namespace

std::unique_ptr<Frontend> createClangLibToolingFrontend() {
  return std::make_unique<ClangLibToolingFrontend>();
}

std::string nativeClangRuntimeVersionV1() {
  return clang::getClangFullVersion();
}

std::optional<std::string> nativeClangRuntimeLibraryPathV1(std::string &error) {
  error.clear();
#if defined(__linux__)
  struct SearchState {
    std::optional<std::string> path;
    bool ambiguous = false;
  } state;
  const auto inspect = [](dl_phdr_info *info, std::size_t, void *opaque) {
    auto &search = *static_cast<SearchState *>(opaque);
    if (info == nullptr || info->dlpi_name == nullptr ||
        info->dlpi_name[0] == '\0') {
      return 0;
    }
    const std::filesystem::path candidate(info->dlpi_name);
    const std::string filename = candidate.filename().string();
    if (!std::string_view(filename).starts_with("libclang-cpp.so"))
      return 0;
    if (search.path && *search.path != info->dlpi_name) {
      search.ambiguous = true;
      return 1;
    }
    search.path = info->dlpi_name;
    return 0;
  };
  dl_iterate_phdr(inspect, &state);
  if (state.ambiguous) {
    error = "multiple clang-cpp shared runtimes are loaded";
    return std::nullopt;
  }
  if (!state.path) {
    error = "loaded clang-cpp shared runtime path is unavailable";
    return std::nullopt;
  }
  return state.path;
#else
  error = "loaded clang-cpp shared runtime path is unsupported on this host";
  return std::nullopt;
#endif
}

} // namespace matcore::mdslc::frontend
