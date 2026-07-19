#include "frontend.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace matcore::mdslc::frontend {
namespace {

using JsonValue = rapidjson::Value;

enum class DeclarationRole {
  none,
  gemm,
  out,
};

struct AstLocation {
  std::string file;
  std::uint64_t offset = 0;
  unsigned line = 0;
  unsigned column = 0;
  bool has_offset = false;
  bool macro = false;
};

struct Declaration {
  std::string id;
  std::string previous_id;
  std::string qualified_name;
  std::string source_file;
  std::vector<std::string> parameter_types;
  bool annotated = false;
  DeclarationRole role = DeclarationRole::none;
};

struct DirectCallee {
  std::string declaration_id;
  const JsonValue *reference = nullptr;
};

struct WalkContext {
  std::string source_file;
  bool in_lambda = false;
  bool in_template = false;
  bool in_constexpr = false;
  bool in_constant_evaluation = false;
};

std::string memberString(const JsonValue &value, const char *key) {
  if (!value.IsObject()) {
    return {};
  }
  const auto iterator = value.FindMember(key);
  if (iterator == value.MemberEnd() || !iterator->value.IsString()) {
    return {};
  }
  return std::string(iterator->value.GetString(), iterator->value.GetStringLength());
}

bool memberBool(const JsonValue &value, const char *key) {
  if (!value.IsObject()) {
    return false;
  }
  const auto iterator = value.FindMember(key);
  return iterator != value.MemberEnd() && iterator->value.IsBool() &&
         iterator->value.GetBool();
}

const JsonValue *memberObject(const JsonValue &value, const char *key) {
  if (!value.IsObject()) {
    return nullptr;
  }
  const auto iterator = value.FindMember(key);
  if (iterator == value.MemberEnd() || !iterator->value.IsObject()) {
    return nullptr;
  }
  return &iterator->value;
}

const JsonValue *innerArray(const JsonValue &value) {
  if (!value.IsObject()) {
    return nullptr;
  }
  const auto iterator = value.FindMember("inner");
  if (iterator == value.MemberEnd() || !iterator->value.IsArray()) {
    return nullptr;
  }
  return &iterator->value;
}

std::string typeName(const JsonValue &value) {
  const JsonValue *type = memberObject(value, "type");
  if (type == nullptr) {
    return {};
  }
  std::string result = memberString(*type, "desugaredQualType");
  if (result.empty()) {
    result = memberString(*type, "qualType");
  }
  // Clang's JSON sometimes omits the enclosing namespace from non-dependent
  // parameter spellings even though the declaration is semantically resolved.
  if (result == "matrix_view &") {
    return "matcore::mdsl::matrix_view &";
  }
  if (result == "const matrix_view &") {
    return "const matcore::mdsl::matrix_view &";
  }
  if (result == "out_arg") {
    return "matcore::mdsl::out_arg";
  }
  if (result == "policy") {
    return "matcore::mdsl::policy";
  }
  return result;
}

AstLocation decodeLocation(const JsonValue &encoded,
                           const std::string &fallback_file) {
  AstLocation result;
  result.file = fallback_file;
  if (!encoded.IsObject()) {
    return result;
  }

  const JsonValue *location = &encoded;
  if (const JsonValue *expansion = memberObject(encoded, "expansionLoc")) {
    result.macro = true;
    location = expansion;
  } else if (const JsonValue *spelling = memberObject(encoded, "spellingLoc")) {
    result.macro = true;
    location = spelling;
  }

  const std::string file = memberString(*location, "file");
  if (!file.empty()) {
    result.file = file;
  }
  if (location->HasMember("offset") && (*location)["offset"].IsUint64()) {
    result.offset = (*location)["offset"].GetUint64();
    result.has_offset = true;
  }
  if (location->HasMember("line") && (*location)["line"].IsUint()) {
    result.line = (*location)["line"].GetUint();
  }
  if (location->HasMember("col") && (*location)["col"].IsUint()) {
    result.column = (*location)["col"].GetUint();
  }
  return result;
}

AstLocation nodeLocation(const JsonValue &node,
                         const std::string &fallback_file) {
  if (const JsonValue *location = memberObject(node, "loc")) {
    AstLocation decoded = decodeLocation(*location, fallback_file);
    if (decoded.has_offset || !memberString(*location, "file").empty()) {
      return decoded;
    }
  }
  if (const JsonValue *range = memberObject(node, "range")) {
    if (const JsonValue *begin = memberObject(*range, "begin")) {
      return decodeLocation(*begin, fallback_file);
    }
  }
  return AstLocation{.file = fallback_file};
}

std::optional<std::pair<std::uint64_t, std::uint64_t>>
sourceRange(const JsonValue &node) {
  const JsonValue *range = memberObject(node, "range");
  const JsonValue *begin = range == nullptr ? nullptr : memberObject(*range, "begin");
  const JsonValue *end = range == nullptr ? nullptr : memberObject(*range, "end");
  if (begin == nullptr || end == nullptr) {
    return std::nullopt;
  }
  if (memberObject(*begin, "spellingLoc") != nullptr ||
      memberObject(*begin, "expansionLoc") != nullptr ||
      memberObject(*end, "spellingLoc") != nullptr ||
      memberObject(*end, "expansionLoc") != nullptr) {
    return std::nullopt;
  }
  if (!begin->HasMember("offset") || !(*begin)["offset"].IsUint64() ||
      !end->HasMember("offset") || !(*end)["offset"].IsUint64()) {
    return std::nullopt;
  }
  std::uint64_t first = (*begin)["offset"].GetUint64();
  std::uint64_t last = (*end)["offset"].GetUint64();
  std::uint64_t token_length = 1;
  if (end->HasMember("tokLen") && (*end)["tokLen"].IsUint64()) {
    token_length = (*end)["tokLen"].GetUint64();
  }
  if (last > UINT64_MAX - token_length) {
    return std::nullopt;
  }
  return std::pair{first, last + token_length};
}

std::string normalizeDisplayPath(const std::string &path) {
  return std::filesystem::path(path).lexically_normal().generic_string();
}

std::string canonicalPath(const std::string &path) {
  std::error_code error;
  const std::filesystem::path canonical =
      std::filesystem::weakly_canonical(std::filesystem::path(path), error);
  if (error) {
    return normalizeDisplayPath(path);
  }
  return canonical.generic_string();
}

bool isPublicHeader(const std::string &path) {
  const std::filesystem::path source(path);
  return source.filename() == "mdsl.h" &&
         source.parent_path().filename() == "matcore";
}

std::string qualifiedName(const std::vector<std::string> &namespaces,
                          const std::string &name) {
  std::string result;
  for (const std::string &component : namespaces) {
    if (!result.empty()) {
      result += "::";
    }
    result += component;
  }
  if (!result.empty()) {
    result += "::";
  }
  result += name;
  return result;
}

bool hasDirectAnnotation(const JsonValue &declaration) {
  const JsonValue *inner = innerArray(declaration);
  if (inner == nullptr) {
    return false;
  }
  for (const JsonValue &child : inner->GetArray()) {
    if (memberString(child, "kind") == "AnnotateAttr") {
      return true;
    }
  }
  return false;
}

DeclarationRole classifyDeclaration(const Declaration &declaration) {
  if (!declaration.annotated || !isPublicHeader(declaration.source_file)) {
    return DeclarationRole::none;
  }
  if (declaration.qualified_name == "matcore::mdsl::gemm" &&
      declaration.parameter_types ==
          std::vector<std::string>{"matcore::mdsl::out_arg",
                                   "const matcore::mdsl::matrix_view &",
                                   "const matcore::mdsl::matrix_view &",
                                   "matcore::mdsl::policy"}) {
    return DeclarationRole::gemm;
  }
  if (declaration.qualified_name == "matcore::mdsl::out" &&
      declaration.parameter_types ==
          std::vector<std::string>{"matcore::mdsl::matrix_view &"}) {
    return DeclarationRole::out;
  }
  return DeclarationRole::none;
}

void collectDeclarations(const JsonValue &node,
                         std::vector<std::string> namespaces,
                         std::string &last_source_file,
                         std::unordered_map<std::string, Declaration> &result) {
  if (!node.IsObject()) {
    return;
  }
  const AstLocation location = nodeLocation(node, last_source_file);
  if (!location.file.empty()) {
    last_source_file = location.file;
  }
  const std::string kind = memberString(node, "kind");
  if (kind == "NamespaceDecl") {
    const std::string name = memberString(node, "name");
    if (!name.empty()) {
      namespaces.push_back(name);
    }
  }

  if (kind == "FunctionDecl" && node.HasMember("loc")) {
    Declaration declaration;
    declaration.id = memberString(node, "id");
    declaration.previous_id = memberString(node, "previousDecl");
    declaration.qualified_name =
        qualifiedName(namespaces, memberString(node, "name"));
    declaration.source_file = location.file;
    declaration.annotated = hasDirectAnnotation(node);
    if (const JsonValue *inner = innerArray(node)) {
      for (const JsonValue &child : inner->GetArray()) {
        if (memberString(child, "kind") == "ParmVarDecl") {
          declaration.parameter_types.push_back(typeName(child));
        }
      }
    }
    declaration.role = classifyDeclaration(declaration);
    if (!declaration.id.empty()) {
      result[declaration.id] = std::move(declaration);
    }
  }

  if (const JsonValue *inner = innerArray(node)) {
    for (const JsonValue &child : inner->GetArray()) {
      collectDeclarations(child, namespaces, last_source_file, result);
    }
  }
}

DeclarationRole roleForId(
    const std::string &id,
    const std::unordered_map<std::string, Declaration> &declarations) {
  std::unordered_set<std::string> visited;
  std::string current = id;
  while (!current.empty() && visited.insert(current).second) {
    const auto iterator = declarations.find(current);
    if (iterator == declarations.end()) {
      return DeclarationRole::none;
    }
    if (iterator->second.role != DeclarationRole::none) {
      return iterator->second.role;
    }
    current = iterator->second.previous_id;
  }
  return DeclarationRole::none;
}

std::optional<DirectCallee> directCallee(const JsonValue &call) {
  const JsonValue *inner = innerArray(call);
  if (inner == nullptr || inner->Empty()) {
    return std::nullopt;
  }
  const JsonValue *expression = &(*inner)[0];
  for (unsigned depth = 0; depth != 16; ++depth) {
    if (memberString(*expression, "kind") == "DeclRefExpr") {
      const JsonValue *referenced = memberObject(*expression, "referencedDecl");
      if (referenced != nullptr &&
          memberString(*referenced, "kind") == "FunctionDecl") {
        return DirectCallee{.declaration_id = memberString(*referenced, "id"),
                            .reference = expression};
      }
      return std::nullopt;
    }
    const JsonValue *children = innerArray(*expression);
    if (children == nullptr || children->Empty()) {
      return std::nullopt;
    }
    expression = &(*children)[0];
  }
  return std::nullopt;
}

bool isSimpleLvalue(const JsonValue &expression, std::string &declared_name) {
  const std::string kind = memberString(expression, "kind");
  if (kind == "DeclRefExpr") {
    const JsonValue *referenced = memberObject(expression, "referencedDecl");
    if (referenced == nullptr) {
      return false;
    }
    const std::string reference_kind = memberString(*referenced, "kind");
    if (reference_kind != "VarDecl" && reference_kind != "ParmVarDecl") {
      return false;
    }
    declared_name = memberString(*referenced, "name");
    return !declared_name.empty();
  }
  if (kind == "MemberExpr") {
    const JsonValue *inner = innerArray(expression);
    if (inner == nullptr || inner->Empty()) {
      return false;
    }
    std::string base_name;
    const std::string base_kind = memberString((*inner)[0], "kind");
    if (base_kind != "CXXThisExpr" &&
        !isSimpleLvalue((*inner)[0], base_name)) {
      return false;
    }
    declared_name = memberString(expression, "name");
    if (declared_name.empty()) {
      return false;
    }
    if (!base_name.empty()) {
      declared_name = base_name + "." + declared_name;
    }
    return true;
  }
  if (kind == "ImplicitCastExpr" || kind == "ParenExpr" ||
      kind == "ExprWithCleanups") {
    const JsonValue *inner = innerArray(expression);
    return inner != nullptr && inner->Size() == 1 &&
           isSimpleLvalue((*inner)[0], declared_name);
  }
  return false;
}

void collectEnumConstants(const JsonValue &node,
                          std::vector<std::pair<std::string, std::string>> &out) {
  if (memberString(node, "kind") == "DeclRefExpr") {
    const JsonValue *referenced = memberObject(node, "referencedDecl");
    if (referenced != nullptr &&
        memberString(*referenced, "kind") == "EnumConstantDecl") {
      out.emplace_back(typeName(*referenced), memberString(*referenced, "name"));
    }
  }
  if (const JsonValue *inner = innerArray(node)) {
    for (const JsonValue &child : inner->GetArray()) {
      collectEnumConstants(child, out);
    }
  }
}

bool parsePolicy(const JsonValue &expression, std::string &target,
                 std::string &fallback) {
  target = "cpu";
  fallback = "error";
  if (memberString(expression, "kind") == "CXXDefaultArgExpr") {
    return true;
  }
  if (memberString(expression, "kind") != "CXXFunctionalCastExpr" &&
      memberString(expression, "kind") != "InitListExpr" &&
      memberString(expression, "kind") != "CXXTemporaryObjectExpr") {
    return false;
  }
  std::vector<std::pair<std::string, std::string>> constants;
  collectEnumConstants(expression, constants);
  for (const auto &[type, value] : constants) {
    if (type == "matcore::mdsl::target") {
      target = value;
    } else if (type == "matcore::mdsl::fallback") {
      fallback = value;
    }
  }
  return true;
}

std::pair<unsigned, unsigned> lineAndColumn(std::string_view source,
                                            std::uint64_t offset) {
  unsigned line = 1;
  unsigned column = 1;
  const std::size_t limit =
      std::min<std::uint64_t>(offset, static_cast<std::uint64_t>(source.size()));
  for (std::size_t index = 0; index < limit; ++index) {
    if (source[index] == '\n') {
      ++line;
      column = 1;
    } else {
      ++column;
    }
  }
  return {line, column};
}

std::string sourceText(const JsonValue &node, std::string_view source) {
  const auto range = sourceRange(node);
  if (!range || range->first > range->second || range->second > source.size()) {
    return {};
  }
  return std::string(source.substr(static_cast<std::size_t>(range->first),
                                   static_cast<std::size_t>(range->second -
                                                            range->first)));
}

std::string siteId(std::string_view translation_unit, std::uint64_t offset,
                   std::string_view kind) {
  // Stable FNV-1a over a versioned, separator-delimited site identity.
  std::uint64_t hash = 14695981039346656037ULL;
  const auto append = [&hash](std::string_view value) {
    for (const unsigned char character : value) {
      hash ^= character;
      hash *= 1099511628211ULL;
    }
    hash ^= 0xffU;
    hash *= 1099511628211ULL;
  };
  append("matcore-site-v0");
  append(translation_unit);
  append(std::to_string(offset));
  append(kind);
  std::ostringstream formatted;
  formatted << "mc_" << std::hex << std::setfill('0') << std::setw(16) << hash;
  return formatted.str();
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

bool runAndCapture(const std::vector<std::string> &arguments,
                   std::size_t maximum_bytes, bool verbose,
                   std::string &standard_output, std::string &error) {
  if (arguments.empty()) {
    error = "internal error: empty compiler command";
    return false;
  }
  if (verbose) {
    std::cerr << "matcore-extract bootstrap command:";
    for (const std::string &argument : arguments) {
      std::cerr << ' ' << shellQuoted(argument);
    }
    std::cerr << '\n';
  }

  int descriptors[2];
  if (::pipe(descriptors) != 0) {
    error = "failed to create Clang output pipe: " +
            std::string(std::strerror(errno));
    return false;
  }
  const pid_t child = ::fork();
  if (child < 0) {
    error = "failed to fork Clang: " + std::string(std::strerror(errno));
    ::close(descriptors[0]);
    ::close(descriptors[1]);
    return false;
  }
  if (child == 0) {
    ::close(descriptors[0]);
    if (::dup2(descriptors[1], STDOUT_FILENO) < 0) {
      _exit(126);
    }
    ::close(descriptors[1]);
    std::vector<char *> raw_arguments;
    raw_arguments.reserve(arguments.size() + 1);
    for (const std::string &argument : arguments) {
      raw_arguments.push_back(const_cast<char *>(argument.c_str()));
    }
    raw_arguments.push_back(nullptr);
    ::execvp(raw_arguments[0], raw_arguments.data());
    std::cerr << "matcore-extract: failed to execute " << arguments[0] << ": "
              << std::strerror(errno) << '\n';
    _exit(127);
  }

  ::close(descriptors[1]);
  char buffer[64U * 1024U];
  bool exceeded_limit = false;
  for (;;) {
    const ssize_t count = ::read(descriptors[0], buffer, sizeof(buffer));
    if (count > 0) {
      const std::size_t bytes_read = static_cast<std::size_t>(count);
      if (standard_output.size() > maximum_bytes ||
          bytes_read > maximum_bytes - standard_output.size()) {
        exceeded_limit = true;
        ::kill(child, SIGKILL);
        break;
      }
      standard_output.append(buffer, bytes_read);
      continue;
    }
    if (count == 0) {
      break;
    }
    if (errno != EINTR) {
      error = "failed reading Clang AST JSON: " +
              std::string(std::strerror(errno));
      ::kill(child, SIGKILL);
      break;
    }
  }
  ::close(descriptors[0]);

  int status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited < 0 && error.empty()) {
    error = "failed waiting for Clang: " + std::string(std::strerror(errno));
  }
  if (exceeded_limit) {
    error = "Clang AST JSON exceeded the configured byte limit (" +
            std::to_string(maximum_bytes) + ")";
    return false;
  }
  if (!error.empty()) {
    return false;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    if (WIFEXITED(status)) {
      error = "Clang parsing/Sema failed with exit code " +
              std::to_string(WEXITSTATUS(status));
    } else {
      error = "Clang parsing/Sema terminated abnormally";
    }
    return false;
  }
  return true;
}

bool forbiddenCompilerArgument(const std::string &argument) {
  return argument == "-c" || argument == "-S" || argument == "-emit-llvm" ||
         argument == "-save-temps" || argument == "--save-temps" ||
         argument == "-E" || argument == "-M" || argument == "-MM" ||
         argument == "-MD" || argument == "-MMD" || argument == "-MJ" ||
         argument == "-MF" || argument == "-MT" || argument == "-MQ" ||
         argument == "-Xclang" || argument == "-load" ||
         argument == "-###" ||
         argument.starts_with("-fplugin=") ||
         (argument.starts_with("-o") && argument.size() > 2) ||
         (argument.starts_with("-MF") && argument.size() > 3) ||
         (argument.starts_with("-MJ") && argument.size() > 3);
}

bool makeCompilerCommand(const Options &options,
                         std::vector<std::string> &command,
                         std::string &error) {
  command.push_back(options.clang_path);
  for (std::size_t index = 0; index < options.compiler_arguments.size(); ++index) {
    const std::string &argument = options.compiler_arguments[index];
    if (argument == options.input_path ||
        canonicalPath(argument) == canonicalPath(options.input_path)) {
      continue;
    }
    if (argument == "-fsyntax-only") {
      continue;
    }
    if (argument == "-x") {
      if (++index == options.compiler_arguments.size()) {
        error = "-x requires a language argument";
        return false;
      }
      continue;
    }
    if (argument == "-o" || argument == "-MF" || argument == "-MT" ||
        argument == "-MQ" || argument == "-MJ") {
      error = "output-producing compiler argument is not valid for extraction: " +
              argument;
      return false;
    }
    if (forbiddenCompilerArgument(argument)) {
      error = "unsafe or conflicting compiler argument for extraction: " +
              argument;
      return false;
    }
    command.push_back(argument);
  }
  command.insert(command.end(), {"-x", "c++", "-fsyntax-only",
                                  "-fno-color-diagnostics", "-Xclang",
                                  "-ast-dump=json", options.input_path});
  return true;
}

class AstJsonBootstrapFrontend final : public Frontend {
public:
  bool extract(const Options &options, Result &result) override;

private:
  void diagnose(Result &result, const AstLocation &location,
                std::string message) const;
  bool processCall(
      const JsonValue &call, const WalkContext &context,
      const std::unordered_map<std::string, Declaration> &declarations,
      const std::string &input_canonical, const std::string &display_path,
      std::string_view source, Result &result,
      std::unordered_set<std::string> &allowed_operation_references) const;
  void scan(
      const JsonValue &node, WalkContext context,
      std::string &last_source_file,
      const std::unordered_map<std::string, Declaration> &declarations,
      const std::string &input_canonical, const std::string &display_path,
      std::string_view source, Result &result,
      std::unordered_set<std::string> &allowed_operation_references) const;
};

void AstJsonBootstrapFrontend::diagnose(Result &result,
                                        const AstLocation &location,
                                        std::string message) const {
  result.diagnostics.push_back(Diagnostic{.file = location.file,
                                          .line = location.line,
                                          .column = location.column,
                                          .message = std::move(message)});
}

bool AstJsonBootstrapFrontend::processCall(
    const JsonValue &call, const WalkContext &context,
    const std::unordered_map<std::string, Declaration> &declarations,
    const std::string &input_canonical, const std::string &display_path,
    std::string_view source, Result &result,
    std::unordered_set<std::string> &allowed_operation_references) const {
  const std::optional<DirectCallee> callee = directCallee(call);
  if (!callee || roleForId(callee->declaration_id, declarations) !=
                     DeclarationRole::gemm) {
    return true;
  }
  const std::string reference_id = memberString(*callee->reference, "id");
  if (!reference_id.empty()) {
    allowed_operation_references.insert(reference_id);
  }

  AstLocation location = nodeLocation(call, context.source_file);
  if (location.file.empty()) {
    location.file = display_path;
  }
  if (location.has_offset && canonicalPath(location.file) == input_canonical) {
    const auto [line, column] = lineAndColumn(source, location.offset);
    location.file = display_path;
    location.line = line;
    location.column = column;
  }

  bool valid = true;
  auto reject = [&](std::string message) {
    diagnose(result, location, std::move(message));
    valid = false;
  };
  const auto call_source_range = sourceRange(call);
  if (location.macro || !call_source_range) {
    reject("Matcore calls generated by macros or unsafe source ranges are not "
           "supported in bootstrap v0");
    return false;
  }
  if (canonicalPath(context.source_file) != input_canonical) {
    reject("Matcore call sites must be written directly in the input .mdsl file");
    return false;
  }
  if (context.in_template) {
    reject("Matcore calls inside templates are not supported in bootstrap v0");
    return false;
  }
  if (context.in_lambda) {
    reject("Matcore calls inside lambdas are not supported in bootstrap v0");
    return false;
  }
  if (context.in_constexpr || context.in_constant_evaluation) {
    reject("constexpr Matcore execution is not supported in bootstrap v0");
    return false;
  }

  const std::string callee_spelling = sourceText(*callee->reference, source);
  if (canonicalPath(context.source_file) == input_canonical &&
      callee_spelling.find("::") == std::string::npos) {
    reject("Matcore operations must be directly qualified (for example, "
           "matcore::mdsl::gemm or md::gemm); unqualified and ADL calls are "
           "rejected");
  }

  const JsonValue *children = innerArray(call);
  if (children == nullptr || children->Size() != 5) {
    reject("bootstrap gemm requires explicit output, lhs, rhs, and policy "
           "arguments after Sema");
    return false;
  }

  const JsonValue &output_expression = (*children)[1];
  const std::optional<DirectCallee> output_wrapper =
      directCallee(output_expression);
  std::string output_name;
  if (!output_wrapper ||
      roleForId(output_wrapper->declaration_id, declarations) !=
          DeclarationRole::out) {
    reject("gemm output must use the canonical matcore::mdsl::out wrapper");
  } else {
    const std::string wrapper_spelling =
        sourceText(*output_wrapper->reference, source);
    if (wrapper_spelling.find("::") == std::string::npos) {
      reject("the out wrapper must be directly namespace-qualified");
    }
    const JsonValue *output_children = innerArray(output_expression);
    if (output_children == nullptr || output_children->Size() != 2 ||
        !isSimpleLvalue((*output_children)[1], output_name)) {
      reject("gemm output must be a stable mutable lvalue with no side effects");
    }
  }

  std::string lhs_name;
  std::string rhs_name;
  if (!isSimpleLvalue((*children)[2], lhs_name)) {
    reject("gemm lhs must be a stable matrix lvalue with no side effects");
  }
  if (!isSimpleLvalue((*children)[3], rhs_name)) {
    reject("gemm rhs must be a stable matrix lvalue with no side effects");
  }
  if (!output_name.empty() &&
      (output_name == lhs_name || output_name == rhs_name)) {
    reject("gemm output must not alias lhs or rhs");
  }

  std::string target;
  std::string fallback;
  if (!parsePolicy((*children)[4], target, fallback)) {
    reject("gemm policy must be the default or an inline policy aggregate so "
           "target and fallback are compile-time visible");
  } else if (target != "cpu" || fallback != "error") {
    reject("bootstrap v0 supports only target=cpu with fallback=error; no "
           "silent fallback is permitted");
  }

  std::vector<ir::SourceRange> argument_ranges;
  for (rapidjson::SizeType index = 1; index <= 3; ++index) {
    const auto range = sourceRange((*children)[index]);
    if (!range || range->first < call_source_range->first ||
        range->second > call_source_range->second) {
      reject("gemm argument does not have a safe main-file source range");
      break;
    }
    argument_ranges.push_back(
        ir::SourceRange{.begin = range->first, .end = range->second});
  }
  if (memberString((*children)[4], "kind") != "CXXDefaultArgExpr") {
    const auto range = sourceRange((*children)[4]);
    if (!range || range->first < call_source_range->first ||
        range->second > call_source_range->second) {
      reject("explicit gemm policy does not have a safe main-file source range");
    } else {
      argument_ranges.push_back(
          ir::SourceRange{.begin = range->first, .end = range->second});
    }
  }

  if (!valid || !location.has_offset) {
    return false;
  }
  ir::Operation operation;
  operation.site_id = siteId(result.module.translation_unit, location.offset,
                             "gemm");
  operation.kind = "gemm";
  operation.canonical_callee = "matcore::mdsl::gemm";
  operation.source = ir::SourceLocation{.file = display_path,
                                        .offset = location.offset,
                                        .line = location.line,
                                        .column = location.column};
  operation.call_range = ir::SourceRange{.begin = call_source_range->first,
                                         .end = call_source_range->second};
  operation.argument_ranges = std::move(argument_ranges);
  operation.output = ir::MatrixValue{.role = "output",
                                     .expression = output_name,
                                     .mutability = "write"};
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
  result.module.operations.push_back(std::move(operation));
  return true;
}

void AstJsonBootstrapFrontend::scan(
    const JsonValue &node, WalkContext context,
    std::string &last_source_file,
    const std::unordered_map<std::string, Declaration> &declarations,
    const std::string &input_canonical, const std::string &display_path,
    std::string_view source, Result &result,
    std::unordered_set<std::string> &allowed_operation_references) const {
  if (!node.IsObject()) {
    return;
  }
  const AstLocation location = nodeLocation(node, last_source_file);
  if (!location.file.empty()) {
    last_source_file = location.file;
    context.source_file = location.file;
  }
  const std::string kind = memberString(node, "kind");
  context.in_lambda = context.in_lambda || kind == "LambdaExpr";
  context.in_template = context.in_template || kind == "FunctionTemplateDecl" ||
                        kind == "ClassTemplateDecl" ||
                        kind == "ClassTemplateSpecializationDecl";
  context.in_constexpr =
      context.in_constexpr ||
      ((kind == "FunctionDecl" || kind == "CXXMethodDecl") &&
       memberBool(node, "constexpr"));
  context.in_constant_evaluation =
      context.in_constant_evaluation || kind == "ConstantExpr" ||
      kind == "StaticAssertDecl";

  if (kind == "CallExpr") {
    processCall(node, context, declarations, input_canonical, display_path,
                source, result, allowed_operation_references);
  }
  if (kind == "DeclRefExpr") {
    const JsonValue *referenced = memberObject(node, "referencedDecl");
    const std::string referenced_id =
        referenced == nullptr ? std::string{} : memberString(*referenced, "id");
    const std::string reference_id = memberString(node, "id");
    if (roleForId(referenced_id, declarations) == DeclarationRole::gemm &&
        !allowed_operation_references.contains(reference_id)) {
      AstLocation diagnostic_location = nodeLocation(node, context.source_file);
      if (diagnostic_location.has_offset &&
          canonicalPath(context.source_file) == input_canonical) {
        const auto [line, column] =
            lineAndColumn(source, diagnostic_location.offset);
        diagnostic_location.file = display_path;
        diagnostic_location.line = line;
        diagnostic_location.column = column;
      }
      diagnose(result, diagnostic_location,
               "indirect or function-pointer references to Matcore operations "
               "are not supported");
    }
  }

  if (const JsonValue *inner = innerArray(node)) {
    for (const JsonValue &child : inner->GetArray()) {
      scan(child, context, last_source_file, declarations, input_canonical,
           display_path, source, result, allowed_operation_references);
    }
  }
}

bool AstJsonBootstrapFrontend::extract(const Options &options, Result &result) {
  result = Result{};
  const std::string display_path = normalizeDisplayPath(options.input_path);
  result.module.translation_unit = display_path;
  result.module.source_file = display_path;
  result.module.producer = "clang-ast-json-bootstrap-v0";

  if (!std::string_view(display_path).ends_with(".mdsl")) {
    result.diagnostics.push_back(
        Diagnostic{.file = display_path,
                   .message = "input source must use the .mdsl extension"});
    return false;
  }
  std::ifstream source_stream(options.input_path, std::ios::binary);
  if (!source_stream) {
    result.diagnostics.push_back(
        Diagnostic{.file = display_path,
                   .message = "unable to open input .mdsl source"});
    return false;
  }
  const std::string source((std::istreambuf_iterator<char>(source_stream)),
                           std::istreambuf_iterator<char>());

  std::vector<std::string> command;
  std::string error;
  if (!makeCompilerCommand(options, command, error)) {
    result.diagnostics.push_back(
        Diagnostic{.file = display_path, .message = std::move(error)});
    return false;
  }
  std::string ast_json;
  if (!runAndCapture(command, options.maximum_ast_bytes, options.verbose,
                     ast_json, error)) {
    result.diagnostics.push_back(
        Diagnostic{.file = display_path, .message = std::move(error)});
    return false;
  }

  rapidjson::Document document;
  document.ParseInsitu(ast_json.data());
  if (document.HasParseError()) {
    result.diagnostics.push_back(Diagnostic{
        .file = display_path,
        .message = "invalid Clang AST JSON at byte " +
                   std::to_string(document.GetErrorOffset()) + ": " +
                   rapidjson::GetParseError_En(document.GetParseError())});
    return false;
  }

  std::unordered_map<std::string, Declaration> declarations;
  std::string declaration_source_file = display_path;
  collectDeclarations(document, {}, declaration_source_file, declarations);

  std::unordered_set<std::string> allowed_operation_references;
  std::string scan_source_file = display_path;
  scan(document, WalkContext{.source_file = display_path}, scan_source_file,
       declarations, canonicalPath(options.input_path), display_path, source,
       result, allowed_operation_references);
  std::sort(result.module.operations.begin(), result.module.operations.end(),
            [](const ir::Operation &left, const ir::Operation &right) {
              return left.source.offset < right.source.offset;
            });
  std::sort(result.diagnostics.begin(), result.diagnostics.end(),
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
  result.diagnostics.erase(
      std::unique(result.diagnostics.begin(), result.diagnostics.end(),
                  [](const Diagnostic &left, const Diagnostic &right) {
                    return left.file == right.file && left.line == right.line &&
                           left.column == right.column &&
                           left.message == right.message;
                  }),
      result.diagnostics.end());
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
  return true;
}

} // namespace

std::unique_ptr<Frontend> createClangAstJsonBootstrapFrontend() {
  return std::make_unique<AstJsonBootstrapFrontend>();
}

} // namespace matcore::mdslc::frontend
