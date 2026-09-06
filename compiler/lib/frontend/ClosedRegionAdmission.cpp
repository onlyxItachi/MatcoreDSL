#include "ClosedRegionAdmission.h"

#define MDSLC_INTERNAL_EMBED_CLOSED_REGION_FIXTURE
#include "../../tests/closed_region/fixture_language.h"
#undef MDSLC_INTERNAL_EMBED_CLOSED_REGION_FIXTURE

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/StmtCXX.h>
#include <clang/AST/TypeLoc.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Basic/Version.h>
#include <clang/Frontend/ASTUnit.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/VirtualFileSystem.h>

#include <map>
#include <set>
#include <utility>

namespace matcore::mdslc::frontend {
namespace {
namespace cr = closed_region;
constexpr const char *input_path = "/__mdsl_private__/input.cpp";
constexpr const char *header_path = "/__mdsl_private__/fixture.h";
constexpr const char *region_annotation = "mdsl.private.closed_region.v1";

std::string digest(const std::string &bytes) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes)), true);
}

bool hasPreprocessing(const std::string &source) {
  clang::LangOptions language;
  language.CPlusPlus = true;
  language.CPlusPlus11 = true;
  language.CPlusPlus20 = true;
  language.Digraphs = true;
  clang::Lexer lexer(clang::SourceLocation::getFromRawEncoding(1), language,
                     source.data(), source.data(), source.data() + source.size());
  clang::Token token;
  do {
    lexer.LexFromRawLexer(token);
    if (token.isOneOf(clang::tok::hash, clang::tok::hashhash) || token.needsCleaning())
      return true;
    if (token.is(clang::tok::raw_identifier) &&
        (token.getRawIdentifier() == "_Pragma" || token.getRawIdentifier() == "__pragma"))
      return true;
  } while (!token.is(clang::tok::eof));
  return false;
}

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

enum class BindingKind { Value, Storage, Shape };
struct Binding {
  BindingKind kind = BindingKind::Value;
  cr::Id id = 0;
  cr::Dimension dimension;
};
using Environment = std::map<const clang::ValueDecl *, Binding>;

// This is AST-to-semantics lowering, not an evaluator. It never runs a body,
// constructs a user object, substitutes a template, or chooses a shape-if arm.
class Admission {
public:
  Admission(clang::ASTContext &context, const std::string &selected,
            cr::Program &program, std::string &error)
      : context_(context), sources_(context.getSourceManager()),
        selected_(selected), program_(program), error_(error) {}

  bool run() {
    discover(context_.getTranslationUnitDecl());
    if (!value_type_ || !storage_type_ || !numerics_type_ || ops_.size() != 6)
      return reject({}, "tool-owned declaration set is incomplete");
    if (entries_.size() != 1 || selected_declarations_.size() != 1)
      return reject({}, "select exactly one non-overloaded region definition");
    const clang::FunctionDecl *entry = *entries_.begin();
    if (!validateFunction(entry, true)) return false;
    cr::Region region;
    region.name = selected_;
    region.site = site(entry->getSourceRange());
    Environment environment;
    std::uint64_t index = 0;
    for (const clang::ParmVarDecl *parameter : entry->parameters()) {
      if (!validateParameter(parameter)) return false;
      Binding binding;
      if (isRecord(parameter->getType(), storage_type_)) {
        binding.kind = BindingKind::Storage;
        binding.id = next_resource_++;
        region.resources.push_back(
            {binding.id, parameter->getNameAsString(), index});
      } else if (isShape(parameter->getType())) {
        binding.kind = BindingKind::Shape;
        binding.dimension.kind = cr::Dimension::Kind::ShapeParameter;
        binding.dimension.reference = next_shape_++;
        region.shape_parameters.push_back({binding.dimension.reference,
                                           parameter->getNameAsString(), index});
      } else {
        return reject(parameter->getLocation(),
                      "region parameters must be Storage or Shape by value");
      }
      environment[parameter] = binding;
      ++index;
    }
    Binding ignored;
    if (!body(entry, environment, region.body, false, ignored)) return false;
    program_.regions.push_back(std::move(region));
    return true;
  }

private:
  bool inMain(clang::SourceLocation location) const {
    return location.isValid() && !location.isMacroID() &&
           sources_.getFileID(location) == sources_.getMainFileID();
  }
  bool inHeader(clang::SourceLocation location) const {
    return location.isValid() && !location.isMacroID() &&
           sources_.getFilename(location) == header_path;
  }
  bool reject(clang::SourceLocation location, const std::string &reason) {
    if (error_.empty()) {
      error_ = program_.source_identity;
      if (location.isValid()) {
        const auto spelling = sources_.getSpellingLoc(location);
        error_ += ":" + std::to_string(sources_.getSpellingLineNumber(spelling)) +
                  ":" + std::to_string(sources_.getSpellingColumnNumber(spelling));
      }
      error_ += ": closed-region admission rejected: " + reason;
    }
    return false;
  }
  cr::SourceSite site(clang::SourceRange range) const {
    cr::SourceSite result;
    const auto begin = sources_.getSpellingLoc(range.getBegin());
    const auto end = clang::Lexer::getLocForEndOfToken(
        sources_.getSpellingLoc(range.getEnd()), 0, sources_,
        context_.getLangOpts());
    result.offset = sources_.getFileOffset(begin);
    result.length = sources_.getFileOffset(end) - result.offset;
    result.line = sources_.getSpellingLineNumber(begin);
    result.column = sources_.getSpellingColumnNumber(begin);
    return result;
  }
  void discover(const clang::DeclContext *context) {
    for (const clang::Decl *declaration : context->decls()) {
      if (const auto *space = llvm::dyn_cast<clang::NamespaceDecl>(declaration)) {
        discover(space);
      } else if (const auto *record =
                     llvm::dyn_cast<clang::CXXRecordDecl>(declaration)) {
        if (!inHeader(record->getLocation()) || !record->isThisDeclarationADefinition())
          continue;
        const std::string name = record->getQualifiedNameAsString();
        if (name == "mdsl_probe::Value") value_type_ = record->getCanonicalDecl();
        if (name == "mdsl_probe::Storage") storage_type_ = record->getCanonicalDecl();
      } else if (const auto *enumeration =
                     llvm::dyn_cast<clang::EnumDecl>(declaration)) {
        if (inHeader(enumeration->getLocation()) &&
            enumeration->getQualifiedNameAsString() == "mdsl_probe::Numerics")
          numerics_type_ = enumeration->getCanonicalDecl();
      } else if (const auto *function =
                     llvm::dyn_cast<clang::FunctionDecl>(declaration)) {
        if (inHeader(function->getLocation())) {
          const std::string name = function->getNameAsString();
          if (name == "read" || name == "gemm" || name == "publish" ||
              name == "observe" || name == "rows" || name == "cols")
            ops_[function->getCanonicalDecl()] = name;
        }
        if (inMain(function->getLocation()) &&
            function->getQualifiedNameAsString() == selected_) {
          selected_declarations_.insert(function->getCanonicalDecl());
          if (function->hasBody()) entries_.insert(function->getDefinition());
        }
      }
    }
  }
  bool isRecord(clang::QualType type,
                const clang::CXXRecordDecl *record) const {
    if (type.isNull() || type.isVolatileQualified() || type->isReferenceType())
      return false;
    const auto *actual = type->getAsCXXRecordDecl();
    return actual && actual->getCanonicalDecl() == record;
  }
  bool isShape(clang::QualType type) const {
    return !type.isNull() && !type.isVolatileQualified() &&
           context_.hasSameType(type.getUnqualifiedType(), context_.UnsignedLongLongTy);
  }
  bool admittedType(clang::QualType type) const {
    return isRecord(type, value_type_) || isRecord(type, storage_type_) || isShape(type);
  }
  bool validateTypeLocation(const clang::TypeSourceInfo *information) {
    if (!information) return reject({}, "missing source type binding");
    for (clang::TypeLoc location = information->getTypeLoc(); !location.isNull();
         location = location.getNextTypeLoc()) {
      if (!inMain(location.getBeginLoc()) || !inMain(location.getEndLoc()))
        return reject(location.getBeginLoc(), "macro or unowned source type spelling");
      if (!location.getAs<clang::DecltypeTypeLoc>().isNull() ||
          !location.getAs<clang::TypeOfExprTypeLoc>().isNull() ||
          !location.getAs<clang::TypeOfTypeLoc>().isNull() ||
          !location.getAs<clang::AttributedTypeLoc>().isNull())
        return reject(location.getBeginLoc(),
                      "expression-bearing or attributed type spelling is outside the closed grammar");
    }
    // An ordinary alias formed outside the region is staged by Clang. Its
    // resolved type must still be one of the exact canonical admitted types;
    // no type-expression callback is run by this admission implementation.
    return true;
  }
  bool validateParameter(const clang::ParmVarDecl *parameter, bool require_name = true) {
    if (!validateTypeLocation(parameter->getTypeSourceInfo())) return false;
    if (parameter->hasDefaultArg() || parameter->hasAttrs() ||
        !admittedType(parameter->getType()) ||
        (require_name && parameter->getName().empty()))
      return reject(parameter->getLocation(),
                    "parameters require named, unqualified-effect-free value types; "
                    "references, pointers, attributes and defaults are unsupported");
    return true;
  }
  bool validateFunction(const clang::FunctionDecl *function, bool entry) {
    if (!inMain(function->getLocation()) || !function->hasBody() ||
        llvm::isa<clang::CXXMethodDecl>(function) || function->isVariadic() ||
        function->isDeleted() ||
        function->getDescribedFunctionTemplate())
      return reject(function->getLocation(),
                    "only source-owned free function definitions are admitted");
    if (!validateTypeLocation(function->getTypeSourceInfo())) return false;
    unsigned markers = 0;
    for (const clang::FunctionDecl *declaration : function->getCanonicalDecl()->redecls()) {
      if (!inMain(declaration->getLocation()))
        return reject(declaration->getLocation(), "helper or region redeclaration is not source-owned");
      const auto *prototype = declaration->getType()->getAs<clang::FunctionProtoType>();
      if (!prototype || prototype->getCallConv() != clang::CC_C)
        return reject(declaration->getLocation(), "nonstandard calling convention is not admitted");
      for (const clang::Attr *attribute : declaration->attrs()) {
        const auto *annotation = llvm::dyn_cast<clang::AnnotateAttr>(attribute);
        if (entry && annotation && annotation->getAnnotation() == region_annotation &&
            inMain(annotation->getLocation())) {
          if (declaration == function) ++markers;
        } else {
          return reject(declaration->getLocation(), "unadmitted function attribute");
        }
      }
      for (const clang::ParmVarDecl *parameter : declaration->parameters())
        if (!validateParameter(parameter, declaration == function)) return false;
    }
    if ((entry && markers != 1) ||
        (entry && !function->getReturnType()->isVoidType()) ||
        (!entry && !isRecord(function->getReturnType(), value_type_) &&
         !isShape(function->getReturnType())))
      return reject(function->getLocation(),
                    "region requires one private marker and void result; helper "
                    "requires a Value or Shape result");
    for (const clang::ParmVarDecl *parameter : function->parameters())
      if (!validateParameter(parameter)) return false;
    return true;
  }
  bool authenticateOperation(const clang::FunctionDecl *function,
                             const std::string &name) {
    const auto *canonical = function->getCanonicalDecl();
    for (const clang::FunctionDecl *declaration : canonical->redecls()) {
      if (!inHeader(declaration->getLocation()) || declaration->hasBody())
        return reject(declaration->getLocation(),
                      "canonical primitive redeclaration or definition is untrusted");
      unsigned annotations = 0;
      for (const clang::Attr *attribute : declaration->attrs()) {
        const auto *annotation = llvm::dyn_cast<clang::AnnotateAttr>(attribute);
        if (!annotation || annotation->getAnnotation() !=
                               "mdsl.private." + name + ".v1")
          return reject(declaration->getLocation(), "untrusted primitive attributes");
        ++annotations;
      }
      if (annotations != 1)
        return reject(declaration->getLocation(), "missing canonical primitive binding");
    }
    return true;
  }
  bool sourceNode(const clang::Stmt *statement) {
    if (++visited_nodes_ > 16384)
      return reject(statement ? statement->getBeginLoc() : clang::SourceLocation(),
                    "bounded admission AST expansion budget exceeded");
    if (!statement || !inMain(statement->getBeginLoc()) ||
        !inMain(statement->getEndLoc()))
      return reject(statement ? statement->getBeginLoc() : clang::SourceLocation(),
                    "macro or non-main-source AST node is unsupported");
    return true;
  }

  // Only semantic-preserving implicit wrappers are removed. User conversions,
  // arbitrary cast kinds, cleanup nodes and expressions are not ignored.
  const clang::Expr *unwrap(const clang::Expr *expression) {
    while (expression) {
      if (!sourceNode(expression) || expression->getType().isVolatileQualified()) {
        reject(expression->getExprLoc(), "volatile or unowned expression");
        return nullptr;
      }
      if (const auto *parentheses = llvm::dyn_cast<clang::ParenExpr>(expression)) {
        expression = parentheses->getSubExpr();
      } else if (const auto *cast = llvm::dyn_cast<clang::ImplicitCastExpr>(expression)) {
        const auto kind = cast->getCastKind();
        if (kind == clang::CK_NoOp || kind == clang::CK_LValueToRValue) {
          if (!admittedType(cast->getType()) ||
              !admittedType(cast->getSubExpr()->getType())) {
            reject(expression->getExprLoc(), "implicit conversion outside canonical types");
            return nullptr;
          }
          expression = cast->getSubExpr();
        } else if (kind == clang::CK_IntegralCast && isShape(cast->getType())) {
          // Clang's conversion of a nonnegative integer literal to Shape is
          // exact. No runtime scalar arithmetic or user evaluation is added.
          const auto *literal = llvm::dyn_cast<clang::IntegerLiteral>(
              cast->getSubExpr()->IgnoreParens());
          if (!literal || literal->getValue().getActiveBits() > 63) {
            reject(expression->getExprLoc(), "only exact nonnegative literal-to-Shape conversion is admitted");
            return nullptr;
          }
          expression = literal;
        } else {
          reject(expression->getExprLoc(), "implicit conversion is not admitted");
          return nullptr;
        }
      } else if (const auto *construction =
                     llvm::dyn_cast<clang::CXXConstructExpr>(expression)) {
        const clang::CXXConstructorDecl *constructor = construction->getConstructor();
        if ((!isRecord(expression->getType(), value_type_) &&
             !isRecord(expression->getType(), storage_type_)) ||
            !constructor->isImplicit() || !constructor->isTrivial() ||
            !constructor->isCopyOrMoveConstructor() ||
            construction->getNumArgs() != 1 ||
            !inHeader(constructor->getParent()->getLocation())) {
          reject(expression->getExprLoc(), "noncanonical construction or hidden special member");
          return nullptr;
        }
        expression = construction->getArg(0);
      } else {
        return expression;
      }
    }
    return nullptr;
  }
  bool bound(const clang::Expr *expression, const Environment &environment,
             Binding &binding, bool shape_literal = false) {
    expression = unwrap(expression);
    if (!expression) return false;
    if (const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression)) {
      const auto found = environment.find(reference->getDecl());
      if (found != environment.end()) {
        binding = found->second;
        return true;
      }
    }
    if (shape_literal) {
      if (const auto *literal = llvm::dyn_cast<clang::IntegerLiteral>(expression)) {
        if (literal->getValue().getActiveBits() <= 63) {
          binding.kind = BindingKind::Shape;
          binding.dimension.literal = literal->getValue().getZExtValue();
          return true;
        }
      }
    }
    return reject(expression->getExprLoc(),
                  "operation arguments must be immutable bound values/resources "
                  "or exact shape literals; nested calls and host expressions are forbidden");
  }
  bool argument(const clang::CallExpr *call, unsigned index, BindingKind kind,
                const Environment &environment, Binding &binding) {
    if (index >= call->getNumArgs() ||
        !bound(call->getArg(index), environment, binding, kind == BindingKind::Shape))
      return false;
    return binding.kind == kind ||
           reject(call->getArg(index)->getExprLoc(), "semantic argument kind mismatch");
  }
  bool numerical(const clang::Expr *expression, cr::NumericalProfile &profile) {
    expression = unwrap(expression);
    if (!expression) return false;
    const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression);
    const auto *constant = reference ?
        llvm::dyn_cast<clang::EnumConstantDecl>(reference->getDecl()) : nullptr;
    if (!constant || !inHeader(constant->getLocation()) ||
        constant->getDeclContext() != numerics_type_)
      return reject(expression->getExprLoc(), "numerics require a canonical explicit profile");
    if (constant->getName() == "strict_f32") profile = cr::NumericalProfile::StrictF32;
    else if (constant->getName() == "reassociate_f32") profile = cr::NumericalProfile::ReassociateF32;
    else return reject(expression->getExprLoc(), "unsupported numerical profile");
    return true;
  }
  bool call(const clang::CallExpr *call, Environment &environment,
            std::vector<cr::Operation> &operations, bool helper, Binding &result) {
    if (!sourceNode(call) || llvm::isa<clang::CXXMemberCallExpr>(call) ||
        llvm::isa<clang::CXXOperatorCallExpr>(call))
      return reject(call->getExprLoc(), "member, virtual and operator calls are not admitted");
    const clang::FunctionDecl *callee = call->getDirectCallee();
    if (!callee) return reject(call->getExprLoc(), "indirect call is not admitted");
    // Verify the callee expression too: direct-callee recovery is not permission
    // to discard a comma expression or other hidden callee evaluation.
    const clang::Expr *callee_expression = call->getCallee();
    if (const auto *cast = llvm::dyn_cast<clang::ImplicitCastExpr>(callee_expression)) {
      if (cast->getCastKind() != clang::CK_FunctionToPointerDecay)
        return reject(call->getExprLoc(), "unadmitted callee conversion");
      callee_expression = cast->getSubExpr();
    }
    const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(callee_expression);
    if (!reference || !sourceNode(callee_expression) ||
        reference->getDecl()->getCanonicalDecl() != callee->getCanonicalDecl())
      return reject(call->getExprLoc(), "callee is not a direct authenticated declaration reference");
    const auto found = ops_.find(callee->getCanonicalDecl());
    if (found == ops_.end()) return helperCall(call, callee, environment, operations, result);
    const std::string &name = found->second;
    if (!authenticateOperation(callee, name)) return false;
    if (helper && (name == "read" || name == "publish" || name == "observe"))
      return reject(call->getExprLoc(), "pure helper cannot access external storage or effects");
    cr::Operation operation;
    operation.site = site(call->getSourceRange());
    operation.helper_calls = helper_sites_;
    Binding first, second;
    if (name == "read") {
      Binding rows, columns;
      if (call->getNumArgs() != 3 ||
          !argument(call, 0, BindingKind::Storage, environment, first) ||
          !argument(call, 1, BindingKind::Shape, environment, rows) ||
          !argument(call, 2, BindingKind::Shape, environment, columns)) return false;
      operation.kind = cr::Operation::Kind::Read;
      operation.result = next_value_++;
      operation.resource = first.id;
      operation.rows = rows.dimension;
      operation.columns = columns.dimension;
      result = {BindingKind::Value, operation.result, {}};
    } else if (name == "gemm") {
      if (call->getNumArgs() != 3 ||
          !argument(call, 0, BindingKind::Value, environment, first) ||
          !argument(call, 1, BindingKind::Value, environment, second) ||
          !numerical(call->getArg(2), operation.numerical_profile)) return false;
      operation.kind = cr::Operation::Kind::Gemm;
      operation.result = next_value_++;
      operation.lhs = first.id;
      operation.rhs = second.id;
      result = {BindingKind::Value, operation.result, {}};
    } else if (name == "publish") {
      if (call->getNumArgs() != 2 ||
          !argument(call, 0, BindingKind::Value, environment, first) ||
          !argument(call, 1, BindingKind::Storage, environment, second)) return false;
      operation.kind = cr::Operation::Kind::Publish;
      operation.lhs = first.id;
      operation.resource = second.id;
    } else if (name == "observe") {
      if (call->getNumArgs() != 1 ||
          !argument(call, 0, BindingKind::Storage, environment, first)) return false;
      operation.kind = cr::Operation::Kind::Observe;
      operation.resource = first.id;
    } else {
      if (call->getNumArgs() != 1 ||
          !argument(call, 0, BindingKind::Value, environment, first)) return false;
      result.kind = BindingKind::Shape;
      result.dimension.kind = name == "rows" ? cr::Dimension::Kind::ValueRows
                                              : cr::Dimension::Kind::ValueColumns;
      result.dimension.reference = first.id;
      return true;
    }
    operations.push_back(std::move(operation));
    return true;
  }
  bool helperCall(const clang::CallExpr *call, const clang::FunctionDecl *callee,
                  Environment &environment, std::vector<cr::Operation> &operations,
                  Binding &result) {
    const clang::FunctionDecl *definition = callee->getDefinition();
    if (!definition || !validateFunction(definition, false))
      return reject(call->getExprLoc(), "unknown host call or unadmitted helper");
    if (active_helpers_.size() >= 16 ||
        active_helpers_.count(definition->getCanonicalDecl()))
      return reject(call->getExprLoc(), "recursive or excessive helper expansion");
    if (call->getNumArgs() != definition->getNumParams())
      return reject(call->getExprLoc(), "helper argument count mismatch");
    Environment helper_environment;
    for (unsigned index = 0; index < call->getNumArgs(); ++index) {
      Binding binding;
      const auto *parameter = definition->getParamDecl(index);
      if (!bound(call->getArg(index), environment, binding, isShape(parameter->getType())))
        return false;
      const bool compatible =
          (binding.kind == BindingKind::Value && isRecord(parameter->getType(), value_type_)) ||
          (binding.kind == BindingKind::Shape && isShape(parameter->getType()));
      if (!compatible)
        return reject(parameter->getLocation(), "pure helper parameters may only be Value or Shape");
      helper_environment[parameter] = binding;
    }
    active_helpers_.insert(definition->getCanonicalDecl());
    helper_sites_.push_back(site(call->getSourceRange()));
    const bool okay = body(definition, helper_environment, operations, true, result);
    helper_sites_.pop_back();
    active_helpers_.erase(definition->getCanonicalDecl());
    return okay;
  }
  bool expression(const clang::Expr *expression, Environment &environment,
                  std::vector<cr::Operation> &operations, bool helper, Binding &result) {
    expression = unwrap(expression);
    if (!expression) return false;
    if (const auto *invocation = llvm::dyn_cast<clang::CallExpr>(expression))
      return call(invocation, environment, operations, helper, result);
    return bound(expression, environment, result, true);
  }
  bool condition(const clang::Expr *expression, Environment &environment,
                 cr::Operation &operation) {
    if (!sourceNode(expression)) return false;
    expression = expression->IgnoreParens();
    const auto *comparison = llvm::dyn_cast<clang::BinaryOperator>(expression);
    if (!comparison || !comparison->isComparisonOp())
      return reject(expression->getExprLoc(), "shape-if requires one builtin shape comparison");
    Binding lhs, rhs;
    if (!bound(comparison->getLHS(), environment, lhs, true) ||
        !bound(comparison->getRHS(), environment, rhs, true) ||
        lhs.kind != BindingKind::Shape || rhs.kind != BindingKind::Shape)
      return reject(expression->getExprLoc(), "shape comparison operands must be Shape values");
    switch (comparison->getOpcode()) {
    case clang::BO_LT: operation.comparison = cr::Comparison::Less; break;
    case clang::BO_LE: operation.comparison = cr::Comparison::LessEqual; break;
    case clang::BO_EQ: operation.comparison = cr::Comparison::Equal; break;
    case clang::BO_NE: operation.comparison = cr::Comparison::NotEqual; break;
    case clang::BO_GT: operation.comparison = cr::Comparison::Greater; break;
    case clang::BO_GE: operation.comparison = cr::Comparison::GreaterEqual; break;
    default: return reject(expression->getExprLoc(), "unsupported shape comparison");
    }
    operation.condition_lhs = lhs.dimension;
    operation.condition_rhs = rhs.dimension;
    return true;
  }
  bool statement(const clang::Stmt *statement, Environment &environment,
                 std::vector<cr::Operation> &operations, bool helper) {
    if (!sourceNode(statement)) return false;
    if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(statement)) {
      Environment nested = environment;
      for (const clang::Stmt *child : compound->body())
        if (!this->statement(child, nested, operations, helper)) return false;
      return true;
    }
    if (const auto *declarations = llvm::dyn_cast<clang::DeclStmt>(statement)) {
      if (!declarations->isSingleDecl())
        return reject(statement->getBeginLoc(), "one immutable local binding per declaration required");
      const auto *variable = llvm::dyn_cast<clang::VarDecl>(declarations->getSingleDecl());
      if (!variable || !inMain(variable->getLocation()) || variable->hasAttrs() ||
          variable->hasGlobalStorage() || !variable->hasInit() ||
          !admittedType(variable->getType()))
        return reject(statement->getBeginLoc(),
                      "local binding has unsupported type, storage, attributes or initialization");
      if (!validateTypeLocation(variable->getTypeSourceInfo())) return false;
      Binding binding;
      if (!expression(variable->getInit(), environment, operations, helper, binding)) return false;
      const bool compatible =
          (binding.kind == BindingKind::Value && isRecord(variable->getType(), value_type_)) ||
          (binding.kind == BindingKind::Storage && isRecord(variable->getType(), storage_type_)) ||
          (binding.kind == BindingKind::Shape && isShape(variable->getType()));
      if (!compatible) return reject(variable->getLocation(), "local semantic type mismatch");
      environment[variable] = binding;
      return true;
    }
    if (const auto *branch = llvm::dyn_cast<clang::IfStmt>(statement)) {
      if (helper || branch->getInit() || branch->getConditionVariable() ||
          branch->isConstexpr() || branch->isConsteval() || !branch->getElse())
        return reject(branch->getIfLoc(),
                      "shape-if requires an explicit else, no init/constexpr, and a region body");
      cr::Operation operation;
      operation.kind = cr::Operation::Kind::ShapeIf;
      operation.site = site(branch->getSourceRange());
      operation.helper_calls = helper_sites_;
      if (!condition(branch->getCond(), environment, operation)) return false;
      Environment then_environment = environment, else_environment = environment;
      if (!this->statement(branch->getThen(), then_environment, operation.then_body, helper) ||
          !this->statement(branch->getElse(), else_environment, operation.else_body, helper))
        return false;
      operations.push_back(std::move(operation));
      return true;
    }
    if (const auto *invocation = llvm::dyn_cast<clang::CallExpr>(statement)) {
      Binding ignored;
      return call(invocation, environment, operations, helper, ignored);
    }
    return reject(statement->getBeginLoc(),
                  std::string("AST statement is outside the closed vocabulary: ") +
                      statement->getStmtClassName());
  }
  bool body(const clang::FunctionDecl *function, Environment &environment,
            std::vector<cr::Operation> &operations, bool helper, Binding &result) {
    const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(function->getBody());
    if (!compound || !sourceNode(compound))
      return reject(function->getLocation(), "function requires an ordinary compound body");
    const auto statements = compound->body();
    for (auto iterator = statements.begin(); iterator != statements.end(); ++iterator) {
      if (const auto *returned = llvm::dyn_cast<clang::ReturnStmt>(*iterator)) {
        if (!sourceNode(returned)) return false;
        if (std::next(iterator) != statements.end())
          return reject(returned->getReturnLoc(), "only a terminal helper return is admitted");
        if (!helper)
          return !returned->getRetValue() ||
                 reject(returned->getReturnLoc(), "region must not return a host value");
        if (!returned->getRetValue() ||
            !expression(returned->getRetValue(), environment, operations, true, result))
          return false;
        if ((result.kind == BindingKind::Value && isRecord(function->getReturnType(), value_type_)) ||
            (result.kind == BindingKind::Shape && isShape(function->getReturnType())))
          return true;
        return reject(returned->getReturnLoc(), "helper return semantic type mismatch");
      }
      if (!statement(*iterator, environment, operations, helper)) return false;
    }
    return !helper || reject(function->getLocation(), "helper requires one terminal return");
  }

  clang::ASTContext &context_;
  clang::SourceManager &sources_;
  const std::string &selected_;
  cr::Program &program_;
  std::string &error_;
  const clang::CXXRecordDecl *value_type_ = nullptr;
  const clang::CXXRecordDecl *storage_type_ = nullptr;
  const clang::EnumDecl *numerics_type_ = nullptr;
  std::map<const clang::FunctionDecl *, std::string> ops_;
  std::set<const clang::FunctionDecl *> entries_;
  std::set<const clang::FunctionDecl *> selected_declarations_;
  std::set<const clang::FunctionDecl *> active_helpers_;
  std::vector<cr::SourceSite> helper_sites_;
  cr::Id next_resource_ = 1, next_shape_ = 1, next_value_ = 1;
  std::size_t visited_nodes_ = 0;
};
} // namespace

struct AuthenticatedClosedRegionEvidence::Payload {
  cr::Program program;
  std::string source;
  std::string region_name;
};
AuthenticatedClosedRegionEvidence::AuthenticatedClosedRegionEvidence(
    std::shared_ptr<const Payload> payload) : payload_(std::move(payload)) {}
const cr::Program &AuthenticatedClosedRegionEvidence::program() const {
  return payload_->program;
}
const std::string &AuthenticatedClosedRegionEvidence::sourceSnapshot() const {
  return payload_->source;
}
const std::string &AuthenticatedClosedRegionEvidence::regionName() const {
  return payload_->region_name;
}

ClosedRegionAdmissionResult admitClosedRegionSource(
    const std::string &source, const std::string &source_identity,
    const std::string &region_name) {
  ClosedRegionAdmissionResult result;
  if (source.empty() || source.size() > 1024 * 1024 || source_identity.empty() ||
      source_identity.size() > 4096 || region_name.empty() ||
      source.find('\0') != std::string::npos || hasPreprocessing(source)) {
    result.error = "closed-region fixture requires a bounded immutable source, identity, "
                   "and no preprocessing directives or embedded NUL";
    return result;
  }
  auto filesystem = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  filesystem->setCurrentWorkingDirectory("/");
  Diagnostics diagnostics;
  const std::vector<std::string> arguments = {
      "-x", "c++", "-std=c++20", "-fsyntax-only", "-nostdinc", "-nostdinc++",
      "-fno-modules", "-fno-delayed-template-parsing", "-include", header_path};
  const clang::tooling::FileContentMappings files = {
      {header_path, detail::closed_region_fixture_source}};
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      source, arguments, input_path, "clang-tool",
      std::make_shared<clang::PCHContainerOperations>(),
      clang::tooling::getClangStripDependencyFileAdjuster(), files,
      &diagnostics, filesystem);
  if (!ast || diagnostics.failed || ast->getDiagnostics().hasErrorOccurred()) {
    result.error = "closed-region Clang syntax/type validation failed: " + diagnostics.text;
    return result;
  }
  result.syntax_valid = true;
  cr::Program program;
  program.source_identity = source_identity;
  program.source_sha256 = digest(source);
  program.header_sha256 = digest(detail::closed_region_fixture_source);
  program.compiler_identity = clang::getClangFullVersion();
  Admission admission(ast->getASTContext(), region_name, program, result.error);
  if (!admission.run() || !cr::verifyProgram(program, result.error)) return result;
  auto payload = std::make_shared<AuthenticatedClosedRegionEvidence::Payload>();
  payload->program = std::move(program);
  payload->source = source;
  payload->region_name = region_name;
  result.evidence = AuthenticatedClosedRegionEvidence(std::move(payload));
  return result;
}

bool verifyClosedRegionMatchesEvidence(
    const AuthenticatedClosedRegionEvidence &evidence, mlir::ModuleOp module,
    std::string &error) {
  const auto replay = admitClosedRegionSource(
      evidence.sourceSnapshot(), evidence.program().source_identity, evidence.regionName());
  if (!replay) {
    error = "sealed source failed re-admission: " + replay.error;
    return false;
  }
  return cr::verifyModuleMatchesProgram(replay.evidence->program(), module, error);
}
} // namespace matcore::mdslc::frontend
