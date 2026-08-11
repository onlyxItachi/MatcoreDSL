#include "recovered_gemm_recognizer.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/ParentMapContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/StmtOpenACC.h>
#include <clang/AST/StmtOpenMP.h>
#include <clang/AST/StmtSYCL.h>
#include <clang/AST/ASTTypeTraits.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace matcore::mdslc::frontend {
namespace {

using ReasonSet = std::set<std::string>;

struct LoopControl {
  const clang::VarDecl *induction = nullptr;
  const clang::ValueDecl *bound = nullptr;
  const clang::Stmt *initialization = nullptr;
  const clang::Expr *condition = nullptr;
  const clang::Expr *increment = nullptr;
  const clang::Expr *bound_expression = nullptr;
};

struct ParsedCanonicalLoop {
  LoopControl outer;
  LoopControl middle;
  LoopControl inner;
  const clang::ForStmt *middle_loop = nullptr;
  const clang::ForStmt *inner_loop = nullptr;
  const clang::VarDecl *accumulator = nullptr;
  const clang::BinaryOperator *update = nullptr;
  const clang::BinaryOperator *store = nullptr;
  const clang::ArraySubscriptExpr *lhs_load = nullptr;
  const clang::ArraySubscriptExpr *rhs_load = nullptr;
  const clang::ArraySubscriptExpr *output_store = nullptr;
  const clang::Expr *lhs_base_expression = nullptr;
  const clang::Expr *rhs_base_expression = nullptr;
  const clang::Expr *output_base_expression = nullptr;
};

const clang::Expr *withoutParenImplicitCasts(const clang::Expr *expression) {
  return expression == nullptr ? nullptr : expression->IgnoreParenImpCasts();
}

const clang::DeclRefExpr *declReference(const clang::Expr *expression) {
  return llvm::dyn_cast_or_null<clang::DeclRefExpr>(
      withoutParenImplicitCasts(expression));
}

bool refersTo(const clang::Expr *expression, const clang::ValueDecl *expected) {
  const clang::DeclRefExpr *reference = declReference(expression);
  return reference != nullptr && reference->getDecl() == expected;
}

std::vector<const clang::Stmt *> statementsIn(const clang::Stmt *body) {
  const auto *compound = llvm::dyn_cast_or_null<clang::CompoundStmt>(body);
  if (compound == nullptr) {
    return {};
  }
  return std::vector<const clang::Stmt *>(compound->body_begin(),
                                          compound->body_end());
}

bool containsNonDefaultAddressSpace(clang::QualType type) {
  if (type.isNull()) {
    return false;
  }
  if (type.getAddressSpace() != clang::LangAS::Default) {
    return true;
  }
  type = type.getCanonicalType();
  if (type.getAddressSpace() != clang::LangAS::Default) {
    return true;
  }
  if (const auto *pointer = type->getAs<clang::PointerType>()) {
    return containsNonDefaultAddressSpace(pointer->getPointeeType());
  }
  if (const auto *reference = type->getAs<clang::ReferenceType>()) {
    return containsNonDefaultAddressSpace(reference->getPointeeType());
  }
  if (const clang::ArrayType *array = type->getAsArrayTypeUnsafe()) {
    return containsNonDefaultAddressSpace(array->getElementType());
  }
  return false;
}

bool isCanonicalSizeType(clang::QualType type,
                         const clang::ASTContext &context) {
  return !containsNonDefaultAddressSpace(type) &&
         context.hasSameType(type.getCanonicalType().getUnqualifiedType(),
                             static_cast<clang::QualType>(context.getSizeType())
                                 .getUnqualifiedType());
}

bool isFloatType(clang::QualType type) {
  return type.getCanonicalType()
      .getUnqualifiedType()
      ->isSpecificBuiltinType(clang::BuiltinType::Float);
}

bool isFloatingEvaluationType(clang::QualType type) {
  const auto *builtin = type.getCanonicalType()
                            .getUnqualifiedType()
                            ->getAs<clang::BuiltinType>();
  return builtin != nullptr && builtin->isFloatingType();
}

bool containsAtomicType(clang::QualType type) {
  if (type.isNull()) {
    return false;
  }
  type = type.getCanonicalType();
  if (type->isAtomicType()) {
    return true;
  }
  if (const auto *pointer = type->getAs<clang::PointerType>()) {
    return containsAtomicType(pointer->getPointeeType());
  }
  if (const auto *reference = type->getAs<clang::ReferenceType>()) {
    return containsAtomicType(reference->getPointeeType());
  }
  const auto *record_type = type->getAs<clang::RecordType>();
  if (record_type == nullptr) {
    return false;
  }
  const std::string name =
      record_type->getDecl()->getCanonicalDecl()->getQualifiedNameAsString();
  return name == "std::atomic" || name == "std::atomic_ref" ||
         name.starts_with("std::atomic<") ||
         name.starts_with("std::atomic_ref<");
}

bool isFloatPointer(clang::QualType type, bool require_const_pointee) {
  if (containsNonDefaultAddressSpace(type)) {
    return false;
  }
  type = type.getCanonicalType().getUnqualifiedType();
  const auto *pointer = type->getAs<clang::PointerType>();
  if (pointer == nullptr || !isFloatType(pointer->getPointeeType())) {
    return false;
  }
  return pointer->getPointeeType().isConstQualified() ==
         require_const_pointee;
}

bool isPositiveFloatZero(const clang::Expr *expression) {
  const auto *literal = llvm::dyn_cast_or_null<clang::FloatingLiteral>(
      withoutParenImplicitCasts(expression));
  return literal != nullptr && isFloatType(literal->getType()) &&
         literal->getValue().isZero() && !literal->getValue().isNegative();
}

bool parseLoopControl(const clang::ForStmt &loop, clang::ASTContext &context,
                      LoopControl &control) {
  const auto *declaration_statement =
      llvm::dyn_cast_or_null<clang::DeclStmt>(loop.getInit());
  if (declaration_statement == nullptr || !declaration_statement->isSingleDecl()) {
    return false;
  }
  const auto *induction = llvm::dyn_cast<clang::VarDecl>(
      declaration_statement->getSingleDecl());
  if (induction == nullptr || !induction->hasInit() ||
      !isCanonicalSizeType(induction->getType(), context)) {
    return false;
  }
  const auto *zero = llvm::dyn_cast_or_null<clang::IntegerLiteral>(
      withoutParenImplicitCasts(induction->getInit()));
  if (zero == nullptr || !zero->getValue().isZero()) {
    return false;
  }

  const auto *condition = llvm::dyn_cast_or_null<clang::BinaryOperator>(
      withoutParenImplicitCasts(loop.getCond()));
  if (condition == nullptr || condition->getOpcode() != clang::BO_LT ||
      !refersTo(condition->getLHS(), induction)) {
    return false;
  }
  const clang::DeclRefExpr *bound = declReference(condition->getRHS());
  if (bound == nullptr || !isCanonicalSizeType(bound->getType(), context)) {
    return false;
  }

  const auto *increment = llvm::dyn_cast_or_null<clang::UnaryOperator>(
      withoutParenImplicitCasts(loop.getInc()));
  if (increment == nullptr || increment->getOpcode() != clang::UO_PreInc ||
      !refersTo(increment->getSubExpr(), induction)) {
    return false;
  }

  control.induction = induction;
  control.bound = bound->getDecl();
  control.initialization = declaration_statement;
  control.condition = condition;
  control.increment = increment;
  control.bound_expression = condition->getRHS();
  return true;
}

bool exactLinearIndex(const clang::Expr *expression,
                      const clang::ValueDecl *row,
                      const clang::ValueDecl *stride,
                      const clang::ValueDecl *column) {
  const auto *add = llvm::dyn_cast_or_null<clang::BinaryOperator>(
      withoutParenImplicitCasts(expression));
  if (add == nullptr || add->getOpcode() != clang::BO_Add ||
      !refersTo(add->getRHS(), column)) {
    return false;
  }
  const auto *multiply = llvm::dyn_cast_or_null<clang::BinaryOperator>(
      withoutParenImplicitCasts(add->getLHS()));
  return multiply != nullptr && multiply->getOpcode() == clang::BO_Mul &&
         refersTo(multiply->getLHS(), row) &&
         refersTo(multiply->getRHS(), stride);
}

const clang::ParmVarDecl *arrayBase(const clang::ArraySubscriptExpr &subscript,
                                    const clang::Expr *&base_expression) {
  base_expression = subscript.getBase();
  const clang::DeclRefExpr *reference = declReference(base_expression);
  return reference == nullptr
             ? nullptr
             : llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl());
}

bool parseCanonicalLoop(const clang::ForStmt &outer_loop,
                        const clang::FunctionDecl &function,
                        clang::ASTContext &context,
                        ParsedCanonicalLoop &parsed) {
  if (!parseLoopControl(outer_loop, context, parsed.outer)) {
    return false;
  }
  const std::vector<const clang::Stmt *> outer_body =
      statementsIn(outer_loop.getBody());
  if (outer_body.size() != 1) {
    return false;
  }
  parsed.middle_loop = llvm::dyn_cast<clang::ForStmt>(outer_body[0]);
  if (parsed.middle_loop == nullptr ||
      !parseLoopControl(*parsed.middle_loop, context, parsed.middle)) {
    return false;
  }

  const std::vector<const clang::Stmt *> middle_body =
      statementsIn(parsed.middle_loop->getBody());
  if (middle_body.size() != 3) {
    return false;
  }
  const auto *accumulator_statement =
      llvm::dyn_cast<clang::DeclStmt>(middle_body[0]);
  if (accumulator_statement == nullptr ||
      !accumulator_statement->isSingleDecl()) {
    return false;
  }
  parsed.accumulator = llvm::dyn_cast<clang::VarDecl>(
      accumulator_statement->getSingleDecl());
  if (parsed.accumulator == nullptr ||
      !isFloatType(parsed.accumulator->getType()) ||
      !isPositiveFloatZero(parsed.accumulator->getInit())) {
    return false;
  }

  parsed.inner_loop = llvm::dyn_cast<clang::ForStmt>(middle_body[1]);
  if (parsed.inner_loop == nullptr ||
      !parseLoopControl(*parsed.inner_loop, context, parsed.inner)) {
    return false;
  }
  const std::vector<const clang::Stmt *> inner_body =
      statementsIn(parsed.inner_loop->getBody());
  if (inner_body.size() != 1) {
    return false;
  }
  parsed.update = llvm::dyn_cast<clang::BinaryOperator>(
      withoutParenImplicitCasts(llvm::dyn_cast<clang::Expr>(inner_body[0])));
  if (parsed.update == nullptr ||
      parsed.update->getOpcode() != clang::BO_AddAssign ||
      !refersTo(parsed.update->getLHS(), parsed.accumulator) ||
      !isFloatType(parsed.update->getType())) {
    return false;
  }
  const auto *product = llvm::dyn_cast_or_null<clang::BinaryOperator>(
      withoutParenImplicitCasts(parsed.update->getRHS()));
  if (product == nullptr || product->getOpcode() != clang::BO_Mul ||
      !isFloatingEvaluationType(product->getType())) {
    return false;
  }
  parsed.lhs_load = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(
      withoutParenImplicitCasts(product->getLHS()));
  parsed.rhs_load = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(
      withoutParenImplicitCasts(product->getRHS()));
  if (parsed.lhs_load == nullptr || parsed.rhs_load == nullptr ||
      !isFloatType(parsed.lhs_load->getType()) ||
      !isFloatType(parsed.rhs_load->getType())) {
    return false;
  }

  parsed.store = llvm::dyn_cast<clang::BinaryOperator>(
      withoutParenImplicitCasts(llvm::dyn_cast<clang::Expr>(middle_body[2])));
  if (parsed.store == nullptr || parsed.store->getOpcode() != clang::BO_Assign ||
      !refersTo(parsed.store->getRHS(), parsed.accumulator)) {
    return false;
  }
  parsed.output_store = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(
      withoutParenImplicitCasts(parsed.store->getLHS()));
  if (parsed.output_store == nullptr ||
      !isFloatType(parsed.output_store->getType())) {
    return false;
  }

  if (function.getReturnType()->isVoidType() == false || function.isVariadic() ||
      function.getNumParams() != 6) {
    return false;
  }
  const clang::ParmVarDecl *output = function.getParamDecl(0);
  const clang::ParmVarDecl *lhs = function.getParamDecl(1);
  const clang::ParmVarDecl *rhs = function.getParamDecl(2);
  const clang::ParmVarDecl *m = function.getParamDecl(3);
  const clang::ParmVarDecl *n = function.getParamDecl(4);
  const clang::ParmVarDecl *k = function.getParamDecl(5);
  if (!isFloatPointer(output->getType(), false) ||
      !isFloatPointer(lhs->getType(), true) ||
      !isFloatPointer(rhs->getType(), true) ||
      !isCanonicalSizeType(m->getType(), context) ||
      !isCanonicalSizeType(n->getType(), context) ||
      !isCanonicalSizeType(k->getType(), context) ||
      parsed.outer.bound != m || parsed.middle.bound != n ||
      parsed.inner.bound != k) {
    return false;
  }

  const clang::ParmVarDecl *lhs_base =
      arrayBase(*parsed.lhs_load, parsed.lhs_base_expression);
  const clang::ParmVarDecl *rhs_base =
      arrayBase(*parsed.rhs_load, parsed.rhs_base_expression);
  const clang::ParmVarDecl *output_base =
      arrayBase(*parsed.output_store, parsed.output_base_expression);
  if (lhs_base != lhs || rhs_base != rhs || output_base != output ||
      !exactLinearIndex(parsed.lhs_load->getIdx(), parsed.outer.induction, k,
                        parsed.inner.induction) ||
      !exactLinearIndex(parsed.rhs_load->getIdx(), parsed.inner.induction, n,
                        parsed.middle.induction) ||
      !exactLinearIndex(parsed.output_store->getIdx(), parsed.outer.induction,
                        n, parsed.middle.induction)) {
    return false;
  }

  const auto *function_body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(function.getBody());
  return function_body != nullptr && function_body->size() == 1 &&
         *function_body->body_begin() == &outer_loop;
}

class SkeletonVisitor final
    : public clang::RecursiveASTVisitor<SkeletonVisitor> {
public:
  bool VisitForStmt(clang::ForStmt *) {
    ++loop_count;
    return true;
  }

  bool VisitBinaryOperator(clang::BinaryOperator *operation) {
    if (operation->getOpcode() == clang::BO_AddAssign) {
      const auto *product = llvm::dyn_cast_or_null<clang::BinaryOperator>(
          withoutParenImplicitCasts(operation->getRHS()));
      if (product != nullptr && product->getOpcode() == clang::BO_Mul) {
        saw_reduction = true;
      }
    }
    if (operation->getOpcode() == clang::BO_Assign &&
        llvm::isa<clang::ArraySubscriptExpr>(
            withoutParenImplicitCasts(operation->getLHS()))) {
      saw_store = true;
    }
    return true;
  }

  unsigned loop_count = 0;
  bool saw_reduction = false;
  bool saw_store = false;
};

class BarrierVisitor final
    : public clang::RecursiveASTVisitor<BarrierVisitor> {
public:
  explicit BarrierVisitor(ReasonSet &reasons) : reasons_(reasons) {}

  bool VisitCallExpr(clang::CallExpr *) {
    reasons_.insert("observable_call");
    return true;
  }

  bool VisitCXXConstructExpr(clang::CXXConstructExpr *) {
    reasons_.insert("observable_call");
    return true;
  }

  bool VisitAtomicExpr(clang::AtomicExpr *) {
    reasons_.insert("atomic_access");
    return true;
  }

  bool VisitExpr(clang::Expr *expression) {
    clang::QualType type = expression->getType();
    if (containsNonDefaultAddressSpace(type)) {
      reasons_.insert("unsupported_address_space");
    }
    if (!type.isNull() &&
        (type.isVolatileQualified() || containsAtomicType(type))) {
      reasons_.insert(containsAtomicType(type) ? "atomic_access"
                                               : "volatile_access");
    }
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *reference) {
    const auto *value = llvm::dyn_cast<clang::ValueDecl>(reference->getDecl());
    if (value != nullptr) {
      clang::QualType type = value->getType();
      if (containsNonDefaultAddressSpace(type)) {
        reasons_.insert("unsupported_address_space");
      }
      if (type.isVolatileQualified()) {
        reasons_.insert("volatile_access");
      }
      if (containsAtomicType(type)) {
        reasons_.insert("atomic_access");
      }
      if (const auto *pointer = type->getAs<clang::PointerType>()) {
        if (pointer->getPointeeType().isVolatileQualified()) {
          reasons_.insert("volatile_access");
        }
        if (containsAtomicType(pointer->getPointeeType())) {
          reasons_.insert("atomic_access");
        }
      }
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *declaration) {
    if (llvm::isa<clang::ParmVarDecl>(declaration)) {
      return true;
    }
    if (declaration->getStorageDuration() != clang::SD_Automatic ||
        declaration->getTLSKind() != clang::VarDecl::TLS_None ||
        declaration->getStorageClass() != clang::SC_None) {
      reasons_.insert("unsupported_storage_duration");
    }
    if (containsNonDefaultAddressSpace(declaration->getType())) {
      reasons_.insert("unsupported_address_space");
    }
    for (const clang::Attr *attribute : declaration->attrs()) {
      reasons_.insert(llvm::isa<clang::CleanupAttr>(attribute)
                          ? "cleanup_attribute"
                          : "unsupported_attribute");
    }
    return true;
  }

  bool VisitIfStmt(clang::IfStmt *) { return controlFlow(); }
  bool VisitSwitchStmt(clang::SwitchStmt *) { return controlFlow(); }
  bool VisitConditionalOperator(clang::ConditionalOperator *) {
    return controlFlow();
  }
  bool VisitBreakStmt(clang::BreakStmt *) { return controlFlow(); }
  bool VisitContinueStmt(clang::ContinueStmt *) { return controlFlow(); }
  bool VisitReturnStmt(clang::ReturnStmt *) { return controlFlow(); }
  bool VisitGotoStmt(clang::GotoStmt *) { return controlFlow(); }
  bool VisitLabelStmt(clang::LabelStmt *) { return controlFlow(); }
  bool VisitCXXThrowExpr(clang::CXXThrowExpr *) { return controlFlow(); }
  bool VisitCXXTryStmt(clang::CXXTryStmt *) { return controlFlow(); }

  bool VisitGCCAsmStmt(clang::GCCAsmStmt *) {
    reasons_.insert("unsupported_control_flow");
    return true;
  }

  bool VisitMSAsmStmt(clang::MSAsmStmt *) {
    reasons_.insert("unsupported_control_flow");
    return true;
  }

  bool VisitOMPExecutableDirective(clang::OMPExecutableDirective *) {
    reasons_.insert("offload_or_parallel_context");
    return true;
  }

  bool VisitOpenACCConstructStmt(clang::OpenACCConstructStmt *) {
    reasons_.insert("offload_or_parallel_context");
    return true;
  }

  bool VisitSYCLKernelCallStmt(clang::SYCLKernelCallStmt *) {
    reasons_.insert("offload_or_parallel_context");
    return true;
  }

  bool VisitAttributedStmt(clang::AttributedStmt *statement) {
    for (const clang::Attr *attribute : statement->getAttrs()) {
      if (llvm::isa<clang::LoopHintAttr>(attribute)) {
        reasons_.insert("optimization_barrier");
      }
    }
    return true;
  }

private:
  bool controlFlow() {
    reasons_.insert("unsupported_control_flow");
    return true;
  }

  ReasonSet &reasons_;
};

bool hasMacroOrigin(const clang::Stmt &statement) {
  if (statement.getBeginLoc().isMacroID() || statement.getEndLoc().isMacroID()) {
    return true;
  }
  for (const clang::Stmt *child : statement.children()) {
    if (child != nullptr && hasMacroOrigin(*child)) {
      return true;
    }
  }
  return false;
}

void inspectAttributedAncestors(const clang::Stmt &statement,
                                clang::ASTContext &context,
                                ReasonSet &reasons, unsigned depth = 0) {
  if (depth > 64) {
    reasons.insert("optimization_barrier");
    return;
  }
  for (const clang::DynTypedNode &parent :
       context.getParents(clang::DynTypedNode::create(statement))) {
    if (const auto *attributed = parent.get<clang::AttributedStmt>()) {
      for (const clang::Attr *attribute : attributed->getAttrs()) {
        if (llvm::isa<clang::LoopHintAttr>(attribute)) {
          reasons.insert("optimization_barrier");
        }
      }
      inspectAttributedAncestors(*attributed, context, reasons, depth + 1);
    }
  }
}

std::optional<RecoveredSourceRange>
tokenRange(clang::SourceRange range, const clang::SourceManager &source_manager,
           const clang::LangOptions &language_options,
           clang::FileID expected_file) {
  clang::SourceLocation begin = range.getBegin();
  clang::SourceLocation end = range.getEnd();
  if (begin.isInvalid() || end.isInvalid() || begin.isMacroID() ||
      end.isMacroID()) {
    return std::nullopt;
  }
  begin = source_manager.getSpellingLoc(begin);
  end = source_manager.getSpellingLoc(end);
  const clang::SourceLocation after_end = clang::Lexer::getLocForEndOfToken(
      end, 0, source_manager, language_options);
  if (after_end.isInvalid() || source_manager.getFileID(begin) != expected_file ||
      source_manager.getFileID(after_end) != expected_file) {
    return std::nullopt;
  }
  const std::uint64_t begin_offset = source_manager.getFileOffset(begin);
  const std::uint64_t end_offset = source_manager.getFileOffset(after_end);
  if (end_offset <= begin_offset) {
    return std::nullopt;
  }
  return RecoveredSourceRange{begin_offset, end_offset};
}

void addRange(std::vector<RecoveredNamedRange> &ranges,
              std::string_view role, clang::SourceRange source_range,
              const clang::SourceManager &source_manager,
              const clang::LangOptions &language_options,
              clang::FileID expected_file, bool &valid) {
  const std::optional<RecoveredSourceRange> range = tokenRange(
      source_range, source_manager, language_options, expected_file);
  if (!range) {
    valid = false;
    return;
  }
  ranges.push_back(
      RecoveredNamedRange{std::string(role), RecoveredSourceRange(*range)});
}

std::string stringifyRoundingMode(llvm::RoundingMode mode) {
  std::string result;
  llvm::raw_string_ostream output(result);
  output << mode;
  return result;
}

std::string stringifyExceptionMode(
    clang::LangOptions::FPExceptionModeKind mode) {
  switch (mode) {
  case clang::LangOptions::FPE_Ignore:
    return "ignore";
  case clang::LangOptions::FPE_MayTrap:
    return "maytrap";
  case clang::LangOptions::FPE_Strict:
    return "strict";
  case clang::LangOptions::FPE_Default:
    return "default";
  }
  return "invalid";
}

std::string stringifyEvaluationMethod(
    clang::LangOptions::FPEvalMethodKind method) {
  switch (method) {
  case clang::LangOptions::FEM_Source:
    return "source";
  case clang::LangOptions::FEM_Double:
    return "double";
  case clang::LangOptions::FEM_Extended:
    return "extended";
  case clang::LangOptions::FEM_UnsetOnCommandLine:
    return "unset";
  }
  return "invalid";
}

bool collectEnclosingFpScopes(
    const clang::DynTypedNode &node, clang::ASTContext &context,
    std::vector<const clang::CompoundStmt *> &scopes, unsigned depth = 0) {
  if (depth > 64) {
    return false;
  }
  const auto parents = context.getParents(node);
  if (parents.empty()) {
    return true;
  }
  // Statements in the authenticated canonical loop have one lexical parent.
  // Refuse to invent a path when Clang reports an ambiguous ownership graph.
  if (parents.size() != 1) {
    return false;
  }
  const clang::DynTypedNode &parent = *parents.begin();
  if (const auto *compound = parent.get<clang::CompoundStmt>()) {
    scopes.push_back(compound);
  }
  if (parent.get<clang::Stmt>() != nullptr) {
    return collectEnclosingFpScopes(parent, context, scopes, depth + 1);
  }
  return true;
}

clang::FPOptions effectiveFpOptionsAt(const clang::BinaryOperator &operation,
                                      clang::ASTContext &context,
                                      bool &scope_valid) {
  std::vector<const clang::CompoundStmt *> scopes;
  scope_valid = collectEnclosingFpScopes(
      clang::DynTypedNode::create(operation), context, scopes);
  clang::FPOptions result(context.getLangOpts());
  for (auto scope = scopes.rbegin(); scope != scopes.rend(); ++scope) {
    result.applyChanges((*scope)->getStoredFPFeaturesOrDefault());
  }
  result.applyChanges(operation.getStoredFPFeaturesOrDefault());
  return result;
}

std::string sourceFileFor(const clang::SourceManager &source_manager,
                          clang::FileID file_id,
                          const RecoveredGemmInspectionInputs &inputs) {
  if (file_id == source_manager.getMainFileID()) {
    return inputs.main_display_path;
  }
  if (const clang::OptionalFileEntryRef entry =
          source_manager.getFileEntryRefForID(file_id)) {
    if (!entry->getFileEntry().tryGetRealPathName().empty()) {
      return entry->getFileEntry().tryGetRealPathName().str();
    }
    return entry->getName().str();
  }
  return {};
}

void addContextReasons(const clang::ForStmt &outer_loop,
                       const clang::FunctionDecl &function,
                       clang::ASTContext &context, ReasonSet &reasons) {
  const clang::SourceManager &source_manager = context.getSourceManager();
  clang::SourceLocation location =
      source_manager.getSpellingLoc(outer_loop.getBeginLoc());
  if (location.isInvalid() || !source_manager.isWrittenInMainFile(location)) {
    reasons.insert("non_main_file");
  }
  if (hasMacroOrigin(outer_loop)) {
    reasons.insert("macro_origin");
  }
  if (function.getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate) {
    reasons.insert("template_context");
  }
  if (const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(&function)) {
    if (method->getParent()->isLambda()) {
      reasons.insert("lambda_context");
    } else {
      reasons.insert("member_function_context");
    }
  }
  if (function.isConstexpr() || function.isConsteval()) {
    reasons.insert("constexpr_context");
  }
  if (function.hasAttr<clang::OptimizeNoneAttr>()) {
    reasons.insert("optimization_barrier");
  }
  for (const clang::Attr *attribute : function.attrs()) {
    if (llvm::isa<clang::OptimizeNoneAttr>(attribute)) {
      continue;
    }
    reasons.insert(
        llvm::isa<clang::OMPDeclareSimdDeclAttr,
                  clang::OMPDeclareTargetDeclAttr>(attribute)
            ? "offload_or_parallel_context"
            : "unsupported_attribute");
  }
  for (const clang::ParmVarDecl *parameter : function.parameters()) {
    if (parameter->getStorageClass() != clang::SC_None ||
        parameter->getTLSKind() != clang::VarDecl::TLS_None) {
      reasons.insert("unsupported_storage_duration");
    }
    if (containsNonDefaultAddressSpace(parameter->getType())) {
      reasons.insert("unsupported_address_space");
    }
    for (const clang::Attr *attribute : parameter->attrs()) {
      reasons.insert(llvm::isa<clang::CleanupAttr>(attribute)
                         ? "cleanup_attribute"
                         : "unsupported_attribute");
    }
  }
  const clang::LangOptions &language_options = context.getLangOpts();
  if (language_options.OpenACC || language_options.isSYCL() ||
      language_options.isTargetDevice()) {
    reasons.insert("offload_or_parallel_context");
  }
  inspectAttributedAncestors(outer_loop, context, reasons);
  BarrierVisitor visitor(reasons);
  visitor.TraverseStmt(const_cast<clang::Stmt *>(function.getBody()));
}

void addFloatingPointReasons(const RecoveredFpProof &proof,
                             ReasonSet &reasons) {
  if (proof.optimization_level == 0) {
    reasons.insert("optimization_barrier");
  }
  if (proof.evaluation_method != "source") {
    reasons.insert("fp_evaluation_method_mismatch");
  }
  if (!proof.allow_reassociation) {
    reasons.insert("fp_reassociation_forbidden");
  }
  if (!proof.contract_across_statement) {
    reasons.insert("fp_contraction_forbidden");
  }
  if (!proof.honor_nans) {
    reasons.insert("fp_nan_contract_mismatch");
  }
  if (!proof.honor_infinities) {
    reasons.insert("fp_infinity_contract_mismatch");
  }
  if (proof.preserve_signed_zero) {
    reasons.insert("fp_signed_zero_mismatch");
  }
  if (proof.allow_reciprocal || proof.allow_approximate_functions ||
      proof.fast_math_profile) {
    reasons.insert("fp_approximation_mismatch");
  }
  if (proof.rounding_mode != "tonearest") {
    reasons.insert("fp_rounding_dynamic");
  }
  if (proof.exception_mode != "ignore" || proof.fenv_access) {
    reasons.insert("fp_exceptions_observable");
  }
  if (proof.denormal_mode != "ieee,ieee" ||
      proof.fp32_denormal_mode != "ieee,ieee") {
    reasons.insert("fp_denormal_mode_mismatch");
  }
}

} // namespace

RecoveredGemmCandidate inspectRecoveredGemmLoop(
    const clang::ForStmt &outer_loop, const clang::FunctionDecl &function,
    clang::ASTContext &context, const RecoveredGemmInspectionInputs &inputs) {
  RecoveredGemmCandidate candidate;
  const clang::SourceManager &source_manager = context.getSourceManager();
  const clang::LangOptions &language_options = context.getLangOpts();
  clang::SourceLocation location =
      source_manager.getSpellingLoc(outer_loop.getBeginLoc());
  const clang::FileID file_id = location.isValid()
                                    ? source_manager.getFileID(location)
                                    : clang::FileID{};
  candidate.source_file = sourceFileFor(source_manager, file_id, inputs);
  candidate.source_identity =
      file_id == source_manager.getMainFileID()
          ? stableSourceIdentity(inputs.main_canonical_path)
          : stableSourceIdentity(candidate.source_file);
  candidate.compilation_identity = inputs.compilation_identity;
  candidate.function_name = function.getQualifiedNameAsString();
  if (location.isValid()) {
    candidate.line = source_manager.getSpellingLineNumber(location);
    candidate.column = source_manager.getSpellingColumnNumber(location);
    candidate.offset = source_manager.getFileOffset(location);
  }

  bool invalid_buffer = false;
  const llvm::StringRef source =
      file_id.isValid() ? source_manager.getBufferData(file_id, &invalid_buffer)
                        : llvm::StringRef{};
  if (!invalid_buffer) {
    const std::array<std::uint8_t, 32> digest =
        llvm::SHA256::hash(llvm::arrayRefFromStringRef(source));
    candidate.source_snapshot_sha256 =
        "sha256:" + llvm::toHex(llvm::ArrayRef(digest), true);
  }
  candidate.site_id = makeStableSiteId(
      candidate.source_identity, candidate.compilation_identity,
      std::string_view(source.data(), source.size()),
      candidate.offset, "recovered.cpp.gemm.v1");

  candidate.fp_proof.optimization_level = inputs.optimization_level;
  candidate.fp_proof.denormal_mode = inputs.denormal_mode;
  candidate.fp_proof.fp32_denormal_mode = inputs.fp32_denormal_mode;

  const std::optional<RecoveredSourceRange> outer_range = tokenRange(
      outer_loop.getSourceRange(), source_manager, language_options, file_id);
  if (outer_range) {
    candidate.outer_loop_range = *outer_range;
  }

  SkeletonVisitor skeleton;
  skeleton.TraverseStmt(const_cast<clang::ForStmt *>(&outer_loop));
  const bool plausible = skeleton.loop_count == 3 && skeleton.saw_reduction &&
                         skeleton.saw_store;

  ReasonSet context_reasons;
  if (plausible) {
    addContextReasons(outer_loop, function, context, context_reasons);
  }

  ParsedCanonicalLoop parsed;
  if (!parseCanonicalLoop(outer_loop, function, context, parsed)) {
    if (plausible && !context_reasons.empty()) {
      candidate.state = RecoveredGemmState::recognized_rejected;
      candidate.rejection_reasons.assign(context_reasons.begin(),
                                         context_reasons.end());
    } else {
      candidate.state = RecoveredGemmState::not_recognized;
      candidate.rejection_reasons = {"canonical_structure_mismatch"};
    }
    return candidate;
  }

  bool fp_scope_valid = false;
  const clang::FPOptions fp =
      effectiveFpOptionsAt(*parsed.update, context, fp_scope_valid);
  candidate.output_parameter = function.getParamDecl(0)->getNameAsString();
  candidate.lhs_parameter = function.getParamDecl(1)->getNameAsString();
  candidate.rhs_parameter = function.getParamDecl(2)->getNameAsString();
  candidate.m_parameter = function.getParamDecl(3)->getNameAsString();
  candidate.n_parameter = function.getParamDecl(4)->getNameAsString();
  candidate.k_parameter = function.getParamDecl(5)->getNameAsString();
  candidate.semantic_contract =
      "f32_row_major_overwrite_m_k__k_n__m_n";
  candidate.fp_proof.allow_reassociation = fp.getAllowFPReassociate();
  candidate.fp_proof.contract_across_statement =
      fp.allowFPContractAcrossStatement();
  candidate.fp_proof.honor_nans = !fp.getNoHonorNaNs();
  candidate.fp_proof.honor_infinities = !fp.getNoHonorInfs();
  candidate.fp_proof.preserve_signed_zero = !fp.getNoSignedZero();
  candidate.fp_proof.allow_reciprocal = fp.getAllowReciprocal();
  candidate.fp_proof.allow_approximate_functions = fp.getAllowApproxFunc();
  candidate.fp_proof.fenv_access = fp.getAllowFEnvAccess();
  candidate.fp_proof.fast_math_profile = language_options.FastMath;
  candidate.fp_proof.evaluation_method =
      stringifyEvaluationMethod(fp.getFPEvalMethod());
  candidate.fp_proof.rounding_mode =
      stringifyRoundingMode(fp.getRoundingMode());
  candidate.fp_proof.exception_mode =
      stringifyExceptionMode(fp.getExceptionMode());

  bool ranges_valid = outer_range.has_value();
  auto add = [&](std::string_view role, const clang::Stmt *statement) {
    if (statement == nullptr) {
      ranges_valid = false;
      return;
    }
    addRange(candidate.proof_ranges, role, statement->getSourceRange(),
             source_manager, language_options, file_id, ranges_valid);
  };
  add("outer_loop", &outer_loop);
  add("outer_init", parsed.outer.initialization);
  add("outer_condition", parsed.outer.condition);
  add("outer_increment", parsed.outer.increment);
  add("outer_bound_m", parsed.outer.bound_expression);
  add("middle_init", parsed.middle.initialization);
  add("middle_condition", parsed.middle.condition);
  add("middle_increment", parsed.middle.increment);
  add("middle_bound_n", parsed.middle.bound_expression);
  add("inner_init", parsed.inner.initialization);
  add("inner_condition", parsed.inner.condition);
  add("inner_increment", parsed.inner.increment);
  add("inner_bound_k", parsed.inner.bound_expression);
  add("accumulator_update", parsed.update);
  add("output_store", parsed.store);
  add("lhs_base", parsed.lhs_base_expression);
  add("rhs_base", parsed.rhs_base_expression);
  add("output_base", parsed.output_base_expression);

  ReasonSet reasons = std::move(context_reasons);
  if (!fp_scope_valid) {
    reasons.insert("fp_scope_unavailable");
  }
  if (!ranges_valid || invalid_buffer || candidate.source_file.empty() ||
      candidate.source_snapshot_sha256.empty()) {
    reasons.insert("unsafe_source_range");
  }
  addFloatingPointReasons(candidate.fp_proof, reasons);
  if (!inputs.denormal_mode_is_ieee || !inputs.fp32_denormal_mode_is_ieee) {
    reasons.insert("fp_denormal_mode_mismatch");
  }

  if (!reasons.empty()) {
    candidate.state = RecoveredGemmState::recognized_rejected;
    candidate.rejection_reasons.assign(reasons.begin(), reasons.end());
    return candidate;
  }

  candidate.state = RecoveredGemmState::recognized_guard_required;
  candidate.required_runtime_guards = {
      "positive_m_n_k",
      "overflow_safe_shapes_elements_and_bytes",
      "nonnull_observable_pointers",
      "natural_f32_alignment",
      "output_disjoint_from_lhs_and_rhs",
      "compatible_runtime_floating_point_environment",
      "legal_cpu_implementation_available",
  };
  return candidate;
}

} // namespace matcore::mdslc::frontend
