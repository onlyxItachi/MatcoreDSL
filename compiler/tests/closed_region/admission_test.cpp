#include "ClosedRegionAdmission.h"
#include "adversarial_sources.h"
#include "adversarial_trace_checks.h"
#include "mlir/IR/Builders.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace cr = matcore::mdslc::closed_region;
namespace fe = matcore::mdslc::frontend;
namespace {
unsigned checks = 0;
unsigned failures = 0;
void check(bool condition, const std::string &label) {
  ++checks;
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << label << '\n';
  }
}
std::string region(const std::string &body,
                   const std::string &helpers = "") {
  return "using namespace mdsl_probe;\n" + helpers +
      "\n[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\n"
      "void region(Storage A, Storage B, Storage C, Storage D, Storage E, "
      "Shape m, Shape k, Shape n, Shape p) {\n" + body + "\n}\n";
}
void collect(const std::vector<cr::Operation> &body,
             cr::Operation::Kind kind,
             std::vector<const cr::Operation *> &result) {
  for (const auto &op : body) {
    if (op.kind == kind) result.push_back(&op);
    collect(op.then_body, kind, result);
    collect(op.else_body, kind, result);
  }
}
struct Positive {
  std::string name;
  std::string source;
  unsigned gemms;
};
void checkSourceSpan(const std::string &source, const cr::SourceSite &site,
                     const std::string &prefix) {
  const bool bounded = site.offset <= source.size() && site.length &&
                       site.length <= source.size() - site.offset;
  check(bounded, "source span is within the independently supplied bytes");
  if (!bounded) return;
  check(source.substr(site.offset, site.length).starts_with(prefix),
        "source span begins at exact fixture operation " + prefix);
  const auto line = 1 + std::count(source.begin(), source.begin() + site.offset, '\n');
  const auto previous = site.offset ? source.rfind('\n', site.offset - 1) : std::string::npos;
  const auto column = site.offset - (previous == std::string::npos ? 0 : previous + 1) + 1;
  check(site.line == static_cast<unsigned long>(line) && site.column == column,
        "source line/column independently match bytes, not a replayed offset helper");
}
void checkSourceBody(const std::string &source, const std::vector<cr::Operation> &body) {
  for (const auto &op : body) {
    const char *prefix = nullptr;
    switch (op.kind) {
    case cr::Operation::Kind::Read: prefix = "read("; break;
    case cr::Operation::Kind::Gemm: prefix = "gemm("; break;
    case cr::Operation::Kind::Publish: prefix = "publish("; break;
    case cr::Operation::Kind::Observe: prefix = "observe("; break;
    case cr::Operation::Kind::ShapeIf: prefix = "if ("; break;
    }
    checkSourceSpan(source, op.site, prefix);
    for (const auto &call : op.helper_calls) checkSourceSpan(source, call, "product(");
    checkSourceBody(source, op.then_body);
    checkSourceBody(source, op.else_body);
  }
}
} // namespace

int main() {
  const std::string helper =
      "template<class T> T product(T a, T b) {\n"
      "  auto c = gemm(a,b,Numerics::strict_f32); return c;\n}\n";
  const std::vector<Positive> positive = {
      {"lhs_rectangular", region(
           "auto a=read(A,2,3); auto b=read(B,3,5);\n"
           "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C);\n"
           "auto d=read(D,5,7); auto e=gemm(c,d,Numerics::strict_f32);\n"
           "publish(e,E); observe(E);"), 2},
      {"rhs_rectangular_late_read", region(
           "auto a=read(A,2,3); auto b=read(B,3,5);\n"
           "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C); observe(C);\n"
           "auto d=read(D,7,2); auto e=gemm(d,c,Numerics::strict_f32);\n"
           "publish(e,E);"), 2},
      {"dynamic_reused_template_helper", region(
           "auto a=read(A,m,k); auto b=read(B,k,n);\n"
           "auto c=product(a,b); publish(c,C);\n"
           "auto d=read(D,n,p); auto e=product(c,d); publish(e,E);", helper), 2},
      {"symbolic_shape_branches", region(
           "auto a=read(A,m,k); auto b=read(B,k,n);\n"
           "auto c=product(a,b);\n"
           "if (m < n) { publish(c,C); observe(C); }\n"
           "else { publish(c,E); observe(E); }", helper), 1},
      {"symbolic_mathematical_branches", region(
           "auto a=read(A,m,k); auto b=read(B,k,n); auto c=product(a,b);\n"
           "if (m < n) { auto d=read(D,n,p); auto e=product(c,d); publish(e,E); }\n"
           "else { auto d=read(D,p,m); auto e=product(d,c); publish(e,E); }",
           helper), 3},
      {"old_value_after_same_storage_publication", region(
           "auto x=read(C,2,2); auto a=read(A,2,3); auto b=read(B,3,2);\n"
           "auto y=gemm(a,b,Numerics::strict_f32); publish(y,C);\n"
           "auto e=gemm(x,x,Numerics::strict_f32); publish(e,E);"), 2},
      {"value_shape_queries", region(
           "auto a=read(A,m,k); auto b=read(B,k,n);\n"
           "auto c=gemm(a,b,Numerics::reassociate_f32);\n"
           "Shape r=rows(c); Shape s=cols(c);\n"
           "auto d=read(D,s,r); auto e=gemm(c,d,Numerics::strict_f32);\n"
           "publish(e,E);"), 2},
      {"dead_result_retains_check", region(
           "auto a=read(A,m,k); auto b=read(B,k,n);\n"
           "auto c=gemm(a,b,Numerics::strict_f32); observe(C);"), 1},
      {"host_type_alias_staged_by_clang", region(
           "Extent rows=m; auto a=read(A,rows,k); auto b=read(B,k,n);\n"
           "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C);",
           "using Extent=decltype(Shape{});"), 1},
      {"explicit_main_prototype_marker",
           "using namespace mdsl_probe;\n"
           "[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\n"
           "void region(Storage A, Storage B, Storage C, Shape m, Shape k, Shape n);\n"
           "void region(Storage A, Storage B, Storage C, Shape m, Shape k, Shape n) {\n"
           "auto a=read(A,m,k); auto b=read(B,k,n);\n"
           "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C);\n}", 1},
  };
  for (const auto &item : positive) {
    auto admission = fe::admitClosedRegionSource(item.source, item.name + ".mdsl");
    check(admission.syntax_valid, item.name + " Clang/Sema");
    check(admission.evidence.has_value(), item.name + " admission: " + admission.error);
    if (!admission.evidence) continue;
    const auto &program = admission.evidence->program();
    check(program.regions.size() == 1, item.name + " one selected region");
    if (program.regions.size() != 1) continue;
    checkSourceBody(item.source, program.regions[0].body);
    std::vector<const cr::Operation *> gemms;
    collect(program.regions[0].body, cr::Operation::Kind::Gemm, gemms);
    check(gemms.size() == item.gemms, item.name + " GEMM count");
    if (item.name == "lhs_rectangular" && gemms.size() == 2)
      check(gemms[1]->lhs == gemms[0]->result &&
            gemms[1]->rhs != gemms[0]->result, "lhs dependency without commutation");
    if (item.name == "rhs_rectangular_late_read" && gemms.size() == 2)
      check(gemms[1]->rhs == gemms[0]->result &&
            gemms[1]->lhs != gemms[0]->result, "rhs dependency without commutation");
    if (item.name == "dynamic_reused_template_helper" && gemms.size() == 2)
      check(!gemms[0]->helper_calls.empty() && !gemms[1]->helper_calls.empty() &&
            gemms[0]->site.offset == gemms[1]->site.offset,
            "same reusable helper body, two retained call bindings");
    if (item.name == "symbolic_shape_branches") {
      std::vector<const cr::Operation *> branches;
      collect(program.regions[0].body, cr::Operation::Kind::ShapeIf, branches);
      check(branches.size() == 1 && !branches[0]->then_body.empty() &&
            !branches[0]->else_body.empty(), "both runtime shape branches retained");
    }
    mlir::MLIRContext context;
    auto built = cr::buildModule(program, context);
    check(static_cast<bool>(built), item.name + " MLIR build: " + built.error);
    if (!built) continue;
    std::string error;
    check(cr::verifyModule(*built.module, error), item.name + " structural verification: " + error);
    check(fe::verifyClosedRegionMatchesEvidence(*admission.evidence, *built.module, error),
          item.name + " native source-paired verification: " + error);
    if (item.name == "rhs_rectangular_late_read")
      matcore::mdslc::test::checkClosedRegionAdversarialTrace(
          *admission.evidence, *built.module, check);
    mlir::Builder builder(&context);
    (*built.module)->setAttr("mdsl_admission.authority", builder.getStringAttr("generated"));
    check(!fe::verifyClosedRegionMatchesEvidence(*admission.evidence, *built.module, error),
          item.name + " forged execution authority rejected");
  }

  for (const auto &item : matcore::mdslc::test::closedRegionAdversarialSources()) {
    auto admission = fe::admitClosedRegionSource(item.source, item.name + ".mdsl");
    check(admission.syntax_valid == item.expect_syntax_valid,
          item.name + " syntax/preflight classification: " + admission.error);
    check(!admission.evidence, item.name + " must reject: " + item.obligation);
    check(!admission.error.empty(), item.name + " actionable rejection");
  }
  // A caller-chosen identity cannot substitute for the sealed bytes. Different
  // valid source with the same label must not pair with the old specimen.
  auto first = fe::admitClosedRegionSource(positive[0].source, "same-name.mdsl");
  auto second = fe::admitClosedRegionSource(positive[1].source, "same-name.mdsl");
  if (first.evidence && second.evidence) {
    mlir::MLIRContext context;
    auto built = cr::buildModule(first.evidence->program(), context);
    std::string error;
    check(built && !fe::verifyClosedRegionMatchesEvidence(
          *second.evidence, *built.module, error), "changed same-label source does not pair");
  }
  auto helperChanged = positive[2].source;
  helperChanged.replace(helperChanged.find("Numerics::strict_f32"),
                        std::string("Numerics::strict_f32").size(),
                        "Numerics::reassociate_f32");
  auto helperBefore = fe::admitClosedRegionSource(positive[2].source, "helper.mdsl");
  auto helperAfter = fe::admitClosedRegionSource(helperChanged, "helper.mdsl");
  check(helperBefore && helperAfter, "both helper numerical profiles are admitted independently");
  if (helperBefore && helperAfter) {
    mlir::MLIRContext context;
    auto built = cr::buildModule(helperBefore.evidence->program(), context);
    std::string error;
    check(built && !fe::verifyClosedRegionMatchesEvidence(
          *helperAfter.evidence, *built.module, error),
          "changed helper body cannot reuse old seal with unchanged caller spelling");
  }
  std::cout << checks << " admission checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
