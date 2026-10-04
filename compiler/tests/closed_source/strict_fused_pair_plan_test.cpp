#include "ClosedHostDerivedPlan.h"
#include "ClosedHostEmitter.h"
#include "frontend.h"
#include "../../lib/support/platform_support.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cg = matcore::mdslc::codegen;
namespace fe = matcore::mdslc::frontend;
namespace cr = matcore::mdslc::closed_region;
namespace support = matcore::mdslc::support;
namespace fs = std::filesystem;
static_assert(!std::is_constructible_v<cg::ClosedHostDerivedPlan,
    std::vector<cg::ClosedHostStrictFusedPair>>);
static_assert(std::is_same_v<decltype(std::declval<cg::ClosedHostDerivedPlan>().strictFusedPairs()),
    const std::vector<cg::ClosedHostStrictFusedPair> &>);

namespace {
unsigned checks = 0, failures = 0;
void check(bool condition, const std::string &label) {
  ++checks;
  if (!condition) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
void write(const fs::path &path, const std::string &source) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << source;
  out.close();
  if (!out) throw std::runtime_error("cannot write owned pair fixture");
}
const std::string helpers =
    "Shape width(Value value) { auto n=cols(value); return n; }\n"
    "Shape height(Value value) { auto m=rows(value); return m; }\n"
    "Value product(Value a,Value b) { auto c=gemm(a,b,Numerics::strict_f32); return c; }\n";
std::string source(const std::string &body, const std::string &name = "region") {
  return "#include <cstdint>\nusing namespace mdsl_probe;\n" + helpers +
    "[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\nvoid " + name +
    "(Storage A,Storage B,Storage D,Storage E,Shape m,Shape k,Shape n,Shape p) {\n" +
    body + "\n}\n";
}
struct Fixture {
  support::TempDirectoryV1 directory;
  fe::Options options;
  Fixture(const std::string &clang, const std::string &resource, const std::string &bytes) {
    std::string error;
    auto temporary = support::create_temp_directory_v1("mdslc-strict-pair-plan", error);
    if (!temporary) throw std::runtime_error(error);
    directory = std::move(*temporary);
    options.input_path = (directory.path() / "math.mdsl").string();
    options.clang_path = clang;
    options.clang_resource_directory = resource;
    options.compiler_arguments = {"-std=c++20", "-I" + directory.path().string()};
    write(options.input_path, bytes);
  }
  fe::ClosedRegionAdmissionResult admit(const std::string &name = "region") const {
    return fe::admitClosedRegionHost(options, directory.path().string(), name);
  }
};
const cr::Operation &operationAt(const cr::Program &program, std::uint64_t wanted) {
  const cr::Operation *found = nullptr;
  std::uint64_t frontier = 0;
  const auto visit = [&](auto &&self, const std::vector<cr::Operation> &body) -> void {
    for (const auto &operation : body) {
      if (++frontier == wanted) found = &operation;
      if (operation.kind == cr::Operation::Kind::ShapeIf) {
        self(self, operation.then_body);
        self(self, operation.else_body);
      }
    }
  };
  visit(visit, program.regions.front().body);
  if (!found) throw std::runtime_error("missing independently numbered pair frontier");
  return *found;
}
std::size_t occurrences(const std::string &bytes, const std::string &part) {
  std::size_t count = 0, position = 0;
  while ((position = bytes.find(part, position)) != std::string::npos) {
    ++count; position += part.size();
  }
  return count;
}
template<class Function> bool logicError(Function call) {
  try { call(); } catch (const std::logic_error &) { return true; }
  return false;
}
void testCase(const std::string &clang, const std::string &resource,
              const std::string &body, const std::vector<std::uint64_t> &starts,
              const std::string &label) {
  Fixture fixture(clang, resource, source(body));
  auto admitted = fixture.admit();
  check(bool(admitted), label + " authentic source admission: " + admitted.error);
  if (!admitted) return;
  auto baseline = cg::deriveClosedHostPlan(*admitted.evidence);
  auto issued = cg::deriveClosedHostPlan(*admitted.evidence, cg::ClosedHostOptimization::StrictFusedPair);
  check(bool(baseline), label + " unchanged baseline remains admissible");
  if (starts.empty()) {
    check(!issued && issued.error.find("eligible adjacent") != std::string::npos,
          label + " zero-eligible explicit mode rejects, never falls back");
    std::string error;
    check(!cg::verifyClosedHostFusedPairProposal(*admitted.evidence,
        cg::ClosedHostOptimization::StrictFusedPair, {}, error),
        label + " empty proposal cannot bypass explicit no-pair rejection");
    return;
  }
  check(bool(issued), label + " checked pair issued: " + issued.error);
  if (!issued || !baseline) return;
  std::vector<cg::ClosedHostStrictFusedPair> expected;
  for (const auto first : starts) {
    const auto &producer = operationAt(admitted.evidence->program(), first);
    const auto &consumer = operationAt(admitted.evidence->program(), first+1);
    cg::ClosedHostStrictFusedPair pair{first,first+1,producer.lhs,producer.rhs,
        consumer.rhs,producer.result,consumer.result};
    for (std::uint64_t read = 1; read < first; ++read) {
      const auto &definition = operationAt(admitted.evidence->program(), read);
      if (definition.kind != cr::Operation::Kind::Read) continue;
      if (definition.result == pair.a) pair.a_read_frontier = read;
      if (definition.result == pair.b) pair.b_read_frontier = read;
      if (definition.result == pair.d) pair.d_read_frontier = read;
    }
    expected.push_back(pair);
  }
  check(issued.plan->strictFusedPairs() == expected && issued.plan->forwardedReads().empty(),
        label + " complete canonical original pair/read-provenance list");
  std::string error;
  check(cg::verifyClosedHostPlan(*admitted.evidence, *issued.plan, error),
        label + " source-paired checked plan accepted: " + error);
  check(cg::verifyClosedHostFusedPairProposal(*admitted.evidence,
      cg::ClosedHostOptimization::StrictFusedPair, expected, error),
      label + " diagnostic proposal equality confers no issuer authority");
  auto optimized = cg::emitClosedHostV1(*admitted.evidence, *issued.plan);
  auto original = cg::emitClosedHostV1(*admitted.evidence, *baseline.plan);
  check(optimized && original, label + " both host realizations emitted");
  if (optimized && original) {
    check(occurrences(optimized.emission->implementation, "session.gemmStrictFusedPair(") == starts.size(),
          label + " one private call per checked pair");
    check(original.emission->implementation.find("gemmStrictFusedPair") == std::string::npos,
          label + " baseline calls original GEMMs");
    check(optimized.emission->semantic_sha256 == original.emission->semantic_sha256 &&
          optimized.emission->source_sha256 == original.emission->source_sha256 &&
          optimized.emission->completion_frontier == original.emission->completion_frontier &&
          optimized.emission->frontiers.size() == original.emission->frontiers.size(),
          label + " original Program/witness and full frontier ledger unchanged");
    check(optimized.emission->plan_sha256 != original.emission->plan_sha256 &&
          optimized.emission->symbol != original.emission->symbol,
          label + " optimization helper ownership cannot be interchanged");
    for (std::size_t i = 0; i < original.emission->frontiers.size(); ++i) {
      const auto &a = optimized.emission->frontiers[i];
      const auto &b = original.emission->frontiers[i];
      check(a.id == b.id && a.kind == b.kind && a.source.file_id == b.source.file_id &&
            a.source.offset == b.source.offset && a.source.length == b.source.length &&
            a.source.line == b.source.line && a.source.column == b.source.column &&
            a.helper_calls.size() == b.helper_calls.size(),
            label + " exact source site and branch-hole enumeration");
      for (std::size_t j=0;j<a.helper_calls.size();++j) {
        const auto &x=a.helper_calls[j], &y=b.helper_calls[j];
        check(x.file_id==y.file_id && x.offset==y.offset && x.length==y.length &&
              x.line==y.line && x.column==y.column,label+" complete original helper call ledger");
      }
    }
    for (const auto &pair : expected)
      check(optimized.emission->implementation.find(
            "mch::Value value_" + std::to_string(pair.intermediate) + ";") == std::string::npos,
            label + " no materialized intermediate Value or uncounted dimension use");
  }
  auto forged = expected;
  const auto reject = [&](const auto &proposal, const std::string &what) {
    check(!cg::verifyClosedHostFusedPairProposal(*admitted.evidence,
        cg::ClosedHostOptimization::StrictFusedPair, proposal, error), label + " rejects " + what);
  };
  for (unsigned field = 0; field < 10; ++field) {
    forged = expected;
    auto &pair = forged.front();
    switch (field) {
    case 0: ++pair.first_frontier; break;
    case 1: ++pair.second_frontier; break;
    case 2: ++pair.a; break;
    case 3: ++pair.b; break;
    case 4: ++pair.d; break;
    case 5: ++pair.intermediate; break;
    case 6: ++pair.result; break;
    case 7: ++pair.a_read_frontier; break;
    case 8: ++pair.b_read_frontier; break;
    case 9: ++pair.d_read_frontier; break;
    }
    reject(forged, "forged pair/provenance field " + std::to_string(field));
  }
  forged = expected; forged.pop_back(); reject(forged, "omitted record");
  forged = expected; forged.push_back(expected.front()); reject(forged, "duplicate record");
  if (expected.size() > 1) {
    forged = expected; std::reverse(forged.begin(), forged.end()); reject(forged, "reordered records");
  }
  check(!cg::verifyClosedHostFusedPairProposal(*admitted.evidence,
      cg::ClosedHostOptimization::None, expected, error) &&
        !cg::verifyClosedHostFusedPairProposal(*admitted.evidence,
      cg::ClosedHostOptimization::PublicationReadForwarding, expected, error) &&
        !cg::verifyClosedHostFusedPairProposal(*admitted.evidence,
      static_cast<cg::ClosedHostOptimization>(-1), expected, error),
        label + " modes are explicit and do not acquire foreign plan authority");
  auto moved_from = *issued.plan;
  auto moved = std::move(moved_from);
  check(!moved_from.valid() && moved.valid() &&
        !cg::verifyClosedHostPlan(*admitted.evidence, moved_from, error) &&
        logicError([&] { (void)moved_from.strictFusedPairs(); }),
        label + " moved-from plan is invalid before getter/consumption");
  const auto retained = *admitted.evidence;
  admitted.evidence.reset();
  check(cg::verifyClosedHostPlan(retained, moved, error), label + " plan owns original evidence");
  write(fixture.options.input_path, source(body) + "\n// changed snapshot\n");
  auto fresh = fixture.admit();
  check(fresh && !cg::verifyClosedHostPlan(*fresh.evidence, moved, error),
        label + " stale exact-source plan rejected");
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  try {
    const std::string clang = argv[1], resource = argv[2];
    const std::string reads = "auto a=read(A,m,k);auto b=read(B,k,n);auto d=read(D,n,p);";
    const std::string producer = "auto c=gemm(a,b,Numerics::strict_f32);";
    const std::string consumer = "auto e=gemm(c,d,Numerics::strict_f32);";
    testCase(clang,resource,reads+producer+consumer+"publish(e,E);observe(E);",{4},"pure lhs chain");
    testCase(clang,resource,reads+"auto c=product(a,b);auto e=product(c,d);publish(e,E);",{4},
        "pure helper-expanded pair retains both helper source ledgers");
    testCase(clang,resource,reads+producer+consumer+
        "auto c2=gemm(a,b,Numerics::strict_f32);auto e2=gemm(c2,d,Numerics::strict_f32);"
        "publish(e2,E);",{4,6},"multiple independent nonoverlapping straight-line pairs");
    testCase(clang,resource,reads+"if(m==p){observe(A);}else{observe(B);observe(D);}"+
        producer+consumer+"publish(e,E);",{8},"pair continuation after unequal branch frontier holes");
    testCase(clang,resource,reads+producer+
        "auto dead=rows(c);auto dead_helper=width(c);"+consumer+"publish(e,E);",{4},
        "dead pure dimension queries have no frontier or retained semantic action");
    testCase(clang,resource,reads+producer+consumer+
        "auto width_c=cols(c);auto q=read(E,m,width_c);",{},"live read dimension");
    testCase(clang,resource,reads+producer+consumer+
        "auto width_c=width(c);auto q=read(E,m,width_c);",{},"live helper dimension");
    testCase(clang,resource,reads+producer+consumer+
        "auto height_c=height(c);if(height_c==m){observe(E);}else{observe(A);}",{},
        "live helper branch condition");
    testCase(clang,resource,reads+producer+consumer+
        "if(m==p){observe(E);}else{auto width_c=cols(c);auto q=read(E,m,width_c);}",{},
        "retained dimension in otherwise untaken arm");
    testCase(clang,resource,reads+producer+"publish(c,E);"+consumer,{},"publication barrier");
    testCase(clang,resource,reads+producer+"observe(E);"+consumer,{},"observation barrier");
    testCase(clang,resource,reads+producer+"auto q=read(E,m,n);"+consumer,{},"late read barrier");
    testCase(clang,resource,reads+producer+"auto e=gemm(d,c,Numerics::strict_f32);",{},"rhs carry not eligible");
    testCase(clang,resource,reads+producer+"auto e=gemm(c,c,Numerics::strict_f32);",{},"C*C not eligible");
    testCase(clang,resource,reads+producer+consumer+"auto q=gemm(c,d,Numerics::strict_f32);",{},
        "multiple semantic intermediate consumers");
    testCase(clang,resource,reads+"auto c=gemm(a,b,Numerics::reassociate_f32);"+consumer,{},
        "first permission is not strict");
    testCase(clang,resource,reads+producer+"auto e=gemm(c,d,Numerics::reassociate_f32);",{},
        "second permission is not strict");
    testCase(clang,resource,"auto a=read(A,m,k);auto b=read(B,k,n);"+producer+
        "auto d=read(D,n,p);"+consumer,{},"external consumer input read after producer");
    testCase(clang,resource,"auto a=read(A,m,k);auto b=read(B,k,n);"
        "auto d=gemm(a,b,Numerics::strict_f32);"+producer+consumer,{},"computed external input");
    testCase(clang,resource,reads+"if(m==p){"+producer+consumer+"publish(e,E);"
        "if(k==n){observe(E);}else{observe(A);observe(B);}}else{"
        +producer+consumer+"publish(e,E);}",{5,12},"unequal nested arm frontier holes");

    auto second_source=source(reads+producer+consumer,"beta");
    second_source.erase(second_source.find(helpers),helpers.size());
    Fixture selected(clang,resource,source(reads+producer+consumer,"alpha")+second_source);
    auto alpha=selected.admit("alpha"), beta=selected.admit("beta");
    check(alpha && beta,"two selected regions under same source snapshot");
    if(alpha && beta) {
      auto plan=cg::deriveClosedHostPlan(*alpha.evidence,cg::ClosedHostOptimization::StrictFusedPair);
      std::string error;
      check(plan && !cg::verifyClosedHostPlan(*beta.evidence,*plan.plan,error),
            "selected-function identity prevents foreign plan replay");
    }
  } catch (const std::exception &error) {
    check(false,std::string("unexpected strict pair plan exception: ")+error.what());
  }
  std::cout << "Strict fused pair plan: " << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
