#include "ClosedHostDerivedPlan.h"
#include "ClosedHostEmitter.h"
#include "ExperimentalRegionEmitter.h"
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
static_assert(!std::is_default_constructible_v<cg::ClosedHostDerivedPlan>);
static_assert(!std::is_constructible_v<cg::ClosedHostDerivedPlan, cr::Program>);
static_assert(!std::is_constructible_v<cg::ClosedHostDerivedPlan,
    std::vector<cg::ClosedHostForwardedRead>>);
static_assert(std::is_same_v<decltype(std::declval<cg::ClosedHostDerivedPlan>().forwardedReads()),
    const std::vector<cg::ClosedHostForwardedRead> &>);

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
  if (!out) throw std::runtime_error("cannot write owned plan fixture");
}
std::string region(const std::string &body, const std::string &name = "region") {
  return "#include <cstdint>\nusing namespace mdsl_probe;\n"
    "[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\nvoid " + name +
    "(Storage A,Shape m,Storage B,Shape k,Storage C,Shape n,"
    "Storage D,Shape p,Storage E,Storage F) {\n" + body + "\n}\n";
}
struct Fixture {
  support::TempDirectoryV1 directory;
  fe::Options options;
  Fixture(const std::string &clang, const std::string &resource,
          const std::string &source) {
    std::string error;
    auto temporary = support::create_temp_directory_v1("mdslc-derived-plan", error);
    if (!temporary) throw std::runtime_error(error);
    directory = std::move(*temporary);
    options.input_path = (directory.path() / "math.mdsl").string();
    options.clang_path = clang;
    options.clang_resource_directory = resource;
    options.compiler_arguments = {"-std=c++20", "-I" + directory.path().string()};
    write(options.input_path, source);
  }
  fe::ClosedRegionAdmissionResult admit(const std::string &name = "region") const {
    return fe::admitClosedRegionHost(options, directory.path().string(), name);
  }
};
cr::Id resource(const cr::Program &program, const std::string &name) {
  for (const auto &item : program.regions.front().resources)
    if (item.name == name) return item.id;
  throw std::runtime_error("missing source resource");
}
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
  if (!found) throw std::runtime_error("missing source frontier");
  return *found;
}
std::size_t occurrences(const std::string &text, const std::string &part) {
  std::size_t count = 0, position = 0;
  while ((position = text.find(part, position)) != std::string::npos) {
    ++count; position += part.size();
  }
  return count;
}
template<class Function> bool logicError(Function call) {
  try { call(); } catch (const std::logic_error &) { return true; }
  return false;
}
void sourceCase(const std::string &clang, const std::string &resource_directory,
                const std::string &body,
                const std::vector<std::pair<std::uint64_t, std::uint64_t>> &frontiers,
                const std::string &label) {
  Fixture fixture(clang, resource_directory, region(body));
  auto admitted = fixture.admit();
  check(bool(admitted), label + " source admitted: " + admitted.error);
  if (!admitted) return;
  auto issued = cg::deriveClosedHostPlan(*admitted.evidence,
      cg::ClosedHostOptimization::PublicationReadForwarding);
  check(bool(issued), label + " plan issued: " + issued.error);
  if (!issued) return;
  std::vector<cg::ClosedHostForwardedRead> expected;
  for (const auto &[read, publication] : frontiers) {
    const auto &write_operation = operationAt(admitted.evidence->program(), publication);
    check(write_operation.kind == cr::Operation::Kind::Publish,
          label + " independently numbered publication");
    expected.push_back({read, publication, write_operation.resource, write_operation.lhs});
  }
  check(issued.plan->forwardedReads() == expected, label + " exact canonical read/version/value list");
  std::string error;
  check(cg::verifyClosedHostPlan(*admitted.evidence, *issued.plan, error),
        label + " checked plan accepted: " + error);
  check(cg::verifyClosedHostForwardingProposal(*admitted.evidence,
      cg::ClosedHostOptimization::PublicationReadForwarding, expected, error),
      label + " exact untrusted proposal has no issuance authority");
  auto optimized = cg::emitClosedHostV1(*admitted.evidence, *issued.plan);
  auto baseline = cg::emitClosedHostV1(*admitted.evidence);
  check(bool(optimized) && bool(baseline), label + " both realizations emitted");
  if (optimized && baseline) {
    check(occurrences(optimized.emission->implementation, "session.readForwarded(") == expected.size(),
          label + " exactly derived reads forwarded");
    check(baseline.emission->implementation.find("readForwarded") == std::string::npos,
          label + " baseline leaves all snapshots");
    check(optimized.emission->semantic_sha256 == baseline.emission->semantic_sha256 &&
          optimized.emission->source_sha256 == baseline.emission->source_sha256 &&
          optimized.emission->frontiers.size() == baseline.emission->frontiers.size() &&
          optimized.emission->completion_frontier == baseline.emission->completion_frontier,
          label + " original witness/source/frontier count and completion unchanged");
    check(optimized.emission->plan_sha256 != baseline.emission->plan_sha256 &&
          optimized.emission->symbol != baseline.emission->symbol,
          label + " same-source optimization identity cannot co-link interchangeably");
    for (std::size_t i = 0; i < optimized.emission->frontiers.size(); ++i) {
      const auto &left = optimized.emission->frontiers[i];
      const auto &right = baseline.emission->frontiers[i];
      check(left.id == right.id && left.kind == right.kind &&
            left.source.file_id == right.source.file_id &&
            left.source.offset == right.source.offset && left.source.length == right.source.length &&
            left.source.line == right.source.line && left.source.column == right.source.column &&
            left.helper_calls.size() == right.helper_calls.size(),
            label + " every original ordered source site retained");
    }
  }
  auto none = cg::deriveClosedHostPlan(*admitted.evidence);
  check(none && none.plan->forwardedReads().empty(), label + " default policy remains unoptimized");
  if (!expected.empty()) {
    auto forged = expected;
    ++forged.front().read_frontier;
    check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
        cg::ClosedHostOptimization::PublicationReadForwarding, forged, error),
        label + " forged original read frontier rejected");
    forged = expected; ++forged.front().publication_frontier;
    check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
        cg::ClosedHostOptimization::PublicationReadForwarding, forged, error),
        label + " forged dominating version rejected");
    forged = expected; forged.front().resource = resource(admitted.evidence->program(), "F");
    check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
        cg::ClosedHostOptimization::PublicationReadForwarding, forged, error),
        label + " descriptor inequality never becomes a disjointness proof");
    forged = expected; ++forged.front().value;
    check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
        cg::ClosedHostOptimization::PublicationReadForwarding, forged, error),
        label + " forged retained value rejected");
    forged = expected; forged.pop_back();
    check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
        cg::ClosedHostOptimization::PublicationReadForwarding, forged, error),
        label + " omitted forwarding record rejected");
    forged = expected; forged.push_back(expected.front());
    check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
        cg::ClosedHostOptimization::PublicationReadForwarding, forged, error),
        label + " duplicated forwarding record rejected");
    check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
        cg::ClosedHostOptimization::None, expected, error),
        label + " no-optimization mode rejects nonempty proposal");
    if (expected.size() > 1) {
      forged = expected; std::reverse(forged.begin(), forged.end());
      check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
          cg::ClosedHostOptimization::PublicationReadForwarding, forged, error),
          label + " reordered complete list rejected");
    }
  }
  check(!cg::deriveClosedHostPlan(*admitted.evidence, static_cast<cg::ClosedHostOptimization>(-1)),
        label + " unknown optimization issuance rejected");
  check(!cg::verifyClosedHostForwardingProposal(*admitted.evidence,
      static_cast<cg::ClosedHostOptimization>(-1), {}, error),
      label + " unknown optimization verification rejected");

  auto original_plan = *issued.plan;
  auto moved_plan = std::move(original_plan);
  check(!original_plan.valid() && moved_plan.valid(), label + " moved-from plan has no authority");
  check(!cg::verifyClosedHostPlan(*admitted.evidence, original_plan, error) &&
        !cg::emitClosedHostV1(*admitted.evidence, original_plan),
        label + " moved-from consumption rejects before getters");
  check(logicError([&] { (void)original_plan.forwardedReads(); }) &&
        logicError([&] { (void)original_plan.optimization(); }) &&
        logicError([&] { (void)original_plan.identity(); }) &&
        logicError([&] { (void)original_plan.semanticIdentity(); }),
        label + " moved-from getters diagnose invalid use");
  auto old_evidence = *admitted.evidence;
  auto owned_evidence = std::move(old_evidence);
  check(!cg::deriveClosedHostPlan(old_evidence) &&
        !cg::verifyClosedHostPlan(old_evidence, moved_plan, error),
        label + " moved-from evidence is not dereferenced");
  admitted.evidence.reset();
  check(cg::verifyClosedHostPlan(owned_evidence, moved_plan, error),
        label + " plan retains immutable admission ownership");
  write(fixture.options.input_path, region(body) + "\n// new source snapshot\n");
  auto fresh = fixture.admit();
  check(fresh && !cg::verifyClosedHostPlan(*fresh.evidence, moved_plan, error) &&
        !cg::emitClosedHostV1(*fresh.evidence, moved_plan),
        label + " plan from stale source snapshot cannot authorize newly admitted source");
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  try {
    const std::string clang = argv[1], resource_directory = argv[2];
    const std::string prefix = "auto a=read(A,m,k); auto b=read(B,k,n); "
      "auto x=gemm(a,b,Numerics::strict_f32); ";
    sourceCase(clang, resource_directory,
      "auto old=read(C,m,n); " + prefix + "publish(x,C); observe(C); "
      "auto late=read(C,m,n); auto squared=gemm(late,late,Numerics::strict_f32); "
      "auto saved=gemm(old,late,Numerics::strict_f32); publish(saved,E);",
      {{7,5}}, "retained old/C*C read");
    sourceCase(clang, resource_directory, prefix +
      "publish(x,C); publish(x,D); auto stale=read(C,m,n); auto current=read(D,m,n); "
      "observe(D); publish(x,C); auto fresh=read(C,m,n);",
      {{7,5},{10,9}}, "every MAY-alias publication invalidates");
    sourceCase(clang, resource_directory, prefix +
      "publish(x,C); if(p>m) { auto blocked=read(C,m,n); publish(x,C); "
      "auto local=read(C,m,n); if(k==n) { publish(x,C); auto nested=read(C,m,n); } "
      "else { auto untaken=read(C,m,n); auto d=read(D,m,n); publish(x,D); "
      "auto local_d=read(D,m,n); } auto after=read(C,m,n); } "
      "else { publish(x,C); auto sibling=read(C,m,n); auto d=read(D,m,n); } "
      "auto join=read(C,m,n);",
      {{8,7},{11,10},{15,14},{18,17}}, "unequal nested branch frontier holes");

    const auto source = region(prefix + "publish(x,C); auto late=read(C,m,n);", "alpha") +
                        region(prefix + "publish(x,C); auto late=read(C,m,n);", "beta");
    Fixture selected(clang, resource_directory, source);
    auto alpha = selected.admit("alpha"), beta = selected.admit("beta");
    check(alpha && beta, "two selected regions share one authentic source TU");
    if (alpha && beta) {
      auto plan = cg::deriveClosedHostPlan(*alpha.evidence,
          cg::ClosedHostOptimization::PublicationReadForwarding);
      std::string error;
      check(plan && !cg::verifyClosedHostPlan(*beta.evidence, *plan.plan, error),
            "same source/host snapshot different selected function rejects foreign plan");
      auto inspection = fe::admitClosedRegionSource(
          "using namespace mdsl_probe;\n"
          "[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\n"
          "void alpha(Storage A,Shape m,Shape n) { auto a=read(A,m,n); }\n",
          "inspection.mdsl", "alpha");
      check(inspection && !cg::deriveClosedHostPlan(*inspection.evidence),
            "host-less fixture evidence cannot issue an executable derivation");
    }
  } catch (const std::exception &error) {
    check(false, std::string("unexpected plan test exception: ") + error.what());
  }
  std::cout << "Closed host derived plan: " << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
