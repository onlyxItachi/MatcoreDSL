#include "frontend.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fe = matcore::mdslc::frontend;
using EvidenceAccess = matcore::mdslc::mlir_bridge::detail::
    AuthenticatedNativeFrontendEvidenceAccessV1;

namespace {
void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void write(const std::filesystem::path &path, const std::string &contents) {
  std::ofstream stream(path, std::ios::binary);
  stream << contents;
  require(static_cast<bool>(stream), "cannot write test fixture " + path.string());
}

struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    for (unsigned attempt = 0; attempt != 100; ++attempt) {
      path = std::filesystem::temp_directory_path() /
             ("mdsl-two-gemm-frontend-" + std::to_string(tick) + "-" +
              std::to_string(attempt));
      if (std::filesystem::create_directory(path)) return;
    }
    throw std::runtime_error("cannot create isolated test directory");
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

constexpr const char *prefix =
    "#include <matcore/mdsl.h>\nnamespace md = matcore::mdsl;\n";
constexpr const char *parameters =
    "md::matrix_view &C, const md::matrix_view &A, const md::matrix_view &B, "
    "md::matrix_view &E, const md::matrix_view &D";
constexpr const char *first = "md::gemm(md::out(C), A, B);\n";
constexpr const char *second = "md::gemm(md::out(E), C, D);\n";

std::string function(const std::string &body) {
  return std::string(prefix) + "void run(" + parameters + ") {\n" + body + "}\n";
}

std::size_t admitted(const fe::Result &result) {
  return std::count_if(result.two_gemm_regions.begin(), result.two_gemm_regions.end(),
                       [](const auto &candidate) { return candidate.admitted; });
}

bool reason(const fe::Result &result, const std::string &fragment) {
  for (const auto &candidate : result.two_gemm_regions)
    for (const auto &message : candidate.rejection_reasons)
      if (message.find(fragment) != std::string::npos) return true;
  return false;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: test REPOSITORY CLANG RESOURCE_DIRECTORY PUBLIC_HEADER\n";
    return 2;
  }
  try {
    TemporaryDirectory temporary;
    fe::Options options;
    options.clang_path = argv[2];
    options.clang_resource_directory = argv[3];
    options.trusted_public_headers = {argv[4]};
    options.inspect_two_gemm_regions = true;
    options.compiler_arguments = {
        "-std=c++20", "-I" + (std::filesystem::path(argv[1]) / "compiler/include").string(),
        "-I" + temporary.path.string()};
    auto frontend = fe::createClangLibToolingFrontend();
    unsigned extraction_count = 0;
    auto extract = [&](const std::string &name, const std::string &source,
                       bool expect_success = true) {
      options.input_path = (temporary.path / (name + ".mdsl")).string();
      write(options.input_path, source);
      fe::Result result;
      const bool success = frontend->extract(options, result);
      ++extraction_count;
      if (success != expect_success) {
        std::string messages;
        for (const auto &diagnostic : result.diagnostics) messages += diagnostic.message + "\n";
        require(false, name + ": unexpected extraction status\n" + messages);
      }
      if (success && options.inspect_two_gemm_regions) {
        require(result.native_evidence && result.native_evidence->valid(), name + ": missing seal");
        require(!result.region_capture_identity.empty(), name + ": missing capture context");
        require(!result.region_dependencies.empty(), name + ": missing parsed dependency closure");
        require(!result.region_native_clang_version.empty(), name + ": missing loaded compiler version");
      }
      return result;
    };

    auto simple = extract("simple", function(std::string(first) + second));
    require(admitted(simple) == 1, "direct consecutive pair was not admitted");
    const auto &pair = simple.two_gemm_regions.front();
    require(pair.sites.size() == 2 && pair.sites[0].capture_ordinal == 0 &&
                pair.sites[1].capture_ordinal == 1, "capture order lost");
    require(pair.sites[0].bindings[0].descriptor_id == pair.sites[1].bindings[1].descriptor_id,
            "producer/consumer descriptor identity not proven");
    require(pair.sites[0].bindings[0].descriptor_id != pair.sites[1].bindings[0].descriptor_id,
            "distinct declaration bindings collapsed");
    for (unsigned stage = 0; stage != 2; ++stage)
      for (const auto &binding : pair.sites[stage].bindings)
        require(binding.snapshot_stage == stage && !binding.declaration_id.empty() &&
                    !binding.source_expression.empty(), "stage binding was not preserved");
    const auto seal = *simple.native_evidence;
    const std::string sealed_identity = simple.region_capture_identity;
    simple.two_gemm_regions.front().admitted = false;
    simple.two_gemm_regions.front().sites[0].bindings[0].descriptor_id = "forged";
    simple.region_capture_identity = "forged";
    simple.source_snapshot = "forged";
    require(admitted(EvidenceAccess::result(seal)) == 1 &&
                EvidenceAccess::result(seal).region_capture_identity == sealed_identity &&
                EvidenceAccess::result(seal).source_snapshot != "forged" &&
                EvidenceAccess::result(seal).two_gemm_regions.front().sites[0].bindings[0].descriptor_id != "forged",
            "mutable inspection result altered immutable admission evidence");
    require(EvidenceAccess::options(seal).inspect_two_gemm_regions, "opt-in not sealed");

    const auto aliases = extract("references", function(
        std::string("auto &alias = C; auto &alias2 = alias;\n") + first +
        "md::gemm(md::out(E), alias2, D);\n"));
    require(admitted(aliases) == 1, "transparent local reference chain rejected");
    const auto &alias_binding = aliases.two_gemm_regions.front().sites[1].bindings[1];
    require(alias_binding.declaration_id != alias_binding.descriptor_id,
            "reference declaration and resolved descriptor identity conflated");

    require(admitted(extract("input_overlap", function(
        std::string("md::gemm(md::out(C), A, A);\n") + second))) == 1,
        "legal input/input descriptor overlap rejected");
    require(admitted(extract("physical_overlap", std::string(prefix) +
        "void run() { float data[1]{}, other[1]{}; "
        "md::matrix_view A{data,1,1}, B{data,1,1}, C{other,1,1}, D{data,1,1}, E{data,1,1};\n" +
        first + second + "}\n")) == 1,
        "distinct bindings with unknown physical overlap were mislabeled illegal");
    const std::string qualified = std::string(prefix) +
        "namespace input { md::matrix_view A, B; }\n"
        "namespace output { md::matrix_view A, E; }\n"
        "void run() { md::gemm(md::out(output::A), input::A, input::B);\n"
        "md::gemm(md::out(output::E), output::A, input::B); }\n";
    require(admitted(extract("qualified", qualified)) == 1,
            "qualified descriptors with the same spelling collapsed");
    options.inspect_two_gemm_regions = false;
    extract("qualified_normal", qualified, false); // Historical capture remains unchanged.
    const auto ordinary = extract("ordinary", function(std::string(first) + second));
    require(!ordinary.native_evidence && ordinary.two_gemm_regions.empty() &&
                ordinary.region_capture_identity.empty(), "ordinary capture gained region authority");
    options.inspect_two_gemm_regions = true;

    require(admitted(extract("try_body", function(
        std::string("try {\n") + first + second + "} catch (...) {}\n"))) == 1,
        "direct try-body pair needed for partial-commit evidence rejected");
    for (const auto &[name, barrier] : {
             std::pair{"observer", "(void)C.data;\n"},
             std::pair{"mutation", "C.rows = 17;\n"},
             std::pair{"empty_statement", ";\n"}}) {
      const auto result = extract(name, function(std::string(first) + barrier + second));
      require(admitted(result) == 0 && reason(result, "intervening host statement"),
              std::string(name) + ": host barrier was crossed without an actionable rejection");
    }
    for (const auto &[name, body] : {
             std::pair{"conditional", std::string("if (C.rows) {\n") + first + second + "}\n"},
             std::pair{"scope", std::string("{\n") + first + "}\n{\n" + second + "}\n"},
             std::pair{"nested_scope", std::string("{\n") + first + second + "}\n"},
             std::pair{"loop", std::string("while (C.rows) {\n") + first + second + "break; }\n"}}) {
      const auto result = extract(name, function(body));
      require(admitted(result) == 0 && reason(result, "control and scope are barriers"),
              std::string(name) + ": unsupported control/scope admitted");
    }
    const auto separate = extract("functions", std::string(prefix) + "void one(" + parameters +
        ") {" + first + "}\nvoid two(" + parameters + ") {" + second + "}\n");
    require(admitted(separate) == 0 && reason(separate, "control and scope"), "cross-function pair admitted");
    const auto rhs = extract("rhs_dependence", function(std::string(first) +
        "md::gemm(md::out(E), D, C);\n"));
    require(admitted(rhs) == 0 && reason(rhs, "second GEMM lhs"), "unsupported RHS dependence admitted");
    const auto reference_call = extract("reference_call", std::string(prefix) +
        "md::matrix_view &choose(md::matrix_view &);\nvoid run(" + parameters + ") {\n"
        "auto &alias = choose(C);\n" + first + "md::gemm(md::out(E), alias, D);\n}\n");
    require(admitted(reference_call) == 0 && reason(reference_call, "transparent local reference"),
            "side-effecting reference origin resolved as a transparent binding");
    const auto reference_cast = extract("reference_cast", function(
        std::string("auto &alias = static_cast<md::matrix_view &>(C);\n") + first +
        "md::gemm(md::out(E), alias, D);\n"));
    require(admitted(reference_cast) == 0 && reason(reference_cast, "transparent local reference"),
            "reference cast admitted beyond the exact binding proof");
    const auto static_reference = extract("static_reference", function(
        std::string("static auto &alias = C;\n") + first +
        "md::gemm(md::out(E), alias, D);\n"));
    require(admitted(static_reference) == 0 && reason(static_reference, "transparent local reference"),
            "a static reference was confused with the current invocation's parameter binding");
    const auto macro_declaration = extract("macro_declaration", std::string(prefix) +
        "#define DESCRIPTOR md::matrix_view C;\n"
        "namespace producer { DESCRIPTOR }\nnamespace consumer { DESCRIPTOR }\n"
        "void run(md::matrix_view &E, const md::matrix_view &A, const md::matrix_view &B) {\n"
        "md::gemm(md::out(producer::C), A, B);\n"
        "md::gemm(md::out(E), consumer::C, B);\n}\n");
    require(admitted(macro_declaration) == 0 && reason(macro_declaration, "transparent local reference"),
            "distinct macro-expanded declarations collapsed to their common spelling location");
    const auto three = extract("three", std::string(prefix) + "void run(" + parameters +
        ", md::matrix_view &F) {\n" + first + second + "md::gemm(md::out(F), E, D);\n}\n");
    require(admitted(three) == 1 && three.two_gemm_regions.back().sites.size() == 1,
            "one source call was admitted into overlapping regions");

    const std::string user_definition = function(std::string(first) + second) +
        "namespace matcore::mdsl { void gemm(out_arg output, const matrix_view &, "
        "const matrix_view &, policy) { output.value->data[0] = 123; } }\n";
    const auto definition = extract("definition_after_calls", user_definition);
    require(admitted(definition) == 0 && reason(definition, "competing GEMM definitions"),
            "a user definition after the call sites acquired region authority");
    options.inspect_two_gemm_regions = false;
    extract("definition_normal", user_definition);
    options.inspect_two_gemm_regions = true;
    const auto user_attribute = extract("attribute_after_calls", function(std::string(first) + second) +
        "namespace matcore::mdsl { [[gnu::noinline]] void gemm(out_arg, const matrix_view &, "
        "const matrix_view &, policy); }\n");
    require(admitted(user_attribute) == 0 && reason(user_attribute, "user redeclaration attributes"),
            "user redeclaration attributes acquired ordered region authority");

    write(temporary.path / "context.h", "inline constexpr int context_extent = 1;\n");
    const std::string dependent = std::string("#include \"context.h\"\n") +
        function(std::string(first) + second);
    const auto before = extract("dependency", dependent);
    write(temporary.path / "context.h", "inline constexpr int context_extent = 2;\n");
    const auto after = extract("dependency", dependent);
    require(before.module.operations[0].site_id == after.module.operations[0].site_id &&
                before.region_capture_identity != after.region_capture_identity,
            "changed dependency bytes reused the same sealed capture context");
    options.compiler_arguments.push_back("-DMDSLC_REGION_CONTEXT_TEST=1");
    const auto changed_options = extract("dependency", dependent);
    require(changed_options.region_capture_identity != after.region_capture_identity &&
                changed_options.module.operations[0].site_id != "",
            "changed effective capture options reused the same sealed context");
    std::cout << "two-GEMM frontend admission: " << extraction_count << " extractions passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "two-GEMM frontend admission failed: " << error.what() << '\n';
    return 1;
  }
}
