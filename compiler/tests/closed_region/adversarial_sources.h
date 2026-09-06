#ifndef MDSLC_CLOSED_REGION_ADVERSARIAL_SOURCES_H
#define MDSLC_CLOSED_REGION_ADVERSARIAL_SOURCES_H

#include <string>
#include <utility>
#include <vector>

namespace matcore::mdslc::test {

// Each source is valid C++20 with the private injected fixture declarations.
// The system-header and preprocessing cases intentionally fail the isolated
// frontend's no-directives preflight; every other case must reach successful Clang Sema
// before admission rejects it. A syntax/preflight failure is not evidence that
// the admission checker rejected hidden C++ behavior.
struct ClosedRegionAdversarialSource {
  std::string name;
  std::string source;
  std::string obligation;
  bool expect_syntax_valid = true;
};

inline std::vector<ClosedRegionAdversarialSource>
closedRegionAdversarialSources() {
  const std::string parameters =
      "Storage A, Storage B, Storage C, Storage D, Storage E, "
      "Shape m, Shape k, Shape n";
  const std::string inputs =
      "auto a = read(A,m,k); auto b = read(B,k,n);\n";
  std::vector<ClosedRegionAdversarialSource> cases;
  auto add = [&](std::string name, std::string declarations, std::string body,
                 std::string obligation, bool expect_syntax_valid = true) {
    cases.push_back({
        std::move(name),
        "using namespace mdsl_probe;\n" + declarations +
            "\n[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\n"
            "void region(" + parameters + ") {\n" + inputs + body + "\n}\n",
        std::move(obligation), expect_syntax_valid});
  };

  add("unknown_host_call", "void host();", "host();",
      "Unknown direct calls must not acquire mathematical purity.");
  add("iostream", "#include <iostream>", "std::cout << m;",
      "Host header preflight must fail before admission, not count as semantic closure.", false);
  add("file_output", "#include <cstdio>", "std::puts(\"observable\");",
      "Host header preflight must fail before admission, not count as semantic closure.", false);
  add("stream_operator",
      "namespace host_io { struct Stream {}; extern Stream output; "
      "Stream &operator<<(Stream &, Shape); }", "host_io::output << m;",
      "Overloaded stream operators are externally observable host effects.");
  add("file_output_declaration", "extern \"C\" int puts(const char *);", "puts(\"observable\");",
      "A valid external IO declaration does not enter the mathematical vocabulary.");
  add("network_call", "extern int send(int, const void *, unsigned long, int);",
      "send(0, &m, sizeof(m), 0);",
      "A declaration without a modeled effect contract cannot authorize IO.");
  add("system_call", "#include <cstdlib>", "std::system(\"true\");",
      "Host header preflight must fail before admission, not count as semantic closure.", false);
  add("system_call_declaration", "extern \"C\" int system(const char *);", "system(\"true\");",
      "No source execution or host process effect is permitted during admission.");
  add("implicit_conversion",
      "void host(); struct Convert { operator Shape() const { host(); return 1ULL; } };",
      "Shape extent = Convert{}; auto x = read(D,extent,n);",
      "Implicit user conversions must not be erased as harmless casts.");
  add("effectful_constructor",
      "void host(); struct Object { Object() { host(); } };", "Object object;",
      "A constructor is executable behavior even if the object is unused.");
  add("effectful_destructor",
      "void host(); struct Cleanup { ~Cleanup() { host(); } };", "Cleanup cleanup;",
      "Unused local cleanup is not mathematical handle bookkeeping.");
  add("cleanup_attribute", "void cleanup(Shape *);",
      "Shape extent __attribute__((cleanup(cleanup))) = m;",
      "An attributed scalar can carry an otherwise hidden exit effect.");
  add("nontrivial_helper_parameter",
      "void host(); struct Box { Box(Value) {} Box(const Box &) { host(); } }; "
      "Value helper(Box, Value v) { return v; }",
      "Box box(a); auto c = helper(box,a); publish(c,C);",
      "A helper's apparently pure body does not discharge parameter copying.");
  add("helper_default_argument",
      "Shape host_shape(); Value helper(Value v, Shape extent = host_shape()) { return v; }",
      "auto c = helper(a); publish(c,C);",
      "Default arguments are evaluated at the call and must not disappear.");
  add("helper_hidden_effect", "void host(); Value helper(Value v) { host(); return v; }",
      "auto c = helper(a); publish(c,C);",
      "Every called helper body must be checked recursively.");
  add("template_hidden_effect",
      "void host(); template<class T> Value helper(T v) { host(); return v; }",
      "auto c = helper(a); publish(c,C);",
      "Template instantiation does not authenticate hidden host effects.");
  add("template_selected_effect",
      "void host(); template<bool Effect> Value helper(Value v) { "
      "if constexpr (Effect) host(); return v; }",
      "auto c = helper<true>(a); publish(c,C);",
      "The instantiated effectful branch must not inherit a template purity label.");
  add("recursive_helper", "Value helper(Value v) { return helper(v); }",
      "auto c = helper(a); publish(c,C);",
      "Unbounded recursion/nontermination is not admitted by a body marker.");
  add("mutual_recursive_helpers",
      "Value second(Value); Value first(Value v) { return second(v); } "
      "Value second(Value v) { return first(v); }",
      "auto c = first(a); publish(c,C);",
      "Recursion detection must cover the transitive helper call graph.");
  add("escaped_value_address", "void escape(void *);", "escape(&a);",
      "Logical value handles cannot escape into mutable host memory.");
  add("escaped_storage_address", "void escape(void *);", "escape(&A);",
      "Resource bindings do not authorize escaped mutable host aliases.");
  add("pointer_local", "", "auto *pointer = &a;",
      "Pointer formation is outside the value-only local grammar.");
  add("reference_local", "", "auto &alias = a;",
      "Mutable C++ reference identity must not stand in for immutable value identity.");
  add("helper_reference_escape",
      "Value *escaped; Value helper(Value &v) { escaped = &v; return v; }",
      "auto c = helper(a); publish(c,C);",
      "Helper signatures and bodies must not widen resource ownership.");
  add("volatile_read", "volatile Shape hardware_extent = 1ULL;",
      "Shape extent = hardware_extent; auto x = read(D,extent,n);",
      "Lvalue-to-rvalue conversion may itself be an observable volatile read.");
  add("volatile_local", "", "volatile Shape extent = m;",
      "Volatile qualification is not an ordinary shape binding.");
  add("atomic_access", "#include <atomic>\nstd::atomic<Shape> state{1ULL};",
      "auto extent = state.load(); auto x = read(D,extent,n);",
      "Host header preflight must fail before admission, not count as semantic closure.", false);
  add("atomic_builtin", "", "Shape state = m; __atomic_fetch_add(&state,1ULL,5);",
      "Builtin atomics expose synchronization even without a standard-library header.");
  add("throw", "", "throw 7;",
      "Unchecked C++ exception frontiers cannot enter mathematical effects.");
  add("try_catch", "", "try { auto c = gemm(a,b,Numerics::strict_f32); } catch (...) {}",
      "Host exception interception cannot rewrite the checked-failure contract.");
  add("virtual_dispatch", "struct Interface { virtual void run() = 0; }; Interface *target;",
      "target->run();",
      "Virtual dispatch has no canonical mathematical implementation identity.");
  add("indirect_call", "void host();", "auto call = &host; call();",
      "Indirect calls cannot bypass direct canonical-declaration authentication.");
  add("lambda_effect", "void host();", "auto callback = [] { host(); }; callback();",
      "A lambda body is not automatically a closed mathematical helper.");
  add("lambda_capture_effect", "Shape host_shape();",
      "auto callback = [extent = host_shape()] { return extent; };",
      "Capture initialization executes before any apparent callback invocation.");
  add("static_local", "", "static auto saved = a;",
      "Static state has cross-invocation identity and initialization semantics.");
  add("thread_local", "", "thread_local auto saved = a;",
      "Thread-local persistence must not become a fresh SSA value.");
  add("dead_host_effect", "void host();", "if (false) { host(); }",
      "Closure is a source admission rule, not a lucky dead-code-elimination result.");
  add("shape_condition_effect", "void host();",
      "if ((host(),m == n)) { publish(a,C); } else { publish(a,E); }",
      "A scalar-looking condition cannot conceal a comma-sequenced host effect.");
  add("short_circuit_effect", "bool host();",
      "if (m == n && host()) { publish(a,C); } else { publish(a,E); }",
      "Short-circuit evaluation does not make its conditional host call pure.");
  add("nested_checked_arguments", "",
      "auto c = gemm(read(A,m,k),read(B,k,n),Numerics::strict_f32); publish(c,C);",
      "Indeterminately sequenced argument checks cannot acquire invented source order.");
  add("assignment", "", "a = b; publish(a,C);",
      "Mutable handle assignment is not an admitted immutable binding.");
  add("numeric_policy_cast", "",
      "auto c = gemm(a,b,static_cast<Numerics>(1)); publish(c,C);",
      "Numerical permissions require canonical enum identity, not an integer cast.");
  add("low_precision_implicit", "", "_Float16 scalar = 0.1f;",
      "Unsupported low-precision rounding must not inherit host implicit conversions.");
  add("user_literal", "void host(); Shape operator\"\"_extent(unsigned long long n) { host(); return n; }",
      "auto x = read(D,1_extent,n);",
      "User-defined numeric literal syntax is an executable function call.");
  add("forged_operation",
      "namespace fake { [[clang::annotate(\"mdsl.private.gemm.v1\")]] "
      "Value gemm(Value,Value,Numerics) noexcept; }",
      "auto c = fake::gemm(a,b,Numerics::strict_f32); publish(c,C);",
      "An annotation on a foreign declaration grants no primitive authority.");
  add("user_primitive_definition",
      "namespace mdsl_probe { Value gemm(Value lhs,Value,Numerics) noexcept { return lhs; } }",
      "auto c = gemm(a,b,Numerics::strict_f32); publish(c,C);",
      "A competing definition must invalidate the canonical primitive binding.");
  add("user_primitive_redeclaration",
      "namespace mdsl_probe { [[gnu::pure]] Value gemm(Value,Value,Numerics) noexcept; }",
      "auto c = gemm(a,b,Numerics::strict_f32); publish(c,C);",
      "Merged user attributes must not alter a trusted primitive's effects.");
  add("primitive_definition_after_region", "",
      "auto c = gemm(a,b,Numerics::strict_f32); publish(c,C);",
      "Whole-TU authentication must see definitions after the selected body.");
  cases.back().source +=
      "namespace mdsl_probe { Value gemm(Value lhs,Value,Numerics) noexcept { return lhs; } }\n";
  add("helper_definition_after_region", "Value helper(Value);",
      "auto c = helper(a); publish(c,C);",
      "A later helper definition must be inspected rather than trusted as an opaque call.");
  cases.back().source += "void host(); Value helper(Value v) { host(); return v; }\n";
  add("helper_attribute_after_region",
      "Value helper(Value lhs,Value rhs) { return gemm(lhs,rhs,Numerics::strict_f32); }",
      "auto c = helper(a,b); publish(c,C);",
      "Complete generic redeclaration traversal must see later helper attributes.");
  cases.back().source +=
      "[[clang::annotate(\"mdsl.private.forged_helper_authority\")]] Value helper(Value,Value);\n";
  add("purity_attribute_host_call", "[[gnu::const]] Shape host_shape();",
      "auto x = read(D,host_shape(),n);",
      "A host declaration's optimization attribute does not create semantic authority.");
  add("purity_attribute_helper",
      "[[gnu::const]] Value helper(Value lhs, Value rhs) { "
      "return gemm(lhs,rhs,Numerics::strict_f32); }",
      "auto c = helper(a,b); publish(c,C);",
      "Host purity attributes cannot erase the helper's ordered checked failures.");
  add("overloaded_selected_region", "", "publish(a,C);",
      "Selection by source name must reject ambiguous overloads rather than choosing one.");
  cases.back().source +=
      "[[clang::annotate(\"mdsl.private.closed_region.v1\")]] "
      "void region(Storage A, Shape m) { auto a = read(A,m,m); observe(A); }\n";
  add("global_value_read", "extern Value global;", "publish(global,C);",
      "A global object is not an authenticated region input or value binding.");
  add("assembly", "", "asm volatile(\"\" ::: \"memory\");",
      "Inline assembly is an opaque host/machine effect.");
  add("unbounded_loop", "", "while (m != 0ULL) { publish(a,C); }",
      "Unbounded host looping is outside bounded shape-dependent selection.");
  add("goto", "", "goto done; done: publish(a,C);",
      "Nonstructured source control must not bypass ordered checks or observations.");
  add("unevaluated_call", "void host();", "auto flag = noexcept(host());",
      "Unevaluated C++ introspection is not a mathematical shape expression.");
  add("decltype_hidden_call", "void host();",
      "decltype((host(),Shape{})) extent = m; auto x = read(D,extent,n);",
      "Unevaluated type expressions require explicit staging, not incidental canonical-type acceptance.");
  add("typeof_hidden_call", "void host();",
      "__typeof__((host(),Shape{})) extent = m; auto x = read(D,extent,n);",
      "GNU typeof inside the closed body is outside the declared type-spelling subset.");
  add("helper_result_decltype", "void host(); "
      "decltype((host(),Shape{})) helper(Shape n) { return n; }",
      "auto extent = helper(m); auto x = read(D,extent,n);",
      "Expression-bearing type spelling in helper signatures needs the same admission rule.");
  add("builtin_source_macro", "", "auto x = read(D,__LINE__,n);",
      "Compiler builtin macro expansion is not an unmarked mathematical shape literal.");
  add("digraph_directive", "%:define UNUSED 1", "publish(a,C);",
      "The preprocessing digraph cannot bypass no-directive preflight.", false);
  add("pragma_operator", "_Pragma(\"clang fp contract(fast)\")", "publish(a,C);",
      "Pragma operators can alter compiler state without an AST operation.", false);
  add("spliced_digraph_directive", std::string("%") + '\\' + "\n:define UNUSED 1",
      "publish(a,C);",
      "Preprocessing token cleaning must not hide a spliced directive.", false);
  return cases;
}

} // namespace matcore::mdslc::test

#endif
