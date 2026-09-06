#ifndef MDSLC_CLOSED_REGION_HOST_ADVERSARIAL_SOURCES_H
#define MDSLC_CLOSED_REGION_HOST_ADVERSARIAL_SOURCES_H

#include <string>
#include <utility>
#include <vector>

namespace matcore::mdslc::test {

// Source-only catalogue. The harness owns temporary files and authenticated
// compiler/include context; none of these host programs is ever executed.
// Auxiliary paths are relative to the per-case source directory.
struct ClosedRegionHostSource {
  std::string name;
  std::string source;
  std::vector<std::pair<std::string, std::string>> headers;
  bool expect_syntax_valid = true;
  bool expect_admission = false;
  std::string obligation;
};

inline std::vector<ClosedRegionHostSource> closedRegionHostSources() {
  const std::string marker =
      "[[clang::annotate(\"mdsl.private.closed_region.v1\")]]\n";
  const std::string signature =
      "void region(Storage A, Storage B, Storage C, Storage D, Storage E, "
      "Shape m, Shape k, Shape n)";
  const std::string inputs =
      "auto a=read(A,m,k); auto b=read(B,k,n);\n";
  const std::string math =
      "auto c=gemm(a,b,Numerics::strict_f32); publish(c,C); observe(C);\n"
      "auto d=read(D,n,m); auto e=gemm(c,d,Numerics::strict_f32); publish(e,E);";
  std::vector<ClosedRegionHostSource> cases;
  auto add = [&](std::string name, std::string prefix, std::string body,
                 bool admitted, std::string obligation,
                 std::vector<std::pair<std::string, std::string>> headers = {}) {
    cases.push_back({std::move(name), std::move(prefix) +
                        "\nusing namespace mdsl_probe;\n" + marker + signature +
                        " {\n" + inputs + body + "\n}\n",
                     std::move(headers), true, admitted, std::move(obligation)});
  };

  add("host_iostream_raii_macro_outside",
      "#include <iostream>\n#include <string>\n"
      "#define HOST_PRINT(x) (std::cout << (x))\n"
      "struct HostGuard { ~HostGuard() { HOST_PRINT(\"cleanup\"); } };\n"
      "void host_work() { HostGuard guard; std::string message=\"outside\"; "
      "HOST_PRINT(message); }\n", math, true,
      "Real standard headers, host IO, RAII and macro expansion outside the selected region remain host semantics.");
  add("host_local_alias_reused_main_helper",
      "#include \"host_types.h\"\n"
      "using namespace mdsl_probe;\n"
      "template<class T> T product(T lhs,T rhs) { "
      "auto v=gemm(lhs,rhs,Numerics::strict_f32); return v; }\n",
      "HostExtent extent=n; auto c=product(a,b); publish(c,C);\n"
      "auto d=read(D,extent,m); auto e=product(c,d); publish(e,E);", true,
      "Header aliases may stage exact canonical types; reused mathematical helper bodies remain main-source owned.",
      {{"host_types.h", "#pragma once\nusing HostExtent=mdsl_probe::Shape;\n"}});
  add("host_transitive_header_effects_outside", "#include \"outer.h\"\n"
      "void host_work() { HostLifetime lifetime; host_network(); }\n", math, true,
      "Transitive dependency snapshots may contain unrelated host effects without importing them into a region.",
      {{"outer.h", "#pragma once\n#include \"inner.h\"\n"},
       {"inner.h", "#pragma once\nvoid host_network();\n"
                   "struct HostLifetime { ~HostLifetime() { host_network(); } };\n"}});
  add("host_optional_include_absent", "#if __has_include(\"optional.h\")\n"
      "#include \"optional.h\"\n#else\nusing OptionalExtent=mdsl_probe::Shape;\n#endif\n",
      "OptionalExtent extent=n; auto c=gemm(a,b,Numerics::strict_f32);\n"
      "auto d=read(D,extent,m); auto e=gemm(c,d,Numerics::strict_f32); publish(e,E);", true,
      "Absent include lookup is host staging whose negative lookup must be frozen for replay.");
  add("host_optional_include_present", "#if __has_include(\"optional.h\")\n"
      "#include \"optional.h\"\n#endif\n", math, true,
      "Positive optional-header lookup and bytes must participate in the captured context.",
      {{"optional.h", "#pragma once\nusing OptionalExtent=mdsl_probe::Shape;\n"}});
  add("host_optional_lookup_without_read", "#if !__has_include(\"presence_only.h\")\n"
      "#error required presence-only dependency is missing\n#endif\n", math, true,
      "A successful lookup without a file read still affects source context and requires deterministic replay.",
      {{"presence_only.h", "This file deliberately is not valid C++ and must never be included.\n"}});

  add("host_date_macro_outside", "const char host_build_date[]=__DATE__;\n", math, false,
      "A volatile builtin outside the region still makes full-TU replay depend on unsealed wall-clock state.");
  add("host_time_macro_outside", "const char host_build_time[]=__TIME__;\n", math, false,
      "Time-dependent preprocessing must not be silently regenerated during source-evidence replay.");
  add("host_timestamp_macro_outside", "const char host_source_time[]=__TIMESTAMP__;\n", math, false,
      "File timestamp preprocessing requires explicit captured authority or rejection, not live metadata reuse.");

  add("host_iostream_inside", "#include <iostream>\n", "std::cout << m;", false,
      "Making iostream available does not admit overloaded observable calls inside the region.");
  add("host_file_inside", "#include <cstdio>\n", "std::puts(\"inside\");", false,
      "Real file/stream declarations do not acquire mathematical or ordered publication authority.");
  add("host_network_header_call", "#include \"network.h\"\n", "host_network();", false,
      "A host effect declared in a captured dependency remains an unknown region call.",
      {{"network.h", "#pragma once\nvoid host_network();\n"}});
  add("host_raii_header_inside", "#include \"lifetime.h\"\n", "HostLifetime lifetime;", false,
      "Header-origin constructors/destructors cannot create hidden cleanup within the region.",
      {{"lifetime.h", "#pragma once\nvoid host_cleanup();\n"
                      "struct HostLifetime { HostLifetime() { host_cleanup(); } "
                      "~HostLifetime() { host_cleanup(); } };\n"}});
  add("host_header_helper", "#include \"helper.h\"\n",
      "auto c=header_product(a,b); publish(c,C);", false,
      "This boundary does not expand helper ownership to header-defined bodies.",
      {{"helper.h", "#pragma once\ninline mdsl_probe::Value header_product("
                    "mdsl_probe::Value a,mdsl_probe::Value b) { "
                    "return mdsl_probe::gemm(a,b,mdsl_probe::Numerics::strict_f32); }\n"}});
  add("host_header_template_helper", "#include \"template_helper.h\"\n",
      "auto c=header_product(a,b); publish(c,C);", false,
      "Clang instantiation does not make a header helper main-source-owned.",
      {{"template_helper.h", "#pragma once\n"
                             "template<class T> T header_product(T a,T b) { "
                             "return mdsl_probe::gemm(a,b,mdsl_probe::Numerics::strict_f32); }\n"}});

  add("host_empty_macro_body", "#define ERASED\n", "ERASED\n" + math, false,
      "An empty macro inside the region is not visible in the AST but remains an unadmitted source construct.");
  add("host_erased_effect_argument", "void host_effect();\n#define ERASE(...)\n",
      "ERASE(host_effect())\n" + math, false,
      "Discarded macro arguments cannot evade the closed-source vocabulary merely because no AST node survives.");
  add("host_macro_call", "#define PRODUCT mdsl_probe::gemm\n",
      "auto c=PRODUCT(a,b,Numerics::strict_f32); publish(c,C);", false,
      "Canonical callee identity alone does not authenticate macro-owned source spelling.");
  add("host_macro_numerics", "#define NUMERIC mdsl_probe::Numerics::strict_f32\n",
      "auto c=gemm(a,b,NUMERIC); publish(c,C);", false,
      "Numerical-profile source ownership remains explicit.");
  add("host_macro_type", "#define EXTENT mdsl_probe::Shape\n",
      "EXTENT extent=m; auto d=read(D,extent,n);", false,
      "A macro-expanded type spelling cannot disappear behind exact canonical type equality.");
  add("host_empty_macro_type", "#define EMPTY\n", "Shape EMPTY extent=m;\n" + math, false,
      "Token-level range admission must detect empty macros inside a local declaration.");
  add("host_macro_condition", "#define LEFT m\n",
      "if (LEFT < n) { observe(C); } else { observe(E); }", false,
      "Builtin comparison semantics do not authenticate a macro-origin condition.");
  add("host_builtin_macro_extent", "", "auto d=read(D,__LINE__,n);", false,
      "Builtin preprocessor expansion is source staging, not an admitted mathematical extent expression.");
  add("host_inactive_effect_body", "void host_effect();\n",
      "#if 0\nhost_effect();\n#endif\n" + math, false,
      "Skipped unsafe source inside a closed body must not silently become an empty semantic region fragment.");
  add("host_body_include", "", "#include \"body.inc\"\n" + math, false,
      "An include inside the selected body remains outside the admitted grammar even when its expansion is empty.",
      {{"body.inc", "/* deliberately no surviving AST statements */\n"}});
  add("host_body_line_directive", "", "#line 700 \"forged.mdsl\"\n" + math, false,
      "Closed-body directives cannot rewrite diagnostic/source bindings.");
  add("host_helper_erased_macro", "#define ERASED\nusing namespace mdsl_probe;\n"
      "Value product(Value lhs,Value rhs) { ERASED "
      "return gemm(lhs,rhs,Numerics::strict_f32); }\n",
      "auto c=product(a,b); publish(c,C);", false,
      "Closure screening must cover recursively admitted helper source, not only the region entry.");
  add("host_helper_inactive_effect", "using namespace mdsl_probe;\nvoid host_effect();\n"
      "Value product(Value lhs,Value rhs) {\n#if 0\nhost_effect();\n#endif\n"
      "return gemm(lhs,rhs,Numerics::strict_f32); }\n",
      "auto c=product(a,b); publish(c,C);", false,
      "Skipped helper source belongs to the proof surface even if Clang removes it from the body AST.");

  add("host_copied_primitive_redeclaration", "#include \"copied.h\"\n", math, false,
      "Copied signature-compatible primitive declarations in a host file do not become tool-owned declarations.",
      {{"copied.h", "#pragma once\nnamespace mdsl_probe { "
                    "Value gemm(Value,Value,Numerics) noexcept; }\n"}});
  add("host_forged_primitive_annotation", "#include \"forged.h\"\n",
      "auto c=forged::gemm(a,b,Numerics::strict_f32); publish(c,C);", false,
      "Names, signatures and matching annotation text cannot grant primitive authority.",
      {{"forged.h", "#pragma once\nnamespace forged { "
                    "[[clang::annotate(\"mdsl.private.gemm.v1\")]] "
                    "mdsl_probe::Value gemm(mdsl_probe::Value,mdsl_probe::Value,"
                    "mdsl_probe::Numerics) noexcept; }\n"}});
  add("host_header_inherited_attribute", "#include \"attribute.h\"\n", math, false,
      "Ambient function attributes inherited from a header remain outside the region contract.",
      {{"attribute.h", "#pragma clang attribute push "
                       "(__attribute__((annotate(\"host.authority\"))), apply_to=function)\n"}});
  cases.back().source += "#pragma clang attribute pop\n";
  add("host_header_fenv_access", "#include \"fp_state.h\"\n", math, false,
      "Inherited floating-environment access cannot silently alter the fixed region numerical/failure contract.",
      {{"fp_state.h", "#pragma STDC FENV_ACCESS ON\n"}});
  add("host_header_fp_contract", "#include \"fp_state.h\"\n", math, false,
      "Ambient reassociation/contraction state must be rejected or staged explicitly, not imported as numerical permission.",
      {{"fp_state.h", "#pragma clang fp contract(fast)\n"}});

  // These affect declaration spelling rather than a surviving body expression.
  cases.push_back({"host_macro_region_marker",
      "#define REGION_MARKER " + marker + "using namespace mdsl_probe;\n"
      "REGION_MARKER " + signature + " {\n" + inputs + math + "\n}\n",
      {}, true, false, "Macro-owned region annotations cannot create source authority."});
  cases.push_back({"host_empty_macro_parameter",
      "#define EMPTY\nusing namespace mdsl_probe;\n" + marker +
      "void region(Storage A, Storage B, Storage C, Storage D, Storage E, "
      "Shape EMPTY m, Shape k, Shape n) {\n" + inputs + math + "\n}\n",
      {}, true, false, "An erased token in parameter spelling must not escape function-range closure checks."});
  cases.push_back({"host_pragma_forged_exact_marker",
      "using namespace mdsl_probe;\n"
      "#pragma clang attribute push "
      "(__attribute__((annotate(\"mdsl.private.closed_region.v1\"))), apply_to=function)\n" +
      signature + " {\n" + inputs + math + "\n}\n#pragma clang attribute pop\n",
      {}, true, false, "An exact marker injected by a pragma is not a declaration-owned source annotation."});
  cases.push_back({"host_continued_pragma_forged_marker",
      "using namespace mdsl_probe;\n#pragma clang attribute push \\\n"
      "(__attribute__((annotate(\"mdsl.private.closed_region.v1\"))), apply_to=function)\n" +
      signature + " {\n" + inputs + math + "\n}\n#pragma clang attribute pop\n",
      {}, true, false, "A marker on a continued pragma line must not evade first-physical-line source fencing."});
  cases.push_back({"host_explicit_prototype_marker_inherited",
      "using namespace mdsl_probe;\n" + marker + signature + ";\n" +
      signature + " {\n" + inputs + math + "\n}\n",
      {}, true, true, "An explicit authenticated main-source prototype marker may be inherited by its definition as in the original grammar."});
  return cases;
}

} // namespace matcore::mdslc::test
#endif
