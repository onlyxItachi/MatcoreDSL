#include "ArtifactSymbolOwnership.h"
#include "platform_support.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cg = matcore::mdslc::codegen;
namespace support = matcore::mdslc::support;
namespace fs = std::filesystem;
unsigned checks = 0, failures = 0;
void check(bool value, const std::string &label) {
  ++checks;
  if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
void write(const fs::path &path, const std::string &contents) {
  std::ofstream file(path);
  file << contents;
  file.close();
  if (!file) throw std::runtime_error("cannot write fixture");
}
struct Fixture {
  support::TempDirectoryV1 temporary;
  llvm::LLVMContext context;
  std::string clang;
  explicit Fixture(std::string compiler) : clang(std::move(compiler)) {
    std::string error;
    auto directory = support::create_temp_directory_v1("mdslc-artifact-ownership", error);
    if (!directory) throw std::runtime_error(error);
    temporary = std::move(*directory);
  }
  void run(std::vector<std::string> arguments) {
    support::ProcessRequestV1 request;
    request.argv = std::move(arguments);
    request.working_directory = temporary.path();
    request.environment = support::compiler_environment_sanitization_v1();
    auto result = support::run_process_v1(request);
    check(result.launched && result.exit_code == 0, "actual Clang compilation " + result.stderr_text);
    if (!result.launched || result.exit_code != 0) throw std::runtime_error("Clang fixture failed");
  }
  std::unique_ptr<llvm::Module> host(const std::string &name, const std::string &source) {
    const auto cpp = temporary.path() / (name + ".cpp");
    const auto ir = temporary.path() / (name + ".ll");
    write(cpp, source);
    run({clang, "-std=c++20", "-O0", "-Xclang", "-disable-llvm-passes",
         "-S", "-emit-llvm", cpp.string(), "-o", ir.string()});
    llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseIRFile(ir.string(), diagnostic, context);
    if (!module) { diagnostic.print("ownership", llvm::errs()); throw std::runtime_error("fixture IR failed"); }
    return module;
  }
  std::unique_ptr<llvm::MemoryBuffer> library(const std::string &name, const std::string &source,
                                             bool shared = true, bool hide_internals = false) {
    const auto cpp = temporary.path() / (name + ".cpp");
    const auto obj = temporary.path() / (name + (shared ? ".so" : ".o"));
    write(cpp, source);
    std::vector<std::string> command{clang, "-std=c++20", "-O0", "-fPIC",
                                     shared ? "-shared" : "-c", cpp.string(), "-o", obj.string()};
    if (hide_internals) {
      command.push_back("-fvisibility=hidden");
      command.push_back("-fvisibility-inlines-hidden");
    }
    run(std::move(command));
    auto buffer = llvm::MemoryBuffer::getFile(obj.string());
    if (!buffer) throw std::runtime_error("fixture object read failed");
    return std::move(*buffer);
  }
};

int main(int argc, char **argv) {
  if (argc != 2 && argc != 4) return 2;
  try {
    Fixture fixture(argv[1]);
    auto runtime = fixture.library("runtime", R"cpp(
      #define PUBLIC __attribute__((visibility("default")))
      extern "C" PUBLIC int matcore_runtime_owned() { return 3; }
      namespace matcore {
      PUBLIC int owned_cpp() { return 3; }
      template<class T> PUBLIC T owned_template(T value) { return value + 1; }
      template int owned_template<int>(int);
      template<class T> PUBLIC T owned_guard(T value) { static T saved = owned_cpp(); return saved + value; }
      template int owned_guard<int>(int);
      }
    )cpp", true, true);
    auto provider = fixture.library("provider", R"cpp(
      extern "C" {
      void *blas_memory_alloc(int) { return nullptr; }
      const char *gotoblas_corename() { return "fixture"; }
      int blas_cpu_number = 1;
      void *gotoblas = nullptr;
      __attribute__((weak)) int provider_weak() { return 1; }
      int provider_impl() { return 1; }
      int provider_alias() __attribute__((alias("provider_impl")));
      auto provider_resolver() -> int (*)() { return &provider_impl; }
      int provider_ifunc() __attribute__((ifunc("provider_resolver")));
      }
    )cpp");
    const cg::TrustedSymbolArtifact artifacts[] = {
      {cg::SymbolArtifactOwner::MatcoreRuntime, runtime->getMemBufferRef()},
      {cg::SymbolArtifactOwner::ExternalProvider, provider->getMemBufferRef()}};
    auto ordinary = fixture.host("ordinary", R"cpp(
      #include <vector>
      extern "C" int matcore_runtime_owned();
      extern "C" int blas_cpu_number;
      int host() { std::vector<int> values(1, 2); return values[0] + matcore_runtime_owned() + blas_cpu_number; }
    )cpp");
    cg::ArtifactSymbolOwnershipReport report;
    std::string error;
    check(cg::verifyHostArtifactSymbolOwnership(*ordinary, artifacts, report, error),
          "ordinary STL host and explicit runtime exports remain legal: " + error);
    check(report.runtime_exports == 6 && report.provider_exports == 9,
          "complete runtime/provider export sets counted without demangling");
    auto iostream = fixture.host("iostream", "#include <iostream>\nvoid host(){std::cout << 7;}\n");
    check(cg::verifyHostArtifactSymbolOwnership(*iostream, artifacts, report, error),
          "real libstdc++ iostream declaration is harmless: " + error);
    for (const auto *assembly : {
        ".globl _ZSt21ios_base_library_initv; .set matcore_runtime_owned,0",
        ".globl _ZSt21ios_base_library_initv\n.globl matcore_runtime_owned",
        ".globl _ZSt21ios_base_library_initv\nmatcore_runtime_owned:",
        ".globl \"_ZSt21ios_base_library_initv\"",
        ".globl matcore_runtime_owned"}) {
      iostream->setModuleInlineAsm(assembly);
      check(!cg::verifyHostArtifactSymbolOwnership(*iostream, artifacts, report, error) &&
            error.find("module assembly") != std::string::npos,
            "iostream exception admits no suffix, label, quoted spelling or other symbol");
    }
    iostream->setModuleInlineAsm(".globl _ZSt21ios_base_library_initv\n");
    auto reserved_iostream = fixture.library("reserved_iostream",
        "extern \"C\" int other() asm(\"_ZSt21ios_base_library_initv\"); int other(){return 1;}\n");
    const cg::TrustedSymbolArtifact reserved[] = {artifacts[0],
        {cg::SymbolArtifactOwner::ExternalProvider, reserved_iostream->getMemBufferRef()}};
    check(!cg::verifyHostArtifactSymbolOwnership(*iostream, reserved, report, error),
          "even exact iostream .globl cannot promote a reserved artifact symbol");
    auto reject = [&](const std::string &name, const std::string &source, const std::string &reason) {
      auto module = fixture.host(name, source);
      const bool admitted = cg::verifyHostArtifactSymbolOwnership(*module, artifacts, report, error);
      check(!admitted && error.find(reason) != std::string::npos, name + " rejects at intended ownership boundary: " + error);
      check(report.runtime_exports == 0 && report.provider_exports == 0, "failure issues no success report");
    };
    reject("runtime_function", "extern \"C\" int matcore_runtime_owned() { return 99; }", "matcore_runtime_owned");
    reject("runtime_cpp", "namespace matcore { int owned_cpp() { return 99; } }", "owned_cpp");
    reject("runtime_template_return", "namespace matcore { template<class T> T owned_template(T) { return 99; } template int owned_template<int>(int); }", "owned_template");
    reject("runtime_local_guard", R"cpp(extern "C" unsigned long long different asm("_ZGVZN7matcore11owned_guardIiEET_S1_E5saved"); unsigned long long different = 9;)cpp", "_ZGVZN7matcore11owned_guard");
    reject("runtime_local_value", R"cpp(extern "C" int different asm("_ZZN7matcore11owned_guardIiEET_S1_E5saved"); int different = 9;)cpp", "_ZZN7matcore11owned_guard");
    reject("provider_internal", "extern \"C\" void *blas_memory_alloc(int) { return nullptr; }", "blas_memory_alloc");
    reject("provider_internal_name", "extern \"C\" const char *gotoblas_corename() { return \"host\"; }", "gotoblas_corename");
    reject("provider_data", "extern \"C\" { int blas_cpu_number = 4; }", "blas_cpu_number");
    reject("provider_pointer_data", "extern \"C\" { void *gotoblas = nullptr; }", "gotoblas");
    reject("provider_weak", "extern \"C\" __attribute__((weak)) int provider_weak() { return 9; }", "provider_weak");
    reject("provider_alias", "extern \"C\" int other() { return 1; } extern \"C\" int provider_alias() __attribute__((alias(\"other\")));", "provider_alias");
    reject("provider_ifunc", "extern \"C\" int other() { return 1; } extern \"C\" auto resolver() -> int (*)() { return &other; } extern \"C\" int provider_ifunc() __attribute__((ifunc(\"resolver\")));", "provider_ifunc");
    reject("runtime_asm_name", "int differently_named() asm(\"matcore_runtime_owned\"); int differently_named() { return 4; }", "matcore_runtime_owned");
    reject("module_asm", R"cpp(asm(".text\n.globl matcore_runtime_owned\nmatcore_runtime_owned:\nret\n"); int host() { return 1; })cpp", "module assembly");
    reject("instruction_asm", R"cpp(void host() { asm volatile(".globl matcore_runtime_owned\n.set matcore_runtime_owned,0"); })cpp", "instruction assembly");
    reject("instruction_callbr", R"cpp(void host() { asm goto("nop" : : : : label); label: return; })cpp", "instruction assembly");
    auto barrier = fixture.host("barrier", "void host() { asm volatile(\"\" ::: \"memory\"); }");
    check(cg::verifyHostArtifactSymbolOwnership(*barrier, artifacts, report, error), "empty barriers cannot define symbols");
    auto local = fixture.host("local", "static int differently_named() asm(\"matcore_runtime_owned\"); static int differently_named() { return 9; } int host() { return differently_named(); }");
    check(cg::verifyHostArtifactSymbolOwnership(*local, artifacts, report, error), "local same-label definition cannot interpose DSO");
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, {}, report, error), "missing runtime rejected");
    const cg::TrustedSymbolArtifact duplicates[] = {artifacts[0], artifacts[0]};
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, duplicates, report, error), "duplicate runtime rejected");
    check(cg::verifyHostArtifactSymbolOwnership(*ordinary, llvm::ArrayRef(artifacts, 1), report, error), "provider OFF remains legal");
    auto candidates = fixture.library("candidates", "extern \"C\" int private_candidate_entry() { return 7; }");
    const cg::TrustedSymbolArtifact with_candidates[] = {artifacts[0],
      {cg::SymbolArtifactOwner::PrivateCandidates, candidates->getMemBufferRef()}};
    check(cg::verifyHostArtifactSymbolOwnership(*ordinary, with_candidates, report, error) &&
          report.runtime_exports == 6 && report.candidate_exports == 1 && report.provider_exports == 0,
          "candidate DSO has distinct counted ownership, independent of optional provider");
    auto candidate_collision = fixture.host("candidate_collision",
        "extern \"C\" int private_candidate_entry() { return 99; }");
    check(!cg::verifyHostArtifactSymbolOwnership(*candidate_collision, with_candidates, report, error) &&
          error.find("private_candidate_entry") != std::string::npos && report.candidate_exports == 0,
          "private candidate export collision refuses success report");
    const cg::TrustedSymbolArtifact duplicate_candidates[] = {
      with_candidates[0], with_candidates[1], with_candidates[1]};
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, duplicate_candidates, report, error),
          "duplicate candidate DSO rejected");
    auto no_exports = fixture.library("no_exports", "int hidden() { return 0; }", true, true);
    const cg::TrustedSymbolArtifact empty_dso{cg::SymbolArtifactOwner::MatcoreRuntime, no_exports->getMemBufferRef()};
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, empty_dso, report, error), "empty export table rejected");
    auto leaked = fixture.library("leaked", "#include <vector>\nint host() { std::vector<int> v(1,2); return v[0]; }");
    const cg::TrustedSymbolArtifact leaked_dso{cg::SymbolArtifactOwner::MatcoreRuntime, leaked->getMemBufferRef()};
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, leaked_dso, report, error), "exported overlapping runtime STL implementation fails closed");
    auto object = fixture.library("not_dso", "extern \"C\" int matcore_runtime_owned() { return 0; }", false);
    const cg::TrustedSymbolArtifact not_dso{cg::SymbolArtifactOwner::MatcoreRuntime, object->getMemBufferRef()};
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, not_dso, report, error), "relocatable artifact is not a DSO");
    auto garbage = llvm::MemoryBuffer::getMemBuffer("not an object", "garbage");
    const cg::TrustedSymbolArtifact malformed{cg::SymbolArtifactOwner::MatcoreRuntime, garbage->getMemBufferRef()};
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, malformed, report, error), "malformed artifact fails closed");
    if (argc == 4) {
      auto real_runtime = llvm::MemoryBuffer::getFile(argv[2]);
      auto real_provider = llvm::MemoryBuffer::getFile(argv[3]);
      if (!real_runtime || !real_provider) throw std::runtime_error("actual artifact input unavailable");
      const cg::TrustedSymbolArtifact actual[] = {
        {cg::SymbolArtifactOwner::MatcoreRuntime, (*real_runtime)->getMemBufferRef()},
        {cg::SymbolArtifactOwner::ExternalProvider, (*real_provider)->getMemBufferRef()}};
      check(cg::verifyHostArtifactSymbolOwnership(*ordinary, actual, report, error), "actual runtime/provider with ordinary STL host: " + error);
      check(report.runtime_exports == 15 && report.provider_exports > 100, "actual runtime exports only15 public functions; provider owns broader exports");
      auto override = fixture.host("actual_provider_override", "extern \"C\" void *blas_memory_alloc(int) { return nullptr; }");
      check(!cg::verifyHostArtifactSymbolOwnership(*override, actual, report, error) && error.find("blas_memory_alloc") != std::string::npos, "actual provider internal export collision rejected");
    }
    const auto original_target = ordinary->getTargetTriple();
    const bool native_arm = original_target.getArch() == llvm::Triple::aarch64;
    ordinary->setTargetTriple(llvm::Triple(native_arm ? "x86_64-unknown-linux-gnu"
                                                    : "aarch64-unknown-linux-gnu"));
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, artifacts, report, error) &&
          error.find("matching the authenticated host target") != std::string::npos &&
          report.runtime_exports == 0 && report.provider_exports == 0,
          "otherwise supported target cannot borrow different-machine DSO authority");
    ordinary->setTargetTriple(llvm::Triple("riscv64-unknown-linux-gnu"));
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, artifacts, report, error),
          "unqualified host architecture rejected");
    ordinary->setTargetTriple(original_target);
    ordinary->setDataLayout("E-p:64:64");
    check(!cg::verifyHostArtifactSymbolOwnership(*ordinary, artifacts, report, error),
          "big-endian host cannot borrow little-endian DSO authority");
  } catch (const std::exception &exception) {
    check(false, exception.what());
  }
  std::cout << checks << " ownership checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
