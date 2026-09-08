#include "FrozenHostCodegen.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace matcore::mdslc::codegen::detail {
namespace {
class Diagnostics final : public clang::DiagnosticConsumer {
public:
  bool failed = false;
  std::string text;
  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic &diagnostic) override {
    clang::DiagnosticConsumer::HandleDiagnostic(level, diagnostic);
    if (level < clang::DiagnosticsEngine::Error) return;
    failed = true;
    llvm::SmallString<256> rendered;
    diagnostic.FormatDiagnostic(rendered);
    text += rendered.str().str() + "\n";
  }
};

// ToolInvocation owns/destroys its action. Take the LLVM module before that
// destruction while keeping the LLVMContext alive in the calling stack.
class CaptureLLVM final : public clang::EmitLLVMOnlyAction {
public:
  CaptureLLVM(llvm::LLVMContext &context, std::unique_ptr<llvm::Module> &module)
      : clang::EmitLLVMOnlyAction(&context), module_(module) {}
  bool BeginInvocation(clang::CompilerInstance &compiler) override {
    // Body replacement must precede even O0 LLVM simplifications/inlining.
    // Source-level Clang/Sema lowering is retained; no LLVM pass may fold host
    // behavior using the still-present source-only intrinsic implementation.
    compiler.getCodeGenOpts().DisableLLVMPasses = true;
    return clang::EmitLLVMOnlyAction::BeginInvocation(compiler);
  }
  void EndSourceFileAction() override {
    clang::EmitLLVMOnlyAction::EndSourceFileAction();
    module_ = takeModule();
  }
private:
  std::unique_ptr<llvm::Module> &module_;
};
} // namespace

bool compileFrozenHostToLLVM(
    const frontend::closed_region_host::HostInputSnapshot &snapshot,
    std::string &llvm_ir, std::string &error) {
  llvm_ir.clear();
  error.clear();
  if (!snapshot.unchanged(error)) return false;
  auto replay = snapshot.replay();
  if (!replay.ok(error)) return false;
  clang::FileSystemOptions file_options;
  file_options.WorkingDir = snapshot.workingDirectory();
  auto files = llvm::makeIntrusiveRefCnt<clang::FileManager>(file_options, replay.filesystem);
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module;
  Diagnostics diagnostics;
  auto arguments = snapshot.arguments();
  // Keep the exact language, preprocessor, resource-dir and target context.
  // The explicit action requests unoptimized code generation; no user backend
  // flags or altered floating-point options enter the ordinary host TU.
  clang::tooling::ToolInvocation invocation(arguments,
      std::make_unique<CaptureLLVM>(context, module), files.get());
  invocation.setDiagnosticConsumer(&diagnostics);
  const bool ran = invocation.run();
  if (!replay.ok(error)) return false;
  if (!ran || diagnostics.failed || !module) {
    error = "closed host LLVM generation failed: " + diagnostics.text;
    return false;
  }
  std::string verification;
  llvm::raw_string_ostream diagnostic_stream(verification);
  if (llvm::verifyModule(*module, &diagnostic_stream)) {
    error = "Clang produced invalid host LLVM IR: " + verification;
    return false;
  }
  if (!snapshot.unchanged(error)) return false;
  llvm::raw_string_ostream output(llvm_ir);
  module->print(output, nullptr);
  return true;
}
} // namespace matcore::mdslc::codegen::detail
