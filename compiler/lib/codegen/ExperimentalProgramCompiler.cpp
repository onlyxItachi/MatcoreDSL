#include "ExperimentalProgramCompiler.h"
#include "AuthenticatedHostThunk.h"
#include "FrozenHostCodegen.h"
#include "../frontend/ProgramHostInterface.h"
#include "../frontend/ClosedRegionAdmissionInternal.h"
#include "platform_support.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace matcore::mdslc::codegen {
namespace {
namespace fs = std::filesystem;
namespace fe = frontend;
namespace cg = matcore::mdslc::codegen;
namespace host = fe::closed_region_host;
using ParameterKind = fe::ClosedRegionParameterBinding::Kind;
using Entries = fe::detail::ProgramRegionOwners;
[[noreturn]] void reject(const std::string &message) { throw std::runtime_error(message); }
void write(const fs::path &path, const std::string &bytes) {
  std::ofstream output(path, std::ios::binary);
  output << bytes; output.close();
  if (!output) reject("cannot write compiler-issued declaration witness");
}
struct Unit {
  fe::Options options;
  fs::path path;
  std::string region;
  std::optional<fe::AuthenticatedClosedRegionEvidence> evidence;
  std::optional<cg::ExperimentalRegionEmission> emission;
  std::shared_ptr<const host::HostInputSnapshot> snapshot;
  std::optional<cg::ExperimentalLLVMCompilation> compilation;
  std::unique_ptr<llvm::Module> raw;
  std::vector<std::string> foreign_definitions;
};
std::string symbolKey(llvm::StringRef name) {
  if (name.starts_with("\1")) name = name.drop_front();
  return name.split('@').first.str();
}
std::unique_ptr<llvm::Module> parseIr(const std::string &ir, llvm::LLVMContext &context) {
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseAssemblyString(ir, diagnostic, context);
  if (!module) reject("frozen Clang LLVM could not be parsed");
  return module;
}

bool distinctSourceIdentities(const std::vector<fs::path> &paths, std::string &error) {
  std::vector<support::FileIdentityV1> identities;
  for (const auto &path : paths) {
    const auto file = support::capture_file_snapshot_v1(path, error);
    if (!error.empty() || !file.exists || !file.regular_file || !file.identity) {
      if (error.empty()) error = "program requires an existing regular source TU: " + path.string();
      return false;
    }
    for (const auto &identity : identities)
      if (support::same_file_identity_v1(identity, file.identity)) {
        error = "duplicate source TU physical identity: " + path.string(); return false;
      }
    identities.push_back(file.identity);
  }
  return true;
}

ExperimentalProgramCompilation compileProgram(
    const std::vector<ExperimentalProgramSource> &sources, const std::string &cwd,
    const ExperimentalCompilerInputs &inputs, ClosedCpuPolicy policy,
    const std::function<void()> &after_interface_capture) {
  if (sources.size() < 2 || sources.size() > 8)
    reject("program requires 2..8 source translation units");
  if (std::count_if(inputs.symbol_artifacts.begin(), inputs.symbol_artifacts.end(),
      [](const auto &artifact) { return artifact.owner == SymbolArtifactOwner::PrivateCandidates; }) != 1)
    reject("program execution requires its isolated private candidate DSO");
  if (!inputs.staging_directory.is_absolute() || !fs::is_directory(inputs.staging_directory))
    reject("program compilation requires an owned absolute staging directory");
  llvm::LLVMContext context; // All modules die before this context, including rejection unwind.
  std::vector<Unit> units;
  std::vector<fs::path> source_paths;
  std::string error;
  for (const auto &source : sources) {
    const fs::path path(source.options.input_path);
    if (!path.is_absolute()) reject("program source paths must be absolute without lexical rewriting");
    source_paths.push_back(path);
    Unit unit;
    unit.options = source.options; unit.path = path; unit.region = source.region;
    units.push_back(std::move(unit));
  }
  if (!distinctSourceIdentities(source_paths, error)) reject(error);
  std::vector<fs::path> helper_directories;
  for (std::size_t i = 0; i < units.size(); ++i) {
    auto directory = inputs.staging_directory / ("helper-" + std::to_string(i));
    if (!fs::create_directory(directory)) reject("program helper staging already exists");
    helper_directories.push_back(directory);
  }
  const auto witness_directory = inputs.staging_directory / "interface-witness";
  if (!fs::create_directory(witness_directory)) reject("program interface staging already exists");
  const auto &artifacts = inputs.symbol_artifacts;
  const fe::ExperimentalRegionHeaders headers{
      (inputs.public_include_directory / "matcore/region.h").string(),
      (inputs.public_include_directory / "matcore/detail/region_storage.h").string()};
  auto optionsFor = [](const Unit &unit) { return unit.options; };
  Entries entries;
  std::map<std::string, std::set<std::size_t>> retired;
  std::set<std::string> helpers;
  for (std::size_t i = 0; i < units.size(); ++i) {
    auto &unit = units[i];
    if (unit.region.empty()) continue;
    auto admitted = fe::admitExperimentalRegionHost(optionsFor(unit), cwd, headers, unit.region);
    if (!admitted) reject("ADMISSION: " + admitted.error);
    unit.evidence = std::move(*admitted.evidence);
    const auto &binding = *unit.evidence->entryBinding();
    if (!entries.emplace(binding.mangled_name, fe::detail::ProgramRegionOwner{i, binding}).second)
      reject("OWNERSHIP: multiple admitted owners for one region symbol");
    auto emitted = cg::emitExperimentalRegion(*unit.evidence, policy);
    if (!emitted) reject(emitted.error);
    unit.emission = std::move(*emitted.emission);
    if (!helpers.insert(unit.emission->helper_symbol).second) reject("OWNERSHIP: duplicate issued helper");
    for (const auto &name : unit.emission->retired_value_helpers) retired[name].insert(i);
    unit.snapshot = fe::detail::ClosedRegionCompilationAccess::host(*unit.evidence);
  }
  if (entries.empty()) reject("program requires at least one admitted region");
  unsigned mains = 0;
  std::optional<std::size_t> main_owner;
  for (std::size_t i = 0; i < units.size(); ++i) {
    auto &unit = units[i];
    auto inspected = fe::detail::inspectProgramHost(unit.options, cwd, headers, entries, i, unit.snapshot);
    if (!inspected) reject("INTERFACE: " + inspected.error);
    unit.snapshot = std::move(inspected.snapshot);
    mains += inspected.interface.main_definitions;
    if (inspected.interface.main_definitions) main_owner = i;
    unit.foreign_definitions = std::move(inspected.interface.foreign_definitions);
    std::string ir;
    if (!cg::detail::compileFrozenHostToLLVM(*unit.snapshot, ir, error)) reject(error);
    unit.raw = parseIr(ir, context);
    cg::ArtifactSymbolOwnershipReport ownership;
    if (!cg::verifyHostArtifactSymbolOwnership(*unit.raw, artifacts, ownership, error)) reject(error);
  }
  if (mains != 1) reject("ENTRY: program requires exactly one ordinary C++ main definition");
  const auto *main_function = units[*main_owner].raw->getFunction("main");
  if (!main_function || main_function->isDeclaration())
    reject("ENTRY: source main does not own the ordinary LLVM main definition");
  // sameAbi deliberately compares complete parameter/target attributes. Clang
  // gives a definition extra sret-noalias and CPU facts absent on its ordinary
  // declaration. Generate a clean declaration counterpart from the already
  // sealed canonical parameter kinds; do not delete facts until a comparison
  // passes or implement a competing ABI comparator.
  std::string witness_source = "#include <matcore/region.h>\nnamespace md = matcore::mdsl;\n"
    "#pragma clang diagnostic ignored \"-Wreturn-type-c-linkage\"\n";
  unsigned ordinal = 0;
  for (const auto &[name, entry] : entries) {
    if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos)
      reject("compiler declaration witness requires ordinary ASCII linker spelling");
    const auto declaration = "__mdsl_program_decl_" + std::to_string(ordinal);
    witness_source += "extern \"C\" md::Result " + declaration + "(";
    for (std::size_t i = 0; i < entry.binding.parameters.size(); ++i) {
      if (i) witness_source += ", ";
      witness_source += entry.binding.parameters[i].kind == ParameterKind::Storage ? "md::Storage" : "md::Shape";
    }
    witness_source += ") noexcept asm(\"" + name + "\");\n";
    witness_source += "extern \"C\" { auto *__mdsl_program_use_" + std::to_string(ordinal) +
      " = &" + declaration + "; }\n";
    witness_source += "extern \"C\" md::Result __mdsl_program_call_" + std::to_string(ordinal++) + "(";
    for (std::size_t i = 0; i < entry.binding.parameters.size(); ++i) {
      if (i) witness_source += ", ";
      witness_source += entry.binding.parameters[i].kind == ParameterKind::Storage ? "md::Storage " : "md::Shape ";
      witness_source += "arg" + std::to_string(i);
    }
    witness_source += ") noexcept { return " + declaration + "(";
    for (std::size_t i = 0; i < entry.binding.parameters.size(); ++i) {
      if (i) witness_source += ", ";
      witness_source += "arg" + std::to_string(i);
    }
    witness_source += "); }\n";
  }
  const auto witness_path = witness_directory / "interface.cpp";
  write(witness_path, witness_source);
  fe::Options witness_options;
  witness_options.input_path = witness_path.string();
  witness_options.clang_path = inputs.clang_path;
  witness_options.clang_resource_directory = inputs.clang_resource_directory;
  witness_options.compiler_arguments = {"-I/__mdsl_private__"};
  if (inputs.address_undefined_sanitizers) witness_options.compiler_arguments.push_back("-fsanitize=address,undefined");
  auto witness_capture = host::prepareHostInputs(witness_options, witness_directory.string(),
    {{"/__mdsl_private__/fixture.h", "#pragma once\n"},
     {"/__mdsl_private__/matcore/region.h", fe::detail::experimentalRegionHeaderSource()},
     {"/__mdsl_private__/matcore/detail/region_storage.h", fe::detail::experimentalRegionStorageHeaderSource()}}, error);
  if (!witness_capture || witness_capture->sourceSnapshot() != witness_source)
    reject("compiler-issued interface witness differs from its source: " + error);
  clang::FileSystemOptions witness_file_options;
  witness_file_options.WorkingDir = witness_directory.string();
  auto witness_files = llvm::makeIntrusiveRefCnt<clang::FileManager>(witness_file_options, witness_capture->fileSystem());
  clang::tooling::ToolInvocation witness_syntax(witness_capture->arguments(),
    std::make_unique<clang::SyntaxOnlyAction>(), witness_files.get());
  if (!witness_syntax.run()) reject("compiler-issued declaration witness failed Sema");
  auto witness_snapshot = witness_capture->freeze(error);
  if (!witness_snapshot) reject(error);
  std::string witness_ir;
  if (!cg::detail::compileFrozenHostToLLVM(*witness_snapshot, witness_ir, error)) reject(error);
  auto witness = parseIr(witness_ir, context);
  std::map<std::string, const llvm::Function *> witness_declarations;
  std::map<std::string, const llvm::CallBase *> witness_calls;
  for (const auto &function : *witness)
    if (!function.isIntrinsic()) witness_declarations.emplace(symbolKey(function.getName()), &function);
  for (const auto &function : *witness)
    for (const auto &block : function)
      for (const auto &instruction : block)
        if (const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction))
          if (const auto *callee = call->getCalledFunction();
              callee && entries.contains(symbolKey(callee->getName())) &&
              !witness_calls.emplace(symbolKey(callee->getName()), call).second)
            reject("compiler call witness has duplicate protected calls");
  if (witness_calls.size() != entries.size()) reject("compiler call witness is incomplete");
  if (after_interface_capture) after_interface_capture();
  for (std::size_t i = 0; i < units.size(); ++i) {
    const auto &module = *units[i].raw;
    if (module.getTargetTriple() != witness->getTargetTriple() || module.getDataLayoutStr() != witness->getDataLayoutStr())
      reject("ABI: source TU target or data layout differs from canonical declaration witness");
    for (const auto &global : module.global_values()) {
      const auto name = symbolKey(global.getName());
      const llvm::GlobalObject *alias_target = nullptr;
      if (const auto *alias = llvm::dyn_cast<llvm::GlobalAlias>(&global))
        alias_target = alias->getAliaseeObject();
      if (const auto *ifunc = llvm::dyn_cast<llvm::GlobalIFunc>(&global))
        alias_target = ifunc->getResolverFunction();
      if (alias_target && entries.contains(symbolKey(alias_target->getName())))
        reject("OWNERSHIP: source LLVM alias/resolver targets an issued region");
      if (name == "main" && !global.isDeclaration() && i != *main_owner)
        reject("ENTRY: foreign source symbol competes with main");
      if (helpers.contains(name)) reject("OWNERSHIP: source names an issued private helper");
      if (auto entry = entries.find(name); entry != entries.end()) {
        if (i != entry->second.owner && !global.isDeclaration()) {
          const auto *object = llvm::dyn_cast<llvm::GlobalObject>(&global);
          reject("OWNERSHIP: competing region definition before linker resolution: " + name +
            " linkage=" + std::to_string(global.getLinkage()) +
            " comdat=" + std::to_string(object && object->hasComdat()));
        }
        const auto *function = llvm::dyn_cast<llvm::Function>(&global);
        const auto *owner = i == entry->second.owner ? units[i].raw->getFunction(name) :
          (witness_declarations.contains(name) ? witness_declarations.at(name) : nullptr);
        if (!function || !owner || !cg::sameAuthenticatedHostABI(*function, *owner, false)) {
          std::string detail;
          llvm::raw_string_ostream out(detail);
          if (function) { out << "\nconsumer: "; function->print(out); }
          if (owner) { out << "\nproducer attributes: "; owner->getAttributes().print(out); }
          reject("ABI: foreign region declaration differs from sealed producer: " + name + detail);
        }
        // sameAbi intentionally is not an effect-summary comparator. Reject
        // additional declaration-side promises (e.g. memory(read/none)) too;
        // a clean compiler witness does not authorize source pure/const facts.
        if (i != entry->second.owner &&
            !cg::sameAuthenticatedHostFunctionAttributes(*function, *owner))
          reject("DECLARATION: foreign LLVM function promises differ from compiler witness: " + name);
      }
      if (auto helper = retired.find(name); helper != retired.end() && !helper->second.contains(i) &&
          (!global.isDeclaration() || !global.use_empty()))
        reject("RETIREMENT: foreign TU uses a retired source-only Value helper: " + name);
    }
    if (!units[i].foreign_definitions.empty())
      reject("OWNERSHIP: foreign source defines an issued region even when Clang emits no definition: " +
        units[i].foreign_definitions.front());
    for (const auto &function : module)
      for (const auto &block : function)
        for (const auto &instruction : block)
          if (const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
            const auto *callee = llvm::dyn_cast<llvm::GlobalValue>(
                call->getCalledOperand()->stripPointerCastsAndAliases());
            if (callee && entries.contains(symbolKey(callee->getName()))) {
              const auto *expected = witness_calls.at(symbolKey(callee->getName()));
              if (!sameAuthenticatedHostCallInterface(*call, *expected))
                reject("CALLSITE: region call ABI or effect promises differ from compiler witness");
            }
          }
  }
  auto unchanged = [&] {
    if (!witness_snapshot->unchanged(error)) reject(error);
    for (const auto &unit : units) {
      if (!unit.snapshot->unchanged(error)) reject(error);
      if (unit.compilation && !unit.compilation->inputsUnchanged(error)) reject(error);
    }
    // The first path check precedes capture. Recheck the now-frozen closure so
    // a path replaced between that check and admission cannot evade uniqueness.
    if (!distinctSourceIdentities(source_paths, error)) reject(error);
  };
  unchanged();
  std::cout << "VALIDATED program ownership and ABI before cross-TU link/optimization\n";
  std::unique_ptr<llvm::Module> combined;
  for (std::size_t i = 0; i < units.size(); ++i) {
    auto &unit = units[i];
    std::unique_ptr<llvm::Module> module;
    if (unit.evidence) {
      auto region_inputs = inputs;
      region_inputs.staging_directory = helper_directories[i];
      auto compiled = cg::compileExperimentalRegionToLLVM(*unit.evidence, region_inputs, policy);
      if (!compiled) reject(compiled.error);
      unit.compilation = std::move(*compiled.compilation);
      module = parseIr(unit.compilation->llvm_ir, context);
    } else module = std::move(unit.raw);
    if (!combined) combined = std::move(module);
    else {
      std::cout << "CROSS_TU_LINK " << i << "\n";
      if (llvm::Linker::linkModules(*combined, std::move(module))) reject("ordinary LLVM program link failed");
    }
    unchanged();
  }
  for (const auto &[name, entry] : entries) {
    auto *function = combined->getFunction(name);
    const auto *original = units[entry.owner].raw->getFunction(name);
    if (!function || function->isDeclaration() || !cg::sameAuthenticatedHostABI(*function, *original, false))
      reject("ABI: linked region no longer matches its issued producer");
  }
  for (const auto &name : helpers) {
    auto *helper = combined->getFunction(name);
    if (!helper || helper->isDeclaration()) reject("OWNERSHIP: issued implementation missing after link");
    helper->setLinkage(llvm::GlobalValue::InternalLinkage);
  }
  std::string verification;
  llvm::raw_string_ostream diagnostics(verification);
  if (llvm::verifyModule(*combined, &diagnostics)) reject("invalid combined LLVM: " + verification);
  std::string ir;
  llvm::raw_string_ostream stream(ir);
  combined->print(stream, nullptr);
  unchanged();
  ExperimentalProgramCompilation result;
  result.llvm_ir = std::move(ir);
  result.inputs.push_back(std::move(witness_snapshot));
  for (auto &unit : units) {
    result.inputs.push_back(std::move(unit.snapshot));
    if (unit.compilation) result.regions.push_back(std::move(*unit.compilation));
  }
  return result;
}
} // namespace

bool ExperimentalProgramCompilation::inputsUnchanged(std::string &error) const {
  if (inputs.size() < 3 || regions.empty()) {
    error = "compiled program is missing its complete source/witness closure"; return false;
  }
  for (const auto &input : inputs)
    if (!input || !input->unchanged(error)) return false;
  for (const auto &region : regions)
    if (!region.inputsUnchanged(error)) return false;
  std::vector<fs::path> source_paths;
  for (std::size_t i = 1; i < inputs.size(); ++i) source_paths.emplace_back(inputs[i]->inputPath());
  return distinctSourceIdentities(source_paths, error);
}
ExperimentalProgramCompilationResult compileExperimentalProgramToLLVMForTesting(
    const std::vector<ExperimentalProgramSource> &sources, const std::string &cwd,
    const ExperimentalCompilerInputs &inputs, ClosedCpuPolicy policy,
    const std::function<void()> &after_interface_capture) {
  ExperimentalProgramCompilationResult result;
  try { result.compilation = compileProgram(sources, cwd, inputs, policy, after_interface_capture); }
  catch (const std::exception &error) { result.error = error.what(); }
  return result;
}
ExperimentalProgramCompilationResult compileExperimentalProgramToLLVM(
    const std::vector<ExperimentalProgramSource> &sources, const std::string &cwd,
    const ExperimentalCompilerInputs &inputs, ClosedCpuPolicy policy) {
  return compileExperimentalProgramToLLVMForTesting(sources, cwd, inputs, policy, {});
}
} // namespace matcore::mdslc::codegen
