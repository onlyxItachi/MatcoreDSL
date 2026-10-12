#include "SourcePattern.h"
#include "region_driver_paths.h"
#include "region_driver_identity.h"
#include "ClosedRegionAdmissionInternal.h"
#include "MatcoreCpuGemmCandidate.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <sys/stat.h>
#include <tuple>

namespace matcore::mdslc::driver {
namespace {
namespace candidate = matcore::mdslc::cpu_candidate;
#if defined(__linux__) && defined(__x86_64__)
constexpr auto machine = llvm::ELF::EM_X86_64;
constexpr auto target = candidate::CpuTargetV1::LinuxX86_64;
#elif defined(__linux__) && defined(__aarch64__) && !defined(__AARCH64EB__)
constexpr auto machine = llvm::ELF::EM_AARCH64;
constexpr auto target = candidate::CpuTargetV1::LinuxAArch64;
#else
#error "Source output patterns require native little-endian Linux x86-64 or AArch64"
#endif

using ElfObject = llvm::object::ELF64LEObjectFile;
std::unique_ptr<llvm::object::ObjectFile> nativeObject(const Artifact &artifact,
                                                     unsigned type) {
  auto parsed = llvm::object::ObjectFile::createObjectFile(
      llvm::MemoryBufferRef(artifact.bytes, "source-pattern compiler artifact"));
  if (!parsed) reject("invalid source-pattern ELF: " + llvm::toString(parsed.takeError()));
  const auto *elf = llvm::dyn_cast<ElfObject>(parsed->get());
  if (!elf || elf->getELFFile().getHeader().e_machine != machine ||
      elf->getELFFile().getHeader().e_type != type)
    reject("source-pattern artifact does not match the native ELF64 target/type");
  return std::move(*parsed);
}

struct DynamicInfo {
  std::string soname;
  std::set<std::string> needed;
  bool symbolic = false;
};
DynamicInfo dynamicInfo(const ElfObject &elf, bool issued_private = false) {
  DynamicInfo result;
  auto entries = elf.getELFFile().dynamicEntries();
  if (!entries) reject("invalid source-pattern dynamic table: " + llvm::toString(entries.takeError()));
  llvm::StringRef strings;
  for (const auto &section : elf.sections()) {
    auto name = section.getName();
    if (!name) reject("invalid ELF section name: " + llvm::toString(name.takeError()));
    if (*name == ".dynstr") {
      auto contents = section.getContents();
      if (!contents) reject("invalid ELF dynamic strings: " + llvm::toString(contents.takeError()));
      if (!strings.empty()) reject("duplicate ELF dynamic string table");
      strings = *contents;
    }
  }
  if (strings.empty()) reject("source-pattern artifact lacks dynamic strings");
  auto stringAt = [&](std::uint64_t index) {
    if (index >= strings.size()) reject("dynamic string index is out of bounds");
    const auto text = strings.drop_front(index);
    const auto end = text.find('\0');
    if (end == llvm::StringRef::npos || end == 0) reject("invalid empty/unterminated dynamic string");
    return text.take_front(end).str();
  };
  for (const auto &entry : *entries) {
    switch (entry.d_tag) {
    case llvm::ELF::DT_SONAME:
      if (!result.soname.empty()) reject("duplicate source-pattern SONAME");
      result.soname = stringAt(entry.d_un.d_val);
      break;
    case llvm::ELF::DT_NEEDED:
      if (!result.needed.insert(stringAt(entry.d_un.d_val)).second)
        reject("duplicate source-pattern dynamic dependency");
      break;
    case llvm::ELF::DT_SYMBOLIC: result.symbolic = true; break;
    case llvm::ELF::DT_FLAGS:
      result.symbolic |= (entry.d_un.d_val & llvm::ELF::DF_SYMBOLIC) != 0;
      if (entry.d_un.d_val & llvm::ELF::DF_TEXTREL)
        reject("source-pattern DSO has text relocations");
      break;
    case llvm::ELF::DT_TEXTREL: reject("source-pattern DSO has text relocations");
    case llvm::ELF::DT_RPATH:
    case llvm::ELF::DT_RUNPATH:
      // Final executable owns exact pinned loading paths. A private candidate
      // never acquires a search path into the source/build/consumer context.
      if (issued_private) reject("source-pattern DSO unexpectedly contains a loader search path");
      break;
    default: break;
    }
  }
  return result;
}

using Exports = std::map<std::string, std::tuple<unsigned, unsigned, unsigned>>;
Exports exportSet(const ElfObject &elf) {
  Exports result;
  for (const auto &symbol : elf.getDynamicSymbolIterators()) {
    auto flags = symbol.getFlags();
    auto name = symbol.getName();
    if (!flags || !name) {
      if (!flags) reject("invalid ELF symbol: " + llvm::toString(flags.takeError()));
      reject("invalid ELF symbol name: " + llvm::toString(name.takeError()));
    }
    if (name->empty() || (*flags & llvm::object::SymbolRef::SF_Undefined) ||
        symbol.getBinding() == llvm::ELF::STB_LOCAL) continue;
    if (!result.emplace(name->str(), std::tuple{symbol.getBinding(),
            symbol.getELFType(), symbol.getOther() & 3U}).second)
      reject("duplicate exported source-pattern symbol");
  }
  if (result.empty()) reject("source-pattern DSO has no owning adapter exports");
  return result;
}

void verifyKit(const Artifact &kit) {
  auto archive = llvm::object::Archive::create(llvm::MemoryBufferRef(kit.bytes, "pinned source-pattern relink kit"));
  if (!archive) reject("invalid source-pattern relink archive: " + llvm::toString(archive.takeError()));
  if ((*archive)->isThin()) reject("source-pattern relink kit must contain all member bytes");
  llvm::Error error = llvm::Error::success();
  unsigned members = 0, owner = 0, missing_leaf = 0;
  for (const auto &child : (*archive)->children(error)) {
    auto binary = child.getAsBinary();
    if (!binary) reject("invalid relink member: " + llvm::toString(binary.takeError()));
    const auto *elf = llvm::dyn_cast<ElfObject>(binary->get());
    if (!elf || elf->getELFFile().getHeader().e_machine != machine ||
        elf->getELFFile().getHeader().e_type != llvm::ELF::ET_REL)
      reject("relink kit contains a nonnative/nonrelocatable member");
    ++members;
    for (const auto &symbol : elf->symbols()) {
      auto flags = symbol.getFlags();
      auto name = symbol.getName();
      if (!flags || !name) {
        if (!flags) reject("invalid relink symbol: " + llvm::toString(flags.takeError()));
        reject("invalid relink symbol name: " + llvm::toString(name.takeError()));
      }
      const bool defined = !(*flags & llvm::object::SymbolRef::SF_Undefined);
      if (*name == candidate::kStrictGemmSymbolV1 ||
          *name == candidate::kStrictGemmCInterfaceV1) {
        if (defined) reject("source-pattern relink kit already defines the requested baseline leaf");
        ++missing_leaf;
      }
      if (defined && *name == "matcore_closed_host_private_value_abi_v2") ++owner;
      if (defined && (name->contains("configureForTesting") ||
                      name->contains("allocationAttemptsForTesting")))
        reject("test authority is not permitted in the production relink kit");
    }
  }
  if (error) reject("cannot traverse relink archive: " + llvm::toString(std::move(error)));
  if (members < 2 || owner != 1 || missing_leaf == 0)
    reject("relink kit lacks its exact adapter owner and missing-leaf closure");
}

void verifyLeafObject(const Artifact &object, bool sanitized) {
  auto parsed = nativeObject(object, llvm::ELF::ET_REL);
  const auto &elf = *llvm::cast<ElfObject>(parsed.get());
  std::set<std::string> required{candidate::kStrictGemmSymbolV1, candidate::kStrictGemmCInterfaceV1};
  const std::set<std::string> sanitizer_imports{
      "__asan_init", "__asan_memset", "__asan_register_elf_globals",
      "__asan_report_load4", "__asan_report_load8", "__asan_report_store4",
      "__asan_version_mismatch_check_v8", "__start_asan_globals", "__stop_asan_globals"};
  for (const auto &symbol : elf.symbols()) {
    auto flags = symbol.getFlags();
    auto name = symbol.getName();
    if (!flags || !name) {
      if (!flags) reject("invalid leaf symbol: " + llvm::toString(flags.takeError()));
      reject("invalid leaf symbol name: " + llvm::toString(name.takeError()));
    }
    if (name->empty()) continue;
    if (*flags & llvm::object::SymbolRef::SF_Undefined) {
      if (*name != "memset" && *name != "_GLOBAL_OFFSET_TABLE_" &&
          !(sanitized && sanitizer_imports.contains(name->str())))
        reject("source-pattern leaf has an unknown external effect/import: " + name->str());
      continue;
    }
    if (symbol.getBinding() == llvm::ELF::STB_LOCAL) continue;
    if (required.erase(name->str())) {
      if (symbol.getBinding() != llvm::ELF::STB_GLOBAL || symbol.getELFType() != llvm::ELF::STT_FUNC)
        reject("source-pattern leaf definition is not an exact strong function");
    } else if (!(sanitized && *name == "___asan_globals_registered" &&
                 (*flags & llvm::object::SymbolRef::SF_Common)))
      reject("source-pattern leaf has an extra exported definition: " + name->str());
  }
  if (!required.empty()) reject("source-pattern object lacks its two issued leaf functions");
}

void verifyCandidate(const Artifact &issued, const Installation &installation,
                     const std::string &soname) {
  auto parsed = nativeObject(issued, llvm::ELF::ET_DYN);
  auto original = nativeObject(installation.candidates, llvm::ELF::ET_DYN);
  const auto &elf = *llvm::cast<ElfObject>(parsed.get());
  const auto &stock = *llvm::cast<ElfObject>(original.get());
  if (exportSet(elf) != exportSet(stock))
    reject("source-pattern DSO exports differ from its pinned isolated adapter");
  const auto dynamic = dynamicInfo(elf, true), stock_dynamic = dynamicInfo(stock);
  if (dynamic.soname != soname || dynamic.soname == stock_dynamic.soname || !dynamic.symbolic)
    reject("source-pattern DSO lacks its unique request-bound symbolic ownership");
  auto runtime = nativeObject(installation.runtime, llvm::ELF::ET_DYN);
  auto allowed = stock_dynamic.needed;
  const auto runtime_soname = dynamicInfo(*llvm::cast<ElfObject>(runtime.get())).soname;
  if (runtime_soname.empty()) reject("canonical Runtime lacks its pinned SONAME");
  allowed.insert(runtime_soname);
  if (dynamic.needed != allowed || dynamic.needed.contains(stock_dynamic.soname))
    reject("source-pattern DSO dependencies differ from the pinned runtime/adapter closure");
  std::set<std::string> leaves{candidate::kStrictGemmSymbolV1, candidate::kStrictGemmCInterfaceV1};
  for (const auto &symbol : elf.symbols()) {
    auto name = symbol.getName();
    if (!name) reject("invalid candidate symbol: " + llvm::toString(name.takeError()));
    if (!leaves.contains(name->str())) continue;
    auto flags = symbol.getFlags();
    if (!flags) reject("invalid candidate symbol flags: " + llvm::toString(flags.takeError()));
    if ((*flags & llvm::object::SymbolRef::SF_Undefined) ||
        symbol.getBinding() != llvm::ELF::STB_LOCAL || symbol.getELFType() != llvm::ELF::STT_FUNC)
      reject("source-pattern leaf is not intrinsically local to its private DSO");
    leaves.erase(name->str());
  }
  if (!leaves.empty()) reject("source-pattern DSO lacks its local issued leaf/wrapper");
}

void boundField(std::string &text, const std::string &value) {
  text += std::to_string(value.size()) + ":" + value;
}

support::ProcessRequestV1 compilerProcess(const fs::path &working_directory) {
  support::ProcessRequestV1 result;
  result.working_directory = working_directory;
  result.environment = support::compiler_environment_sanitization_v1();
  result.environment.push_back({"TMPDIR", (working_directory / "tool-tmp").string()});
  for (const auto *name : {"COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH", "LD_RUN_PATH",
       "LDEMULATION", "GNUTARGET", "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"})
    result.environment.push_back({name, std::nullopt});
  return result;
}
void runCompiler(const support::ProcessRequestV1 &process, const char *phase) {
  auto result = support::run_process_v1(process);
  std::cerr << result.stderr_text;
  if (!result.launched || result.exit_code != 0)
    reject(std::string("source-pattern ") + phase + " failed: " + result.error);
}
void newPrivateOutput(const fs::path &path) {
  if (fs::symlink_status(path).type() != fs::file_type::not_found)
    reject("source-pattern issued output already exists: " + path.string());
}
} // namespace

OutputTileRequest parseOutputTileRequest(const std::string &text) {
  OutputTileRequest result;
  const auto comma = text.find(',');
  auto parse = [](const std::string &field, std::int64_t &value) {
    if (field.empty() || field.front() == '0' ||
        field.find_first_not_of("0123456789") != std::string::npos) return false;
    const auto parsed = std::from_chars(field.data(), field.data() + field.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == field.data() + field.size();
  };
  std::string error;
  if (comma == std::string::npos || !parse(text.substr(0, comma), result.m) ||
      !parse(text.substr(comma + 1), result.n) ||
      !matcore::mdslc::gemm_pattern::validateOutputTilePatternV1({result.m, result.n}, error))
    reject("--gemm-output-tiles requires canonical decimal M,N in [1,64], without leading zeros");
  return result;
}
void validateSourcePatternInvocation(codegen::ClosedCpuPolicy policy,
    codegen::ClosedHostOptimization optimization, bool compile_only) {
  if (policy != codegen::ClosedCpuPolicy::GeneratedStrict)
    reject("--gemm-output-tiles requires explicit --candidate generated-strict");
  if (compile_only) reject("--gemm-output-tiles is unavailable with -c; its private DSO requires complete executable linking");
  if (optimization != codegen::ClosedHostOptimization::None &&
      optimization != codegen::ClosedHostOptimization::PublicationReadForwarding)
    reject("--gemm-output-tiles is incompatible with strict-fused-pair");
}

SourcePatternBundle::SourcePatternBundle(const fs::path &output)
    : path_(output.string() + ".mdslc-private") {
  if (!output.is_absolute()) reject("source-pattern output must be an absolute new path");
  if (::mkdir(path_.c_str(), 0700) != 0)
    reject("source-pattern private bundle must be new: " + path_.string() + ": " + std::strerror(errno));
  std::string error;
  ownership_ = support::capture_file_snapshot_v1(path_, error);
  if (!error.empty() || !ownership_.identity) {
    // Do not perform a recursive cleanup without a proven directory owner.
    reject("cannot capture newly created source-pattern bundle: " + error);
  }
  try {
    unchangedOwnership();
    fs::create_directory(path_ / "tool-tmp");
  } catch (...) {
    try {
      unchangedOwnership();
      std::error_code ignored;
      fs::remove_all(path_, ignored);
    } catch (const std::exception &) {
      // A substituted path must not become a recursive deletion target.
    }
    throw;
  }
}
void SourcePatternBundle::unchangedOwnership() const {
  std::string error;
  const auto now = support::capture_file_snapshot_v1(path_, error);
  const auto status = fs::symlink_status(path_);
  if (!error.empty() || !now.exists || status.type() != fs::file_type::directory ||
      status.permissions() != fs::perms::owner_all ||
      !support::same_file_identity_v1(ownership_.identity, now.identity) ||
      ownership_.path_identity_chain != now.path_identity_chain)
    reject("source-pattern private bundle ownership/path changed: " + path_.string());
}
SourcePatternBundle::~SourcePatternBundle() {
  if (retained_) return;
  try {
    unchangedOwnership();
    std::error_code error;
    fs::remove_all(path_, error);
    if (error) std::cerr << "mdslc-region: cannot remove owned failed source-pattern bundle: " << error.message() << '\n';
  } catch (const std::exception &error) {
    std::cerr << "mdslc-region: preserved unverified source-pattern bundle path: " << error.what() << '\n';
  }
}

SourcePatternInstance SourcePatternInstance::issue(const Installation &installation,
    SourcePatternBundle &bundle, const frontend::AuthenticatedClosedRegionEvidence &evidence,
    const std::string &region, codegen::ClosedCpuPolicy policy,
    codegen::ClosedHostOptimization optimization, OutputTileRequest tiles) {
  return SourcePatternInstance(installation, bundle, evidence, region, policy, optimization, tiles, {});
}
SourcePatternInstance SourcePatternInstance::issueForTesting(const Installation &installation,
    SourcePatternBundle &bundle, const frontend::AuthenticatedClosedRegionEvidence &evidence,
    const std::string &region, codegen::ClosedCpuPolicy policy,
    codegen::ClosedHostOptimization optimization, OutputTileRequest tiles,
    const SourcePatternTestHooks &hooks) {
  return SourcePatternInstance(installation, bundle, evidence, region, policy, optimization, tiles, hooks);
}
SourcePatternInstance::SourcePatternInstance(const Installation &installation,
    SourcePatternBundle &bundle, const frontend::AuthenticatedClosedRegionEvidence &evidence,
    const std::string &region, codegen::ClosedCpuPolicy policy,
    codegen::ClosedHostOptimization optimization, OutputTileRequest tiles,
    const SourcePatternTestHooks &hooks)
    : installation_(&installation), bundle_(&bundle),
      host_(frontend::detail::ClosedRegionCompilationAccess::host(evidence)),
      kit_(Artifact::capture(installation.installed.build_tree ? fs::path(REGION_BUILD_RELINK_KIT) :
          installation.installed.candidates.parent_path() / REGION_RELINK_KIT_FILENAME, REGION_RELINK_KIT_SHA)),
      exports_(Artifact::capture(installation.installed.build_tree ? fs::path(REGION_BUILD_EXPORTS) :
          installation.installed.candidates.parent_path() / REGION_EXPORTS_FILENAME, REGION_EXPORTS_SHA)) {
  validateSourcePatternInvocation(policy, optimization, false);
  if (!host_ || !evidence.entryBinding() || evidence.program().regions.size() != 1 ||
      evidence.program().regions.front().name != region)
    reject("source-pattern instance requires its authenticated named source closure");
  verifyKit(kit_);
  unchanged();
  mlir::MLIRContext context;
  auto artifact = candidate::issueStrictGemmArtifactV1(context, bool(REGION_SANITIZED),
      candidate::StrictGemmScheduleV1::OutputTiledMKN, target,
      candidate::StrictCpuIsaV1::Baseline, {tiles.m, tiles.n});
  if (!artifact) reject(artifact.error);
  const std::vector<std::pair<std::string, std::string>> stages{
      {"gemm.ll", artifact.llvm_ir}, {"gemm.ll.manifest", artifact.manifest},
      {"gemm.ll.semantic.mlir", artifact.semantic_ir}, {"gemm.ll.structured.mlir", artifact.structured_ir},
      {"gemm.ll.bufferized.mlir", artifact.bufferized_ir}, {"gemm.ll.scheduled.mlir", artifact.scheduled_ir},
      {"gemm.ll.transform.mlir", artifact.transform_ir}};
  for (const auto &[name, bytes] : stages) {
    const auto path = bundle.path() / name;
    newPrivateOutput(path);
    write(path, bytes);
    const auto sha = digest(bytes);
    issued_.push_back(Artifact::capture(path, sha.c_str()));
  }
  if (hooks.after_issue) hooks.after_issue(bundle.path());
  unchanged();
  const auto object_path = bundle.path() / "gemm.o";
  newPrivateOutput(object_path);
  auto process = compilerProcess(bundle.path());
  process.argv = {REGION_CLANG, "--no-default-config", "-resource-dir=" REGION_RESOURCE_DIR,
      "-c", "-x", "ir", (bundle.path() / "gemm.ll").string(), "-O2", "-fPIC", "-ffp-contract=off",
      "--target=" + std::string(candidate::cpuTargetTripleV1(target)), "-o", object_path.string()};
  if (REGION_SANITIZED) process.argv.push_back("-fsanitize=address");
  runCompiler(process, "leaf compilation");
  issued_.push_back(Artifact::capture(object_path));
  verifyLeafObject(issued_.back(), bool(REGION_SANITIZED));
  if (hooks.after_object) hooks.after_object(bundle.path());
  unchanged();

  std::string manifest = "schema=matcore-source-output-pattern-v1\nsource_authority=authenticated_in_process_source_closure\n";
  auto add = [&](const std::string &key, const std::string &value) { manifest += key + "=" + value + "\n"; };
  add("source_context_sha256", host_->identity());
  add("region_name_hex", llvm::toHex(llvm::arrayRefFromStringRef(region), true));
  add("candidate", "generated-strict");
  add("optimization", codegen::closedHostOptimizationName(optimization));
  add("pattern_parameters", "M:" + std::to_string(tiles.m) + ",N:" + std::to_string(tiles.n) + ",K:untiled");
  add("toolchain", "21.1.8"); add("target", candidate::cpuTargetTripleV1(target));
  add("address_sanitizer", REGION_SANITIZED ? "function_attributes_and_object" : "off");
  add("clang_sha256", installation.clang.sha); add("linker_sha256", installation.linker.sha);
  add("relink_kit_sha256", kit_.sha); add("exports_sha256", exports_.sha);
  add("canonical_runtime_sha256", installation.runtime.sha); add("stock_candidates_sha256", installation.candidates.sha);
  add("public_header_sha256", installation.public_header.sha);
  add("storage_header_sha256", installation.storage_header.sha);
  add("private_header_sha256", installation.private_header.sha);
  add("provider_sha256", installation.provider ? installation.provider->sha : "none");
  add("accelerator_count", std::to_string(installation.accelerators.size()));
  for (std::size_t index = 0; index < installation.accelerators.size(); ++index)
    add("accelerator_" + std::to_string(index) + "_sha256", installation.accelerators[index].sha);
  add("issuer_manifest_sha256", digest(artifact.manifest)); add("semantic_sha256", digest(artifact.semantic_ir));
  add("structured_sha256", digest(artifact.structured_ir)); add("bufferized_sha256", digest(artifact.bufferized_ir));
  add("transform_sha256", digest(artifact.transform_ir)); add("scheduled_sha256", digest(artifact.scheduled_ir));
  add("llvm_sha256", digest(artifact.llvm_ir)); add("object_sha256", issued_.back().sha);
  std::string binding;
  boundField(binding, "matcore-source-output-pattern-binding-v1"); boundField(binding, manifest);
  const auto identity = digest(binding);
  add("binding_sha256", identity);
  soname_ = "libmdslc-pattern-" + identity + ".so";
  add("candidate_soname", soname_);
  const auto dso_path = bundle.path() / soname_;
  newPrivateOutput(dso_path);
  process = compilerProcess(bundle.path());
  process.argv = {REGION_CLANG, "--no-default-config", "-resource-dir=" REGION_RESOURCE_DIR,
      "--ld-path=" REGION_LINKER, "--target=" + std::string(candidate::cpuTargetTripleV1(target)),
      "-shared", "-o", dso_path.string(), "-Xlinker", "--whole-archive", kit_.path.string(),
      "-Xlinker", "--no-whole-archive", object_path.string(), "-Xlinker", "-Bsymbolic",
      "-Xlinker", "--version-script=" + exports_.path.string(), "-Xlinker", "-soname",
      "-Xlinker", soname_, "-Xlinker", "--no-as-needed", installation.runtime.path.string(),
      "-Xlinker", "--as-needed", "-lm", "-pthread"};
  if (!REGION_SANITIZED) {
    process.argv.insert(process.argv.end(), {"-Xlinker", "--no-undefined"});
  }
  for (const auto &accelerator : installation.accelerators) process.argv.push_back(accelerator.path.string());
  runCompiler(process, "private candidate link");
  candidates_ = Artifact::capture(dso_path);
  verifyCandidate(candidates_, installation, soname_);
  if (hooks.after_dso) hooks.after_dso(bundle.path());
  unchanged();
  add("dso_sha256", candidates_.sha);
  add("manifest_authority", "inspection_only_never_imported");
  const auto manifest_path = bundle.path() / "source-pattern.manifest";
  newPrivateOutput(manifest_path);
  write(manifest_path, manifest);
  const auto manifest_sha = digest(manifest);
  issued_.push_back(Artifact::capture(manifest_path, manifest_sha.c_str()));
  compilation_binding_ = std::shared_ptr<const codegen::SourcePatternCompilationBinding>(
      new codegen::SourcePatternCompilationBinding(evidence, candidates_.sha, policy, optimization,
          !installation.accelerators.empty(), bool(REGION_SANITIZED)));
  unchanged();
}

void SourcePatternInstance::unchanged() const {
  bundle_->unchangedOwnership();
  std::string error;
  if (!host_ || !host_->unchanged(error)) reject(error.empty() ? "source-pattern closure is missing" : error);
  installation_->unchanged(); kit_.unchanged(); exports_.unchanged();
  for (const auto &artifact : issued_) artifact.unchanged();
  if (!candidates_.path.empty()) candidates_.unchanged();
}
codegen::ExperimentalCompilerInputs SourcePatternInstance::compilerInputs(const fs::path &staging,
    const frontend::AuthenticatedClosedRegionEvidence &evidence) const {
  unchanged();
  auto inputs = installation_->compilerInputs(staging);
  for (auto &artifact : inputs.symbol_artifacts)
    if (artifact.owner == codegen::SymbolArtifactOwner::PrivateCandidates)
      artifact.bytes = llvm::MemoryBufferRef(candidates_.bytes, "source-issued isolated private candidates");
  inputs.source_pattern_binding = compilation_binding_;
  std::string error;
  if (!compilation_binding_ || !compilation_binding_->matches(evidence, inputs,
          codegen::ClosedCpuPolicy::GeneratedStrict,
          // The immutable binding independently rejects a later compiler call
          // with another optimization. Here it checks the actual source seal.
          compilation_binding_->optimization_, error))
    reject(error.empty() ? "source-pattern compilation binding is missing" : error);
  return inputs;
}
void SourcePatternInstance::verifyExecutable(const Artifact &artifact) const {
  auto parsed = llvm::object::ObjectFile::createObjectFile(
      llvm::MemoryBufferRef(artifact.bytes, "source-pattern final executable"));
  if (!parsed) reject("invalid source-pattern executable: " + llvm::toString(parsed.takeError()));
  const auto *elf = llvm::dyn_cast<ElfObject>(parsed->get());
  if (!elf || elf->getELFFile().getHeader().e_machine != machine)
    reject("source-pattern executable target does not match its private candidate");
  // Executables intentionally have pinned RUNPATHs, unlike their private DSO.
  auto entries = elf->getELFFile().dynamicEntries();
  if (!entries) reject("invalid executable dynamic table: " + llvm::toString(entries.takeError()));
  llvm::StringRef strings;
  for (const auto &section : elf->sections()) {
    auto name = section.getName();
    if (!name) reject("invalid executable section: " + llvm::toString(name.takeError()));
    if (*name == ".dynstr") {
      auto contents = section.getContents();
      if (!contents) reject("invalid executable dynamic strings: " + llvm::toString(contents.takeError()));
      strings = *contents;
    }
  }
  unsigned selected = 0;
  for (const auto &entry : *entries) {
    if (entry.d_tag != llvm::ELF::DT_NEEDED) continue;
    if (entry.d_un.d_val >= strings.size()) reject("invalid executable dependency index");
    const auto text = strings.drop_front(entry.d_un.d_val);
    const auto end = text.find('\0');
    if (end == llvm::StringRef::npos) reject("unterminated executable dependency");
    const auto needed = text.take_front(end);
    selected += needed == soname_;
    if (needed == installation_->installed.candidates.filename().string() ||
        (needed.starts_with("libmdslc-pattern-") && needed != soname_))
      reject("source-pattern executable links a stock or foreign candidate DSO");
  }
  if (selected != 1) reject("source-pattern executable does not own exactly its requested candidate DSO");
}
} // namespace matcore::mdslc::driver
