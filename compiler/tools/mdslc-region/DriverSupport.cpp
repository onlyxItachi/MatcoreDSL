#include "DriverSupport.h"
#include "region_driver_paths.h"
#include "region_driver_identity.h"
#include "ExperimentalRegionCompiler.h"
#include "ClosedRegionAdmission.h"
#include "platform_support.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace matcore::mdslc::driver {
[[noreturn]] void reject(const std::string &message) { throw std::runtime_error(message); }

std::string digest(const std::string &bytes) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(bytes)), true);
}
bool same(const support::FileSnapshotV1 &a, const support::FileSnapshotV1 &b) {
  return a.version == b.version && a.exists == b.exists && a.regular_file == b.regular_file &&
    a.normalized_path == b.normalized_path && a.size_bytes == b.size_bytes &&
    a.last_write_time_ticks == b.last_write_time_ticks &&
    support::same_file_identity_v1(a.identity, b.identity) && a.path_identity_chain == b.path_identity_chain;
}
Artifact Artifact::capture(const fs::path &path, const char *expected) {
    std::string error;
    auto before = support::capture_file_snapshot_v1(path, error);
    if (!error.empty() || !before.exists || !before.regular_file || !before.identity ||
        before.size_bytes > 512ULL * 1024 * 1024)
      reject("invalid or oversized compiler artifact: " + path.string() + ": " + error);
    auto buffer = llvm::MemoryBuffer::getFile(path.string());
    if (!buffer) reject("cannot read compiler input: " + path.string());
    auto bytes = (*buffer)->getBuffer().str();
    auto sha = digest(bytes);
    auto after = support::capture_file_snapshot_v1(path, error);
    if (!error.empty() || !same(before, after) || (expected && sha != expected))
      reject("compiler artifact changed or differs from its compiled identity: " + path.string());
    return {path, std::move(after), std::move(sha), std::move(bytes)};
  }
void Artifact::unchanged() const {
    auto now = capture(path);
    if (!same(snapshot, now.snapshot) || sha != now.sha)
      reject("compiler artifact changed during compilation: " + path.string());
  }
void verifyOutput(const Artifact &artifact, bool compile_only) {
  auto parsed = llvm::object::ObjectFile::createObjectFile(
      llvm::MemoryBufferRef(artifact.bytes, "issued compiler output"));
  if (!parsed) reject("compiler output is not an object: " + llvm::toString(parsed.takeError()));
  const auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(parsed->get());
  if (!elf || elf->getELFFile().getHeader().e_machine != llvm::ELF::EM_X86_64)
    reject("compiler output is not a Linux x86-64 ELF artifact");
  const auto &file = elf->getELFFile();
  const auto &header = file.getHeader();
  if (compile_only) {
    if (header.e_type != llvm::ELF::ET_REL)
      reject("compiler object output is not relocatable ELF");
    return;
  }
  if ((header.e_type != llvm::ELF::ET_EXEC && header.e_type != llvm::ELF::ET_DYN) ||
      !header.e_entry || (fs::status(artifact.path).permissions() & fs::perms::owner_exec) == fs::perms::none)
    reject("compiler executable output lacks executable ELF identity");
  auto segments = file.program_headers();
  if (!segments) reject("compiler executable has invalid segments: " + llvm::toString(segments.takeError()));
  for (const auto &segment : *segments)
    if (segment.p_type == llvm::ELF::PT_LOAD && (segment.p_flags & llvm::ELF::PF_X) &&
        header.e_entry >= segment.p_vaddr && header.e_entry - segment.p_vaddr < segment.p_memsz)
      return;
  reject("compiler executable entry is not in an executable load segment");
}
Layout layout() {
  std::string error;
  auto executable = support::current_executable_path_v1(error);
  if (!executable) reject(error);
  // Build-tree use is available only at the exact configured executable path.
  // A copied/installed driver must never fall back into the source checkout.
  if (support::paths_refer_to_same_location_v1(*executable, REGION_BUILD_DRIVER, error))
    return {REGION_BUILD_INCLUDE, REGION_BUILD_PRIVATE_HEADER, REGION_BUILD_CANDIDATES, REGION_BUILD_RUNTIME, true};
  if (!error.empty()) reject(error);
  const auto prefix = (executable->parent_path() / REGION_PREFIX_FROM_BIN).lexically_normal();
  const auto private_dir = prefix / REGION_INSTALL_PRIVATE;
  return {prefix / REGION_INSTALL_INCLUDE, private_dir / "include/closed_host_v1.h",
          private_dir / REGION_CANDIDATES_FILENAME,
          prefix / REGION_INSTALL_LIB / REGION_RUNTIME_FILENAME};
}

Staging::Staging(const fs::path &parent) {
    auto pattern = (parent / ".mdslc-region-XXXXXX").string();
    std::vector<char> bytes(pattern.begin(), pattern.end()); bytes.push_back('\0');
    auto *created = ::mkdtemp(bytes.data());
    if (!created) reject("cannot create private output staging: " + std::string(std::strerror(errno)));
    path = created;
    fs::create_directory(path / "helper");
    fs::create_directory(path / "tool-tmp");
  }
Staging::~Staging() { std::error_code ignored; fs::remove_all(path, ignored); }

void write(const fs::path &path, const std::string &bytes) {
  std::ofstream out(path, std::ios::binary); out << bytes; out.close();
  if (!out) reject("cannot write compiler-owned artifact: " + path.string());
}

void validateNewOutput(const fs::path &output) {
  std::string error;
  if (!support::prospective_output_path_supported_v1(output, error)) reject(error);
  if (!fs::is_directory(output.parent_path())) reject("output parent does not exist");
  if (fs::symlink_status(output).type() != fs::file_type::not_found)
    reject("output already exists; use a new output path (no overwrite is performed)");
}
codegen::ClosedCpuPolicy parseCandidatePolicy(const std::string &name) {
  if (name == "automatic") return codegen::ClosedCpuPolicy::Automatic;
  if (name == "native-strict") return codegen::ClosedCpuPolicy::NativeStrict;
  if (name == "generated-strict") return codegen::ClosedCpuPolicy::GeneratedStrict;
  if (name == "generated-reassociate") return codegen::ClosedCpuPolicy::GeneratedReassociate;
  if (name == "existing-native") return codegen::ClosedCpuPolicy::ExistingNative;
  if (name == "openblas") return codegen::ClosedCpuPolicy::OpenBLAS;
  reject("unknown built-in candidate: " + name);
}
const char *candidatePolicyUsage() {
  return "automatic|native-strict|generated-strict|generated-reassociate|existing-native|openblas";
}
Installation::Installation()
    : installed(layout()), clang(Artifact::capture(REGION_CLANG, REGION_CLANG_SHA)),
      linker(Artifact::capture(REGION_LINKER, REGION_LINKER_SHA)),
      candidates(Artifact::capture(installed.candidates,
          installed.build_tree ? REGION_CANDIDATES_SHA : REGION_INSTALLED_CANDIDATES_SHA)),
      runtime(Artifact::capture(installed.runtime,
          installed.build_tree ? REGION_RUNTIME_SHA : REGION_INSTALLED_RUNTIME_SHA)),
      provider(std::strlen(REGION_PROVIDER_PATH)
          ? std::optional<Artifact>(Artifact::capture(REGION_PROVIDER_PATH, REGION_PROVIDER_SHA))
          : std::nullopt),
      public_header(Artifact::capture(installed.include / "matcore/region.h")),
      storage_header(Artifact::capture(installed.include / "matcore/detail/region_storage.h")),
      private_header(Artifact::capture(installed.header)) {}
std::vector<codegen::TrustedSymbolArtifact> Installation::symbolArtifacts() const {
  std::vector<codegen::TrustedSymbolArtifact> result{
      {codegen::SymbolArtifactOwner::MatcoreRuntime,
       llvm::MemoryBufferRef(runtime.bytes, "canonical Matcore Runtime")},
      {codegen::SymbolArtifactOwner::PrivateCandidates,
       llvm::MemoryBufferRef(candidates.bytes, "isolated private candidates")}};
  if (provider) result.push_back({codegen::SymbolArtifactOwner::ExternalProvider,
      llvm::MemoryBufferRef(provider->bytes, "canonical OpenBLAS provider")});
  return result;
}
void Installation::unchanged() const {
  for (const auto *artifact : {&candidates, &runtime, &public_header, &storage_header, &private_header, &clang, &linker})
    artifact->unchanged();
  if (provider) provider->unchanged();
}
frontend::Options Installation::options(const fs::path &source, const std::vector<std::string> &host_options) const {
  frontend::Options result;
  result.input_path = source.string(); result.clang_path = REGION_CLANG;
  result.clang_resource_directory = REGION_RESOURCE_DIR;
  result.compiler_arguments = host_options;
  result.compiler_arguments.push_back("-I" + installed.include.string());
  if (REGION_SANITIZED) result.compiler_arguments.push_back("-fsanitize=address,undefined");
  return result;
}
codegen::ExperimentalCompilerInputs Installation::compilerInputs(const fs::path &staging) const {
  return {REGION_CLANG, REGION_RESOURCE_DIR, installed.include, installed.header,
          staging, bool(REGION_SANITIZED), symbolArtifacts()};
}
void compileAndPublish(const Installation &installation, const Staging &staging,
    const fs::path &output, bool compile_only, const std::string &derived_ir,
    const std::function<void()> &inputs_unchanged) {
  const auto &installed = installation.installed;
  const auto &provider = installation.provider;
  const auto ir = staging.path / "host.ll", binary = staging.path / "result";
  auto unchanged = [&] { inputs_unchanged(); installation.unchanged(); };
  unchanged();
  write(ir, derived_ir);
  auto issued_ir = Artifact::capture(ir);
  support::ProcessRequestV1 process;
  process.working_directory = staging.path;
  process.environment = support::compiler_environment_sanitization_v1();
  // Clang may create/delete a temporary object when linking LLVM input. An
  // inherited TMPDIR can be the captured host working/include directory; our
  // own compilation would then invalidate its metadata. This pre-created
  // private child is outside the source input set, without exempting any input
  // directory from unchanged verification.
  process.environment.push_back({"TMPDIR", (staging.path / "tool-tmp").string()});
  // These are tool/library search inputs, not source semantics. Only configured
  // system-toolchain paths and explicit Matcore artifacts may reach final link.
  for (const auto *name : {"COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH",
       "LD_RUN_PATH", "LDEMULATION", "GNUTARGET", "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH"})
    process.environment.push_back({name, std::nullopt});
  process.argv = {REGION_CLANG, "--no-default-config", "-resource-dir=" REGION_RESOURCE_DIR,
                  "-x", "ir", ir.string(), "-O2", "-o", binary.string()};
  if (REGION_SANITIZED) process.argv.push_back("-fsanitize=address,undefined");
  if (compile_only) process.argv.push_back("-c");
  else {
    process.argv.push_back("--ld-path=" REGION_LINKER);
    // Both owning DSOs are explicit dependencies. Runtime is not an indirect
    // RUNPATH search through the private library's installation directory.
    process.argv.insert(process.argv.end(), {"-x", "none", "-Xlinker", "--no-as-needed",
      installed.candidates.string(), installed.runtime.string(),
      "-lm", "-pthread", "-Xlinker", "-rpath", "-Xlinker", installed.runtime.parent_path().string(),
      "-Xlinker", "-rpath", "-Xlinker", installed.candidates.parent_path().string()});
    // Runtime's installed image has no RPATH. Bind its already-authenticated
    // provider as a direct dependency too: executable RUNPATH does not cover
    // a transitive Runtime -> provider lookup. Never rediscover it with -l.
    if (provider)
      process.argv.insert(process.argv.end(), {provider->path.string(),
        "-Xlinker", "-rpath", "-Xlinker", provider->path.parent_path().string()});
  }
  auto linked = support::run_process_v1(process);
  std::cerr << linked.stderr_text;
  if (!linked.launched || linked.exit_code != 0) reject("ordinary Clang artifact compilation/link failed: " + linked.error);
  issued_ir.unchanged(); unchanged();
  const auto final = Artifact::capture(binary);
  verifyOutput(final, compile_only);
  // Same filesystem, atomic no-clobber publication. An output racing into
  // existence is never overwritten; failed compiles leave no published file.
  fs::create_hard_link(binary, output);
}
} // namespace matcore::mdslc::driver
