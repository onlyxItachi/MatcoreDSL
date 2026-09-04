#ifndef MATCORE_MDSLC_FRONTEND_H
#define MATCORE_MDSLC_FRONTEND_H

#include "../ir/matcore_ir.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace matcore::mdslc::mlir_bridge::detail {
class AuthenticatedNativeFrontendEvidenceAccessV1;
}

namespace matcore::mdslc::frontend {

namespace detail {
class NativeFrontendEvidenceIssuerV1;
}

struct Diagnostic {
  std::string file;
  unsigned line = 0;
  unsigned column = 0;
  std::string message;
};

struct Options {
  std::string input_path;
  std::string clang_path = "/usr/bin/clang++-21";
  // Authenticated Clang builtin-header directory. It is an implementation
  // input, not part of the source-level compilation identity.
  std::string clang_resource_directory;
  std::vector<std::string> compiler_arguments;
  std::vector<std::string> trusted_public_headers;
  std::size_t maximum_ast_bytes = 512U * 1024U * 1024U;
  // Native-only, diagnostic recovery experiment. This never authorizes a
  // source rewrite and never adds a recovered operation to Matcore IR v0/v1.
  bool inspect_recovered_cpp_gemm = false;
  bool verbose = false;
};

enum class RecoveredGemmState {
  not_recognized,
  recognized_rejected,
  recognized_guard_required,
  raised,
};

struct RecoveredSourceRange {
  std::uint64_t begin = 0;
  std::uint64_t end = 0;
};

struct RecoveredNamedRange {
  std::string role;
  RecoveredSourceRange range;
};

struct RecoveredFpProof {
  bool allow_reassociation = false;
  bool contract_across_statement = false;
  bool honor_nans = true;
  bool honor_infinities = true;
  bool preserve_signed_zero = true;
  bool allow_reciprocal = false;
  bool allow_approximate_functions = false;
  bool fenv_access = false;
  bool fast_math_profile = false;
  std::string evaluation_method;
  std::string rounding_mode;
  std::string exception_mode;
  std::string denormal_mode;
  std::string fp32_denormal_mode;
  unsigned optimization_level = 0;
};

// Typed, diagnostic-only record for conservative recovery. It is not a
// serialized optimizer IR and is deliberately unable to masquerade as an
// explicit matcore::mdsl declaration or Matcore IR v1 capture record.
struct RecoveredGemmCandidate {
  RecoveredGemmState state = RecoveredGemmState::not_recognized;
  std::string pattern = "canonical-row-major-f32-gemm-v1";
  std::string site_id;
  std::string source_file;
  std::string source_identity;
  std::string compilation_identity;
  std::string source_snapshot_sha256;
  std::string function_name;
  // Populated only after the exact typed loop contract has been authenticated.
  // A merely plausible or not-recognized loop must not claim GEMM semantics.
  std::string semantic_contract;
  std::string output_parameter;
  std::string lhs_parameter;
  std::string rhs_parameter;
  std::string m_parameter;
  std::string n_parameter;
  std::string k_parameter;
  unsigned line = 0;
  unsigned column = 0;
  std::uint64_t offset = 0;
  RecoveredSourceRange outer_loop_range;
  std::vector<RecoveredNamedRange> proof_ranges;
  RecoveredFpProof fp_proof;
  std::vector<std::string> required_runtime_guards;
  std::vector<std::string> rejection_reasons;
};

// Native-only, immutable-by-API evidence issued after a successful LibTooling
// extraction. It seals copies of every Result field plus the effective Options
// used for that extraction. It is deliberately non-default-constructible and
// non-aggregate: mutable diagnostic Result fields are not an authorization
// boundary.
class AuthenticatedNativeFrontendEvidenceV1 {
public:
  AuthenticatedNativeFrontendEvidenceV1(
      const AuthenticatedNativeFrontendEvidenceV1 &) = default;
  AuthenticatedNativeFrontendEvidenceV1 &operator=(
      const AuthenticatedNativeFrontendEvidenceV1 &) = default;
  AuthenticatedNativeFrontendEvidenceV1(
      AuthenticatedNativeFrontendEvidenceV1 &&) noexcept = default;
  AuthenticatedNativeFrontendEvidenceV1 &operator=(
      AuthenticatedNativeFrontendEvidenceV1 &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept;

private:
  struct Payload;

  explicit AuthenticatedNativeFrontendEvidenceV1(
      std::shared_ptr<const Payload> payload)
      : payload_(std::move(payload)) {}

  std::shared_ptr<const Payload> payload_;

  friend class detail::NativeFrontendEvidenceIssuerV1;
  friend class ::matcore::mdslc::mlir_bridge::detail::
      AuthenticatedNativeFrontendEvidenceAccessV1;
};

struct Result {
  ir::Module module;
  std::vector<Diagnostic> diagnostics;
  // Exact bytes parsed for source ranges and later consumed by codegen.
  std::string source_snapshot;
  std::vector<RecoveredGemmCandidate> recovered_gemm_candidates;
  // Present only for a successful native extraction requested in explicit
  // recovery-inspection mode. Ordinary compilation does not pay the cost of
  // the sealed source snapshot copy.
  std::optional<AuthenticatedNativeFrontendEvidenceV1> native_evidence;
};

class Frontend {
public:
  virtual ~Frontend() = default;
  virtual bool extract(const Options &options, Result &result) = 0;
};

// Bootstrap-only fallback. It executes Clang parsing/Sema and consumes Clang's
// structural JSON AST. The interface is intentionally independent of this
// representation so LibTooling can replace it without changing the IR layer.
std::unique_ptr<Frontend> createClangAstJsonBootstrapFrontend();

// Supported frontend. It runs Clang parsing/Sema in-process and authenticates
// operation declarations and source ranges through LibTooling APIs.
std::unique_ptr<Frontend> createClangLibToolingFrontend();

// Internal startup attestation for the Clang implementation actually loaded
// into matcore-extract. The path query is available on Linux shared-runtime
// builds and fails closed when the loaded clang-cpp DSO is ambiguous.
std::string nativeClangRuntimeVersionV1();
std::optional<std::string> nativeClangRuntimeLibraryPathV1(
    std::string &error);

// Shared naming contract used by every frontend producer.
std::string stableSourceIdentity(const std::string &canonical_input);
std::string stableCompilationIdentity(const Options &options);
std::string makeStableSiteId(std::string_view source_identity,
                             std::string_view compilation_identity,
                             std::string_view source, std::uint64_t offset,
                             std::string_view kind);
std::string_view recoveredGemmStateName(RecoveredGemmState state);
std::string serializeRecoveredGemmInspection(const Result &result);

} // namespace matcore::mdslc::frontend

namespace matcore::mdslc::mlir_bridge::detail {

// Internal bridge access to the native frontend's immutable evidence payload.
// Definitions live with the native issuer, where the private payload is
// complete. Callers cannot construct or modify the evidence through this seam.
class AuthenticatedNativeFrontendEvidenceAccessV1 {
public:
  static const frontend::Options &
  options(const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence);
  static const frontend::Result &
  result(const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence);
};

} // namespace matcore::mdslc::mlir_bridge::detail

#endif // MATCORE_MDSLC_FRONTEND_H
