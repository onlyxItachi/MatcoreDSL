#include "native_frontend_evidence_internal.h"

#include <memory>
#include <utility>

namespace matcore::mdslc::frontend {

struct AuthenticatedNativeFrontendEvidenceV1::Payload {
  Options options;
  Result result;
};

bool AuthenticatedNativeFrontendEvidenceV1::valid() const noexcept {
  return static_cast<bool>(payload_);
}

AuthenticatedNativeFrontendEvidenceV1
detail::NativeFrontendEvidenceIssuerV1::issue(
    const Result &result, const Options &effective_options) {
  auto payload =
      std::make_shared<AuthenticatedNativeFrontendEvidenceV1::Payload>();
  payload->options = effective_options;
  payload->result = result;
  return AuthenticatedNativeFrontendEvidenceV1(std::move(payload));
}

} // namespace matcore::mdslc::frontend

namespace matcore::mdslc::mlir_bridge::detail {

const frontend::Options &AuthenticatedNativeFrontendEvidenceAccessV1::options(
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence) {
  return evidence.payload_->options;
}

const frontend::Result &AuthenticatedNativeFrontendEvidenceAccessV1::result(
    const frontend::AuthenticatedNativeFrontendEvidenceV1 &evidence) {
  return evidence.payload_->result;
}

} // namespace matcore::mdslc::mlir_bridge::detail
