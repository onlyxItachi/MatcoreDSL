#ifndef MATCORE_MDSLC_CLOSED_ROCDL_CANDIDATE_V1_H
#define MATCORE_MDSLC_CLOSED_ROCDL_CANDIDATE_V1_H

#include "closed_host_v1.h"

namespace matcore::mdslc::runtime::closed_host_v1::detail {
// Private forced candidate, not source authority or a public residency API.
// HIP discovery runs on an isolated joined thread, including for empty math.
Code rocdlCandidateAvailable() noexcept;
// Immutable, live contiguous host inputs; isolated private host output. The
// function stages through private device storage and returns only after checked
// completion/cleanup. Output stays unchanged on every defined failure return.
// Unknown GPU completion quarantines owned staging/resources and permanently
// poisons this runtime instance's AMD candidate; it never frees possibly live
// device/host buffers, silently falls back, resets a device or changes its flags.
Code rocdlGemmCandidate(CandidateInput lhs, CandidateInput rhs,
                        CandidateOutput output) noexcept;
} // namespace matcore::mdslc::runtime::closed_host_v1::detail

#endif
