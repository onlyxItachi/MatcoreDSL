#ifndef MATCORE_MDSLC_CLOSED_CUDA_CANDIDATE_V1_H
#define MATCORE_MDSLC_CLOSED_CUDA_CANDIDATE_V1_H

#include "closed_host_v1.h"

namespace matcore::mdslc::runtime::closed_host_v1::detail {
// Force-only, compiler-bound sm_89 implementation. Driver discovery creates no
// context. Every driver operation runs on a fresh joined worker, preserving
// caller context/TLS/FP state. Execution uses a fresh private context and owned
// transfer staging. Output must be private Session-owned storage; it changes
// only after checked execution, copy-back and resource cleanup all succeed.
Code cudaCandidateAvailable() noexcept;
Code cudaGemmCandidate(CandidateInput, CandidateInput, CandidateOutput) noexcept;
// Separate compiler-issued combined realization. Uses the same poison domain
// as the single-GEMM route; one private device panel, one kernel, no C Value.
// These entry points do not grant source authority or permit image injection.
// Nonempty E requires N>0; Session owns the semantic N=0 positive-zero bypass.
Code cudaFusedPairImageAvailable() noexcept;
Code cudaFusedPairCandidate(CandidateInput a, CandidateInput b, CandidateInput d,
                           CandidateOutput e) noexcept;
} // namespace matcore::mdslc::runtime::closed_host_v1::detail
#endif
