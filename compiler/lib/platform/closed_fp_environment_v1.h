#ifndef MATCORE_MDSLC_CLOSED_FP_ENVIRONMENT_V1_H
#define MATCORE_MDSLC_CLOSED_FP_ENVIRONMENT_V1_H

#include <cfenv>
#include <cstdint>

namespace matcore::mdslc::platform {

constexpr bool closedFpPlatformSupportedV1() noexcept {
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
  return true;
#else
  return false;
#endif
}

// A deliberately exact baseline profile: nearest-even, gradual underflow,
// masked traps, no default-NaN or alternative FP behavior. Reject all unknown
// nonzero bits rather than overlooking new controls such as FIZ/AH.
constexpr bool closedAarch64FpControlCompatibleV1(std::uint64_t fpcr) noexcept {
  return fpcr == 0;
}

// Internal synchronous closed-region scope, distinct from legacy control-only
// inspection/restoration. Captures the complete supported thread FP controls
// AND status, normalizes privately, and restores exactly before normal return.
// No public source authority, candidate availability or provider proof follows.
// A failed restoration terminates: returning normally would break the contract.
class ClosedFpEnvironmentV1 {
public:
  ClosedFpEnvironmentV1() noexcept;
  ~ClosedFpEnvironmentV1();
  ClosedFpEnvironmentV1(const ClosedFpEnvironmentV1 &) = delete;
  ClosedFpEnvironmentV1 &operator=(const ClosedFpEnvironmentV1 &) = delete;
  ClosedFpEnvironmentV1(ClosedFpEnvironmentV1 &&) = delete;
  ClosedFpEnvironmentV1 &operator=(ClosedFpEnvironmentV1 &&) = delete;
  bool valid() const noexcept { return valid_; }
  bool controlsUnchanged() const noexcept;
  void restore() noexcept;

private:
  bool captured_ = false;
  bool valid_ = false;
#if defined(__linux__) && defined(__x86_64__)
  std::fenv_t saved_{};
  std::uint32_t saved_mxcsr_ = 0, expected_mxcsr_ = 0;
  std::uint16_t saved_control_ = 0, saved_status_ = 0, expected_control_ = 0;
#elif defined(__linux__) && defined(__aarch64__)
  std::uint64_t saved_fpcr_ = 0, saved_fpsr_ = 0;
#endif
};

} // namespace matcore::mdslc::platform
#endif
