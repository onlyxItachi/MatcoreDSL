#ifndef MATCORE_MDSLC_PLATFORM_FP_COMPILE_CONTRACT_V1_H
#define MATCORE_MDSLC_PLATFORM_FP_COMPILE_CONTRACT_V1_H

#if !defined(MATCORE_MDSLC_PRECISE_FP_PROFILE) || \
    MATCORE_MDSLC_PRECISE_FP_PROFILE != 1
#error "MDSLC CPU execution code requires its target-local precise FP profile"
#endif

#if defined(__FAST_MATH__)
#error "MDSLC CPU execution code must not be compiled with fast-math"
#endif

#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0
#error "MDSLC CPU execution code must preserve NaN and infinity semantics"
#endif

#if defined(_M_FP_FAST)
#error "MDSLC CPU execution code must not be compiled with /fp:fast"
#endif

#endif
