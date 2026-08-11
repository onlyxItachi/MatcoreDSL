#ifndef MATCORE_MDSLC_PLATFORM_FP_COMPILE_CONTRACT_V1_H
#define MATCORE_MDSLC_PLATFORM_FP_COMPILE_CONTRACT_V1_H

#include <float.h>

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

#if !defined(FLT_EVAL_METHOD)
#error "MDSLC CPU execution code requires a verifiable FP evaluation method"
#elif FLT_EVAL_METHOD != 0
#error "MDSLC F32 execution requires source-type evaluation without excess precision"
#endif

#endif
