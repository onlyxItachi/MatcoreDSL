include_guard(GLOBAL)

# Keep compiler-family policy in one place.  clang-cl reports Clang as the
# compiler ID while using the MSVC command-line and ABI frontend, so key this
# distinction from MSVC/CMAKE_*_COMPILER_FRONTEND_VARIANT rather than from the
# compiler ID alone.
set(MDSLC_USING_MSVC_ABI_FRONTEND OFF)
if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  set(MDSLC_USING_MSVC_ABI_FRONTEND ON)
endif()

set(MDSLC_USING_CLANG_CL OFF)
if(MDSLC_USING_MSVC_ABI_FRONTEND AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(MDSLC_USING_CLANG_CL ON)
endif()

function(mdslc_reject_unsafe_global_fp_flags)
  get_cmake_property(mdslc_all_variables VARIABLES)
  set(mdslc_fp_flag_variables "")
  foreach(mdslc_variable IN LISTS mdslc_all_variables)
    if(mdslc_variable MATCHES "^CMAKE_(C|CXX)_FLAGS($|_)")
      list(APPEND mdslc_fp_flag_variables "${mdslc_variable}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES mdslc_fp_flag_variables)

  set(mdslc_unsafe_fp_tokens
    -Ofast
    -ffast-math
    -funsafe-math-optimizations
    -ffinite-math-only
    -fno-honor-infinities
    -fno-honor-nans
    -fapprox-func
    -freciprocal-math
    -fassociative-math
    -fno-signed-zeros
    -mdaz-ftz
    -ffp-model=fast
    -ffp-model=aggressive
    /fp:fast
  )
  foreach(mdslc_variable IN LISTS mdslc_fp_flag_variables)
    set(mdslc_value "${${mdslc_variable}}")
    foreach(mdslc_token IN LISTS mdslc_unsafe_fp_tokens)
      string(FIND "${mdslc_value}" "${mdslc_token}" mdslc_token_offset)
      if(NOT mdslc_token_offset EQUAL -1)
        message(FATAL_ERROR
          "MDSLC rejects unsafe global floating-point option '${mdslc_token}' "
          "from ${mdslc_variable}. The explicit-gemm-f32-v1 contract requires "
          "finite/NaN/Inf preservation, round-to-nearest-even, masked traps, "
          "and gradual subnormals.")
      endif()
    endforeach()

    string(REGEX MATCHALL
      "-fdenormal-fp-math(-f32)?=[^ \\t;]+"
      mdslc_denormal_options "${mdslc_value}")
    foreach(mdslc_denormal IN LISTS mdslc_denormal_options)
      if(NOT mdslc_denormal MATCHES "=ieee(,ieee)?$")
        message(FATAL_ERROR
          "MDSLC rejects non-IEEE global denormal option "
          "'${mdslc_denormal}' from ${mdslc_variable}.")
      endif()
    endforeach()

    string(REGEX MATCHALL
      "-ffp-eval-method=[^ \\t;]+"
      mdslc_evaluation_options "${mdslc_value}")
    foreach(mdslc_evaluation IN LISTS mdslc_evaluation_options)
      if(NOT mdslc_evaluation STREQUAL "-ffp-eval-method=source")
        message(FATAL_ERROR
          "MDSLC rejects unsafe global floating-point option "
          "'${mdslc_evaluation}' from ${mdslc_variable}. The "
          "explicit-gemm-f32-v1 contract requires source-type evaluation "
          "and F32 accumulation without excess intermediate precision.")
      endif()
    endforeach()
  endforeach()
endfunction()

function(mdslc_target_enable_precise_fp target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR
      "mdslc_target_enable_precise_fp: unknown target '${target_name}'")
  endif()
  set(mdslc_fp_contract_header
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../lib/platform/fp_compile_contract_v1.h")
  target_compile_definitions(${target_name} PRIVATE
    MATCORE_MDSLC_PRECISE_FP_PROFILE=1)
  if(MDSLC_USING_MSVC_ABI_FRONTEND)
    target_compile_options(${target_name} PRIVATE
      /fp:precise
      "/FI${mdslc_fp_contract_header}")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${target_name} PRIVATE
      -fno-fast-math
      -ffp-contract=fast
      -ffp-eval-method=source
      -fhonor-infinities
      -fhonor-nans
      -fno-finite-math-only
      -fno-approx-func
      -fno-reciprocal-math
      -fdenormal-fp-math=ieee
      -include "${mdslc_fp_contract_header}")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(${target_name} PRIVATE
      -fno-fast-math
      -ffp-contract=fast
      -fno-finite-math-only
      -fno-reciprocal-math
      -include "${mdslc_fp_contract_header}")
  else()
    message(FATAL_ERROR
      "MDSLC has no authenticated precise floating-point profile for "
      "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
  endif()
endfunction()

mdslc_reject_unsafe_global_fp_flags()

function(mdslc_target_enable_warnings target_name)
  cmake_parse_arguments(PARSE_ARGV 1 MDSLC_WARNINGS "WERROR" "" "")
  if(MDSLC_WARNINGS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "mdslc_target_enable_warnings(${target_name}): unrecognized arguments: "
      "${MDSLC_WARNINGS_UNPARSED_ARGUMENTS}")
  endif()

  if(MDSLC_USING_MSVC_ABI_FRONTEND)
    target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    if(MDSLC_WARNINGS_WERROR)
      target_compile_options(${target_name} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    if(MDSLC_WARNINGS_WERROR)
      target_compile_options(${target_name} PRIVATE -Werror)
    endif()
  endif()
endfunction()

function(mdslc_target_enable_debug_optimization target_name)
  if(MDSLC_USING_MSVC_ABI_FRONTEND)
    set(mdslc_debug_optimization /O2)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(mdslc_debug_optimization -O2)
  else()
    return()
  endif()

  if(CMAKE_CONFIGURATION_TYPES)
    target_compile_options(
      ${target_name} PRIVATE "$<$<CONFIG:Debug>:${mdslc_debug_optimization}>")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "")
    target_compile_options(${target_name} PRIVATE ${mdslc_debug_optimization})
  endif()
endfunction()

# These options are intentionally attached only to the RapidJSON translation
# units.  The MSVC frontend form keeps clang-cl in CL mode; non-Clang MSVC
# builds omit Clang-specific sanitizer and compatibility switches entirely.
set(MDSLC_RAPIDJSON_NO_POINTER_OVERFLOW_OPTION "")
if(MDSLC_USING_CLANG_CL)
  set(MDSLC_RAPIDJSON_NO_POINTER_OVERFLOW_OPTION
    "/clang:-fno-sanitize=pointer-overflow")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(MDSLC_RAPIDJSON_NO_POINTER_OVERFLOW_OPTION
    "-fno-sanitize=pointer-overflow")
endif()
