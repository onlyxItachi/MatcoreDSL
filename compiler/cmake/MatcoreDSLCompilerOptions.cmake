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
