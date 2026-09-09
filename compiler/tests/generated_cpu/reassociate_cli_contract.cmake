cmake_minimum_required(VERSION 3.20)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef nonce)
set(work "${CMAKE_CURRENT_BINARY_DIR}/reassociate-cli-${nonce}")
file(MAKE_DIRECTORY "${work}")
execute_process(COMMAND "${ISSUER}" --output "${work}/positive.ll" --candidate=reassociate-register
  RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "Positive issuer control failed: ${result}\n${error}")
endif()
file(READ "${work}/positive.ll.manifest" manifest)
if(NOT manifest MATCHES "profile=reassociate_f32" OR
   NOT manifest MATCHES "source_authority=none_builtin_primitive_only")
  message(FATAL_ERROR "Positive control has wrong profile/authority manifest")
endif()
set(case 0)
foreach(options IN ITEMS
    "--candidate=reassociate-register;--schedule=row-contiguous"
    "--candidate=reassociate-register;--candidate=reassociate-register"
    "--candidate=strict;--input=untrusted.mlir"
    "--candidate=reassociate-register;--fast-math"
    "--candidate=reassociate-register;--target=x86-64-v3"
    "--candidate=reassociate-register;--asan;--asan")
  math(EXPR case "${case} + 1")
  set(output "${work}/negative-${case}.ll")
  execute_process(COMMAND "${ISSUER}" --output "${output}" ${options}
    RESULT_VARIABLE result ERROR_VARIABLE error)
  if(NOT result STREQUAL "2" OR EXISTS "${output}" OR
     NOT error MATCHES "No source/MLIR input is accepted")
    message(FATAL_ERROR "Issuer accepted unsupported invocation ${case}: ${result}\n${error}")
  endif()
endforeach()
message(STATUS "Closed reassociate issuer: positive control and six refused invocations")
