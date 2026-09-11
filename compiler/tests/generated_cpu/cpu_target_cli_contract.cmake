cmake_minimum_required(VERSION 3.20)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef nonce)
set(work "${CMAKE_CURRENT_BINARY_DIR}/cpu-target-cli-${nonce}")
file(MAKE_DIRECTORY "${work}")
foreach(target IN ITEMS default linux-x86_64 linux-aarch64)
  set(options)
  if(NOT target STREQUAL "default")
    list(APPEND options "--target=${target}")
  endif()
  execute_process(COMMAND "${ISSUER}" --output "${work}/${target}.ll" ${options}
    RESULT_VARIABLE result ERROR_VARIABLE error)
  if(NOT result STREQUAL "0")
    message(FATAL_ERROR "Fixed target ${target} failed: ${result}\n${error}")
  endif()
endforeach()
file(SHA256 "${work}/default.ll" default_sha)
file(SHA256 "${work}/linux-x86_64.ll" x86_sha)
file(READ "${work}/default.ll.manifest" default_manifest)
file(READ "${work}/linux-x86_64.ll.manifest" x86_manifest)
file(READ "${work}/linux-aarch64.ll.manifest" arm_manifest)
if(NOT default_sha STREQUAL x86_sha OR NOT default_manifest STREQUAL x86_manifest)
  message(FATAL_ERROR "Explicit baseline x86 target changed the default artifact")
endif()
if(NOT arm_manifest MATCHES "target=aarch64-unknown-linux-gnu\n" OR
   NOT arm_manifest MATCHES "profile=strict_f32\n" OR
   NOT arm_manifest MATCHES "source_authority=none_builtin_primitive_only\n")
  message(FATAL_ERROR "ARM target did not retain strict builtin-only authority")
endif()
execute_process(COMMAND "${ISSUER}" --output "${work}/arm-row-asan.ll"
    --target=linux-aarch64 --asan --schedule=row-contiguous
  RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "ARM row-contiguous ASan option composition failed: ${error}")
endif()
set(case 0)
foreach(options IN ITEMS
    "--target=linux-aarch64;--candidate=reassociate-register"
    "--target=linux-aarch64;--target=linux-x86_64"
    "--target=linux-x86_64;--target=linux-x86_64"
    "--target=aarch64-unknown-linux-gnu"
    "--target=linux-aarch64;--input=untrusted.mlir"
    "--target=linux-aarch64;--target-features=+sve"
    "--target=linux-aarch64;--fast-math"
    "--target=linux-aarch64;--asan;--asan")
  math(EXPR case "${case} + 1")
  set(output "${work}/negative-${case}.ll")
  execute_process(COMMAND "${ISSUER}" --output "${output}" ${options}
    RESULT_VARIABLE result ERROR_VARIABLE error)
  file(GLOB published "${output}*")
  if(NOT result STREQUAL "2" OR published OR
     NOT error MATCHES "No source/MLIR input is accepted")
    message(FATAL_ERROR "Unsupported target invocation ${case} was not refused cleanly: ${result}\n${error}")
  endif()
endforeach()
message(STATUS "Closed target CLI: four positive controls, unchanged x86 default, eight refused invocations")
