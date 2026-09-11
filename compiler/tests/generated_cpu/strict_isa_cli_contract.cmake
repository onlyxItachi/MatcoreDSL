cmake_minimum_required(VERSION 3.20)
string(RANDOM LENGTH 14 ALPHABET 0123456789abcdef nonce)
set(work "${CMAKE_CURRENT_BINARY_DIR}/strict-isa-cli-${nonce}")
file(MAKE_DIRECTORY "${work}")
set(index 0)
foreach(arguments IN ITEMS "--isa=avx" "--isa=avx2" "--isa=avx512f"
    "--schedule=row-contiguous;--isa=avx512bw" "--schedule=row-contiguous;--isa=avx;--isa=avx2"
    "--schedule=row-contiguous;--isa=avx;--target=linux-aarch64"
    "--candidate=reassociate-register;--isa=baseline" "--target-features=+avx2"
    "--schedule=row-contiguous;--isa=+avx2" "--schedule=row-contiguous;--isa=avx,fma")
  math(EXPR index "${index}+1")
  set(output "${work}/bad-${index}.ll")
  execute_process(COMMAND "${ISSUER}" --output "${output}" ${arguments}
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result STREQUAL "2" OR EXISTS "${output}")
    message(FATAL_ERROR "Unclosed ISA input not rejected before issuance: ${arguments}: ${result}\n${stdout}\n${stderr}")
  endif()
endforeach()
foreach(isa IN ITEMS avx avx2 avx512f)
  set(output "${work}/${isa}.ll")
  execute_process(COMMAND "${ISSUER}" --output "${output}" --schedule=row-contiguous "--isa=${isa}"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result STREQUAL "0" OR NOT EXISTS "${output}.manifest")
    message(FATAL_ERROR "Closed ISA issuance failed: ${result}\n${stdout}\n${stderr}")
  endif()
  file(READ "${output}.manifest" manifest)
  if(NOT manifest MATCHES "\nisa=${isa}\n")
    message(FATAL_ERROR "Wrong ISA manifest")
  endif()
endforeach()
message(STATUS "Closed ISA CLI: 10 refused and 3 exact realizations")
