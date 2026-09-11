cmake_minimum_required(VERSION 3.20)
# Test-only corruption of issued LLVM; never input to the source/primitive issuer.
# Prove the downstream arithmetic checker notices an actual FMA realization,
# independently of successful numerical or manifest self-consistency checks.
string(RANDOM LENGTH 14 ALPHABET 0123456789abcdef nonce)
set(ir "${CMAKE_CURRENT_BINARY_DIR}/strict-fused-negative-${nonce}.ll")
file(READ "${ORIGINAL_IR}" contents)
string(REPLACE "fmul float" "fmul contract float" contents "${contents}")
string(REPLACE "fadd float" "fadd contract float" contents "${contents}")
file(WRITE "${ir}" "${contents}")
file(SHA256 "${ir}" digest)
file(READ "${ORIGINAL_IR}.manifest" manifest)
string(REGEX REPLACE "llvm_sha256=[0-9a-f]+" "llvm_sha256=${digest}" manifest "${manifest}")
file(WRITE "${ir}.manifest" "${manifest}")
execute_process(COMMAND "${CLANG}" -c -x ir "${ir}" -O3 -ffp-contract=fast -o "${ir}.o"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status STREQUAL "0")
  message(FATAL_ERROR "FMA test control did not compile: ${output}\n${error}")
endif()
execute_process(COMMAND "${OBJDUMP}" -d "${ir}.o" RESULT_VARIABLE status OUTPUT_VARIABLE assembly ERROR_VARIABLE error)
if(NOT status STREQUAL "0" OR NOT assembly MATCHES "vfmadd[^\n]*zmm")
  message(FATAL_ERROR "FMA negative control did not contain actual fused ZMM arithmetic: ${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" "-DOBJECT=${ir}.o" "-DIR=${ir}" -DISA=avx512f
  "-DNM=${NM}" "-DOBJDUMP=${OBJDUMP}" -DSANITIZED=OFF -P "${CHECKER}"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(status STREQUAL "0" OR NOT error MATCHES "Invalid strict ISA machine/arithmetic footprint")
  message(FATAL_ERROR "Actual fused object escaped strict arithmetic guard: ${status}\n${output}\n${error}")
endif()
message(STATUS "Actual ZMM FMA negative-control object was rejected by strict arithmetic guard")
