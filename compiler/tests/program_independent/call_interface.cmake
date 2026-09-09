cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED WORK)
  string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef nonce)
  set(WORK "${CMAKE_CURRENT_BINARY_DIR}/program-call-interface-${nonce}")
endif()
file(MAKE_DIRECTORY "${WORK}")
execute_process(COMMAND "${CLANG}" -x c++ -std=c++20 -O0 -S -emit-llvm
  -Xclang -disable-O0-optnone "-I${COMPILER_SOURCE}/include"
  "-I${COMPILER_SOURCE}/tests/program/fixtures" "${CMAKE_CURRENT_LIST_DIR}/main_weakref_pure.cpp"
  -o "${WORK}/actual-clang.ll" RESULT_VARIABLE compiled ERROR_VARIABLE error)
if(NOT compiled STREQUAL "0")
  message(FATAL_ERROR "Real Clang weakref fixture emission failed: ${compiled}\n${error}")
endif()
execute_process(COMMAND "${TEST}" "${WORK}/actual-clang.ll"
  RESULT_VARIABLE tested OUTPUT_VARIABLE output ERROR_VARIABLE error)
string(STRIP "${output}" output)
if(NOT tested STREQUAL "0" OR NOT error STREQUAL "" OR
   NOT output STREQUAL "Call-interface checks 36, failures 0")
  message(FATAL_ERROR "Callsite contract test failed: ${tested}\n${output}\n${error}")
endif()
message(STATUS "Real-Clang callsite ABI/effects: ${output}")
