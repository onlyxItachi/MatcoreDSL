cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED EXECUTABLE OR NOT EXISTS "${EXECUTABLE}")
  message(FATAL_ERROR "Actual input alias executable is required")
endif()
execute_process(COMMAND "${EXECUTABLE}" --corrupt-output
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
string(STRIP "${output}" output)
if(status STREQUAL "77" AND output MATCHES "^SKIP strict ISA" AND error STREQUAL "")
  message(STATUS "${output}")
  return()
endif()
if(NOT "${status}" STREQUAL "1" OR
   NOT "${output}" STREQUAL "input alias execution: 72 cases, 86898 checks, 54 failures" OR
   NOT error MATCHES "FAIL input-alias strict result")
  message(FATAL_ERROR "Expected deliberate result corruption, not arbitrary failure: ${status}\n${output}\n${error}")
endif()
