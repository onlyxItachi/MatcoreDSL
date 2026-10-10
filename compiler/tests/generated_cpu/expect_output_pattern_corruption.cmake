cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED EXECUTABLE OR NOT EXISTS "${EXECUTABLE}")
  message(FATAL_ERROR "Actual output-pattern executable is required")
endif()
execute_process(COMMAND "${EXECUTABLE}" --corrupt-output
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60)
string(STRIP "${output}" output)
if(status STREQUAL "77" AND output MATCHES "^SKIP strict ISA" AND error STREQUAL "")
  message(STATUS "${output}")
  return()
endif()
if(NOT status STREQUAL "1" OR
   NOT output MATCHES "^output pattern raw leaf tile=[1-9][0-9]*x[1-9][0-9]*: [1-9][0-9]* cases, [1-9][0-9]* checks, [1-9][0-9]* failures$" OR
   NOT error MATCHES "FAIL output pattern")
  message(FATAL_ERROR "Expected detected output corruption, not arbitrary failure: ${status}\n${output}\n${error}")
endif()
