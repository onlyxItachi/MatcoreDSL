cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED DRIVER OR NOT IS_ABSOLUTE "${DRIVER}" OR NOT EXISTS "${DRIVER}")
  message(FATAL_ERROR "An actual built mdslc-region DRIVER is required")
endif()
if(CASE STREQUAL "repeated_include")
  set(expected "PASS library_repeated_include 67 checks")
elseif(CASE STREQUAL "escaped_address" OR CASE STREQUAL "escaped_wrapper")
  set(rejection "retired Value helper has a remaining nonhelper use:")
else()
  message(FATAL_ERROR "CASE must be repeated_include, escaped_address or escaped_wrapper")
endif()
set(source "${CMAKE_CURRENT_LIST_DIR}/${CASE}.mdsl")
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef identity)
set(work "${CMAKE_CURRENT_BINARY_DIR}/library-adversary/${CASE}-${identity}")
file(MAKE_DIRECTORY "${work}")
set(executable "${work}/generated")
execute_process(
  COMMAND "${DRIVER}" "${source}" --region library_region
    --candidate generated-strict -o "${executable}"
  WORKING_DIRECTORY "${work}"
  RESULT_VARIABLE compile_status OUTPUT_VARIABLE compile_output
  ERROR_VARIABLE compile_error TIMEOUT 120)
if(DEFINED rejection)
  # These cases must reach ownership retirement, not fail earlier admission.
  if(NOT "${compile_status}" MATCHES "^[1-9][0-9]*$" OR
     EXISTS "${executable}" OR NOT "${compile_error}" MATCHES "${rejection}")
    message(FATAL_ERROR "Expected exact helper-retirement rejection, got ${compile_status}:\n${compile_output}\n${compile_error}")
  endif()
  message(STATUS "PASS library_${CASE} exact retirement rejection")
  return()
endif()
if(NOT "${compile_status}" STREQUAL "0" OR NOT EXISTS "${executable}")
  message(FATAL_ERROR "Library source compilation failed (${compile_status}):\n${compile_output}\n${compile_error}")
endif()
execute_process(COMMAND "${executable}" WORKING_DIRECTORY "${work}"
  RESULT_VARIABLE run_status OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error TIMEOUT 60)
string(STRIP "${run_output}" run_output)
if(NOT "${run_status}" STREQUAL "0" OR
   NOT "${run_output}" STREQUAL "${expected}" OR NOT "${run_error}" STREQUAL "")
  message(FATAL_ERROR "Library source oracle failed (${run_status}): expected '${expected}', got:\n${run_output}\n${run_error}\nExecutable retained: ${executable}")
endif()
file(REMOVE "${executable}")
message(STATUS "${run_output}")
