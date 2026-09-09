cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED DRIVER OR NOT IS_ABSOLUTE "${DRIVER}" OR NOT EXISTS "${DRIVER}")
  message(FATAL_ERROR "An actual built mdslc-region DRIVER is required")
endif()
set(source "${CMAKE_CURRENT_LIST_DIR}/cache_schedule_source_contract.mdsl")
set(expected "PASS cache_schedule_source_contract 225 cases 563296 checks")
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef identity)
# Separate concurrent invocations: the compiler authenticates its working
# directory, so another invocation must not create/remove sibling artifacts.
set(work "${CMAKE_CURRENT_BINARY_DIR}/cache-schedule-source/${identity}")
file(MAKE_DIRECTORY "${work}")
set(executable "${work}/generated")
execute_process(
  COMMAND "${DRIVER}" "${source}" --region cache_contract_region
    --candidate generated-strict -o "${executable}"
  WORKING_DIRECTORY "${work}"
  RESULT_VARIABLE compile_status OUTPUT_VARIABLE compile_output
  ERROR_VARIABLE compile_error TIMEOUT 120)
if(NOT "${compile_status}" STREQUAL "0" OR NOT EXISTS "${executable}")
  message(FATAL_ERROR "Source compilation failed (${compile_status}):\n${compile_output}\n${compile_error}")
endif()
execute_process(COMMAND "${executable}" WORKING_DIRECTORY "${work}"
  RESULT_VARIABLE run_status OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error TIMEOUT 60)
string(STRIP "${run_output}" run_output)
if(NOT "${run_status}" STREQUAL "0" OR
   NOT "${run_output}" STREQUAL "${expected}" OR NOT "${run_error}" STREQUAL "")
  message(FATAL_ERROR "Cache source oracle failed (${run_status}): expected '${expected}', got:\n${run_output}\n${run_error}\nExecutable retained: ${executable}")
endif()
# Remove only the exact fresh executable created by this successful test.
file(REMOVE "${executable}")
message(STATUS "${run_output}")
