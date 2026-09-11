cmake_minimum_required(VERSION 3.20)
if(NOT DEFINED POLICY)
  set(POLICY generated-strict)
endif()
if(DEFINED PROBE)
  execute_process(COMMAND "${PROBE}" RESULT_VARIABLE probe_status OUTPUT_VARIABLE probe_output ERROR_VARIABLE probe_error)
  if(probe_status STREQUAL "77" AND probe_output MATCHES "^SKIP strict ISA" AND probe_error STREQUAL "")
    message(STATUS "${probe_output}")
    return()
  elseif(NOT probe_status STREQUAL "0")
    message(FATAL_ERROR "ISA execution probe failed: ${probe_output}\n${probe_error}")
  endif()
endif()
if(NOT DEFINED DRIVER OR NOT IS_ABSOLUTE "${DRIVER}" OR NOT EXISTS "${DRIVER}")
  message(FATAL_ERROR "An actual built mdslc-region DRIVER is required")
endif()
if(CASE STREQUAL "source")
  set(region contract_region)
  set(expected "PASS hpc_source_contract 4140 checks")
elseif(CASE STREQUAL "same_value")
  set(region same_value_region)
  set(expected "PASS hpc_same_value_contract 1503 checks")
else()
  message(FATAL_ERROR "CASE must be exactly source or same_value")
endif()
set(source "${CMAKE_CURRENT_LIST_DIR}/schedule_${CASE}_contract.mdsl")
set(work "${CMAKE_CURRENT_BINARY_DIR}/schedule-source-${CASE}")
file(MAKE_DIRECTORY "${work}")
# The production driver deliberately never overwrites an existing artifact.
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef identity)
set(executable "${work}/generated-${identity}")
execute_process(
  COMMAND "${DRIVER}" "${source}" --region "${region}"
    --candidate "${POLICY}" -o "${executable}"
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
  message(FATAL_ERROR "Source oracle failed (${run_status}): expected '${expected}', got:\n${run_output}\n${run_error}\nExecutable retained: ${executable}")
endif()
# Remove only the exact fresh executable created by this successful test.
file(REMOVE "${executable}")
message(STATUS "${run_output}")
