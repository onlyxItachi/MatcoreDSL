cmake_minimum_required(VERSION 3.24)
include("${CMAKE_CURRENT_LIST_DIR}/experimental_regions_consumer_identity.cmake")

if(DEFINED NEGATIVE_CASE)
  if(NEGATIVE_CASE STREQUAL "mislabeled_candidates")
    matcore_expect_installed_consumer_output(candidates
      "Independent private Value: 85 checks, 32000 ownership cycles, 0 failures\n")
  elseif(NEGATIVE_CASE STREQUAL "mislabeled_result")
    matcore_expect_installed_consumer_output(result "151358 candidate checks, 0 failures\n")
  elseif(NEGATIVE_CASE STREQUAL "zero_checks")
    matcore_expect_installed_consumer_output(candidates "0 candidate checks, 0 failures\n")
  elseif(NEGATIVE_CASE STREQUAL "failed_checks")
    matcore_expect_installed_consumer_output(candidates "151358 candidate checks, 1 failures\n")
  elseif(NEGATIVE_CASE STREQUAL "extra_summary")
    matcore_expect_installed_consumer_output(candidates
      "151358 candidate checks, 0 failures\nIndependent private Value: 85 checks, 32000 ownership cycles, 0 failures\n")
  elseif(NEGATIVE_CASE STREQUAL "unknown_consumer")
    matcore_expect_installed_consumer_output(unknown "151358 candidate checks, 0 failures\n")
  else()
    message(FATAL_ERROR "Unknown negative control ${NEGATIVE_CASE}")
  endif()
  message(FATAL_ERROR "Negative control did not reject through the identity checker")
endif()

# Reproduce the exact installer dispatch under the intended policy while a
# same-named artifact path exists. CMake 4 removed actual CMP0054 OLD behavior.
set(candidates "/scratch/private/libmatcore_closed_candidates_isolated_v1.so")
set(test candidates)
if(test STREQUAL "result")
  set(selected result)
elseif(test STREQUAL "candidates")
  set(selected candidates)
else()
  set(selected private_value)
endif()
if(NOT selected STREQUAL "candidates")
  message(FATAL_ERROR "NEW-policy installer dispatch selected ${selected}, not candidates")
endif()
message(STATUS "NEW-policy installer dispatch selects the candidate oracle")

# This is a documented OLD double-dereference emulation, not a CMake 3 run:
# the formerly quoted RHS is expanded to the existing candidates path.
if(test STREQUAL "${candidates}")
  message(FATAL_ERROR "OLD-policy emulation unexpectedly selected candidates")
endif()
message(STATUS "Documented OLD lookup emulation selects the private_value fallback")

matcore_expect_installed_consumer_output(result "Experimental owning Result: 37 checks, 0 failures\n")
matcore_expect_installed_consumer_output(candidates "151358 candidate checks, 0 failures\n")
matcore_expect_installed_consumer_output(candidates "121215 candidate checks, 0 failures\n")
matcore_expect_installed_consumer_output(private_value
  "Independent private Value: 85 checks, 32000 ownership cycles, 0 failures\n")
foreach(negative IN ITEMS mislabeled_candidates mislabeled_result zero_checks
    failed_checks extra_summary unknown_consumer)
  execute_process(COMMAND "${CMAKE_COMMAND}" "-DNEGATIVE_CASE=${negative}"
    -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(status EQUAL 0 OR NOT error MATCHES "output identity mismatch")
    message(FATAL_ERROR "Identity negative control ${negative} failed: ${output}\n${error}")
  endif()
endforeach()
message(STATUS "Four positive output identities and six rejection controls passed")

if(DEFINED INSTALLED_PREFIX)
  foreach(consumer IN ITEMS result candidates private_value)
    execute_process(COMMAND "${INSTALLED_PREFIX}/${consumer}"
      RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT status EQUAL 0)
      message(FATAL_ERROR "Existing installed ${consumer} failed: ${output}\n${error}")
    endif()
    matcore_expect_installed_consumer_output("${consumer}" "${output}")
    message(STATUS "Verified existing installed ${consumer}: ${output}")
  endforeach()
endif()
