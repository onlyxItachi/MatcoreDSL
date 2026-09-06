execute_process(COMMAND "${PROGRAM}" success
  RESULT_VARIABLE success OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT success EQUAL 0)
  message(FATAL_ERROR "The archive ordinary-success control failed: ${success} ${error}")
endif()
execute_process(COMMAND "${PROGRAM}"
  RESULT_VARIABLE failed OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT REQUIRED_REPRODUCTION AND failed EQUAL 0)
  message(STATUS "Instrumented archive completed the failure sweep without exposing the Release weak cleanup seam; this is not an intrinsic ownership proof")
elseif(NOT failed EQUAL EXPECTED)
  message(FATAL_ERROR "Weak archive cleanup counterexample did not reproduce exit ${EXPECTED}: ${failed} ${error}")
else()
  message(STATUS "Ordinary math passed, but archive weak cleanup selected host exit ${EXPECTED}")
endif()
