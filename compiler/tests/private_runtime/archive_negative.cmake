execute_process(COMMAND "${PROGRAM}" success
  RESULT_VARIABLE success OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT success EQUAL 0)
  message(FATAL_ERROR "The archive ordinary-success control failed: ${success} ${error}")
endif()
execute_process(COMMAND "${PROGRAM}"
  RESULT_VARIABLE failed OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT failed EQUAL EXPECTED)
  message(FATAL_ERROR "Weak archive cleanup counterexample did not reproduce exit ${EXPECTED}: ${failed} ${error}")
endif()
message(STATUS "Ordinary math passed, but archive weak cleanup selected host exit ${EXPECTED}")
