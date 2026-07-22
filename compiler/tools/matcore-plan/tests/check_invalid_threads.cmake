if(NOT DEFINED MATCORE_PLAN)
  message(FATAL_ERROR "MATCORE_PLAN executable is required")
endif()

execute_process(
  COMMAND "${MATCORE_PLAN}" --m 4 --k 4 --n 4 --threads 0
  RESULT_VARIABLE PLAN_STATUS
  OUTPUT_VARIABLE PLAN_OUTPUT
  ERROR_VARIABLE PLAN_ERROR
)
if(PLAN_STATUS EQUAL 0)
  message(FATAL_ERROR "zero threads unexpectedly produced a plan")
endif()
string(CONCAT PLAN_DIAGNOSTIC "${PLAN_OUTPUT}" "${PLAN_ERROR}")
if(NOT PLAN_DIAGNOSTIC MATCHES "threads must be a positive uint32")
  message(FATAL_ERROR
    "zero threads lacked an actionable diagnostic: ${PLAN_DIAGNOSTIC}")
endif()
