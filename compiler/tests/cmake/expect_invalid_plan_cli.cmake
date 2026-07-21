if(NOT DEFINED MATCORE_PLAN)
  message(FATAL_ERROR "MATCORE_PLAN executable is required")
endif()

execute_process(
  COMMAND "${MATCORE_PLAN}" --m 4 --k 4 --n 4 --alignment 6
  RESULT_VARIABLE PLAN_STATUS
  OUTPUT_VARIABLE PLAN_OUTPUT
  ERROR_VARIABLE PLAN_ERROR
)
if(PLAN_STATUS EQUAL 0)
  message(FATAL_ERROR "invalid alignment unexpectedly produced a plan")
endif()
string(CONCAT PLAN_DIAGNOSTIC "${PLAN_OUTPUT}" "${PLAN_ERROR}")
if(NOT PLAN_DIAGNOSTIC MATCHES
   "alignment must be a power of two and at least 4 bytes")
  message(FATAL_ERROR
    "invalid alignment lacked its actionable diagnostic: ${PLAN_DIAGNOSTIC}")
endif()
