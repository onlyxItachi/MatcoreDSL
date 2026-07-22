if(NOT DEFINED MATCORE_PLAN)
  message(FATAL_ERROR "MATCORE_PLAN executable is required")
endif()

execute_process(
  COMMAND "${MATCORE_PLAN}" --m 127 --k 131 --n 129 --alignment 64
    --threads 1 --variant auto
  RESULT_VARIABLE FIRST_STATUS
  OUTPUT_VARIABLE FIRST_OUTPUT
  ERROR_VARIABLE FIRST_ERROR
)
execute_process(
  COMMAND "${MATCORE_PLAN}" --m 127 --k 131 --n 129 --alignment 64
    --threads 1 --variant auto
  RESULT_VARIABLE SECOND_STATUS
  OUTPUT_VARIABLE SECOND_OUTPUT
  ERROR_VARIABLE SECOND_ERROR
)
if(NOT FIRST_STATUS EQUAL SECOND_STATUS OR
   NOT FIRST_OUTPUT STREQUAL SECOND_OUTPUT OR
   NOT FIRST_ERROR STREQUAL SECOND_ERROR)
  message(FATAL_ERROR
    "identical planner v2 invocations produced different diagnostics")
endif()
if(NOT FIRST_STATUS EQUAL 0 OR
   NOT FIRST_OUTPUT MATCHES "cpu-implementation-resources-v1" OR
   NOT FIRST_OUTPUT MATCHES "openblas-linked=" OR
   NOT FIRST_OUTPUT MATCHES "native-packed-workspace-bytes=" OR
   NOT FIRST_OUTPUT MATCHES "cpu-planner-v2")
  message(FATAL_ERROR
    "planner v2 deterministic diagnostic is incomplete: ${FIRST_OUTPUT}"
    "${FIRST_ERROR}")
endif()
