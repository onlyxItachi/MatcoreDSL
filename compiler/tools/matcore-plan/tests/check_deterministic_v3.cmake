if(NOT DEFINED MATCORE_PLAN)
  message(FATAL_ERROR "MATCORE_PLAN executable is required")
endif()

set(COMMAND_ARGS --m 1024 --k 1024 --n 1024 --alignment 64 --threads 4
  --affinity none --numa single-node --variant auto)
execute_process(
  COMMAND "${MATCORE_PLAN}" ${COMMAND_ARGS}
  RESULT_VARIABLE FIRST_STATUS
  OUTPUT_VARIABLE FIRST_OUTPUT
  ERROR_VARIABLE FIRST_ERROR
)
execute_process(
  COMMAND "${MATCORE_PLAN}" ${COMMAND_ARGS}
  RESULT_VARIABLE SECOND_STATUS
  OUTPUT_VARIABLE SECOND_OUTPUT
  ERROR_VARIABLE SECOND_ERROR
)
if(NOT FIRST_STATUS EQUAL SECOND_STATUS OR
   NOT FIRST_OUTPUT STREQUAL SECOND_OUTPUT OR
   NOT FIRST_ERROR STREQUAL SECOND_ERROR)
  message(FATAL_ERROR
    "identical planner v3 invocations produced different diagnostics")
endif()
if(NOT FIRST_STATUS EQUAL 0 OR
   NOT FIRST_OUTPUT MATCHES "cpu-capabilities-v2" OR
   NOT FIRST_OUTPUT MATCHES "cpu-topology-v1" OR
   NOT FIRST_OUTPUT MATCHES "cpu-execution-policy-v1" OR
   NOT FIRST_OUTPUT MATCHES "shared-workspace=" OR
   NOT FIRST_OUTPUT MATCHES "per-worker-workspace=" OR
   NOT FIRST_OUTPUT MATCHES "cpu-planner-v3")
  message(FATAL_ERROR
    "planner v3 deterministic diagnostic is incomplete: ${FIRST_OUTPUT}"
    "${FIRST_ERROR}")
endif()
