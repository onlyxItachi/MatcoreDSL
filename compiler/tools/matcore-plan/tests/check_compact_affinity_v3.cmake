if(NOT DEFINED MATCORE_PLAN)
  message(FATAL_ERROR "MATCORE_PLAN executable is required")
endif()

execute_process(
  COMMAND "${MATCORE_PLAN}" --m 1024 --k 1024 --n 1024 --alignment 64
    --threads 4 --smt physical --affinity compact --numa single-node
    --variant cpu.native-parallel.avx2-fma.f32.v1
  RESULT_VARIABLE PLAN_STATUS
  OUTPUT_VARIABLE PLAN_OUTPUT
  ERROR_VARIABLE PLAN_ERROR
)
if(NOT PLAN_STATUS EQUAL 0)
  message(FATAL_ERROR
    "compact worker affinity failed on a Linux build: ${PLAN_OUTPUT}${PLAN_ERROR}")
endif()
foreach(EVIDENCE IN ITEMS
    "smt=physical"
    "smt-placement-enforced=true"
    "affinity=compact"
    "affinity-status=worker affinity was applied completely"
    "affinity-applied=true"
    "numa-memory-placement=false"
    "cpu-planner-v3 request=7"
    "selected=cpu.native-parallel.avx2-fma.f32.v1")
  string(FIND "${PLAN_OUTPUT}" "${EVIDENCE}" EVIDENCE_OFFSET)
  if(EVIDENCE_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "compact affinity diagnostic omitted '${EVIDENCE}': ${PLAN_OUTPUT}")
  endif()
endforeach()
