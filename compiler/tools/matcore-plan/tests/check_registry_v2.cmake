if(NOT DEFINED MATCORE_PLAN)
  message(FATAL_ERROR "MATCORE_PLAN executable is required")
endif()

set(STABLE_VARIANTS
  cpu.reference.f32.v1
  cpu.tiled.f32.v1
  cpu.compiler-vectorized.avx2-fma.f32.v1
  cpu.external.openblas.f32.v1
  cpu.native-packed.avx2-fma.f32.v1
)

foreach(VARIANT IN LISTS STABLE_VARIANTS)
  execute_process(
    COMMAND "${MATCORE_PLAN}" --m 33 --k 35 --n 37 --alignment 64
      --threads 1 --variant "${VARIANT}"
    RESULT_VARIABLE PLAN_STATUS
    OUTPUT_VARIABLE PLAN_OUTPUT
    ERROR_VARIABLE PLAN_ERROR
  )
  string(CONCAT PLAN_DIAGNOSTIC "${PLAN_OUTPUT}" "${PLAN_ERROR}")
  string(FIND "${PLAN_DIAGNOSTIC}" "${VARIANT}" VARIANT_OFFSET)
  if(PLAN_DIAGNOSTIC MATCHES "unsupported variant")
    message(FATAL_ERROR
      "registered variant ${VARIANT} was rejected by the CLI parser: "
      "${PLAN_DIAGNOSTIC}")
  endif()
  if(NOT PLAN_DIAGNOSTIC MATCHES "cpu-planner-v2" OR
     VARIANT_OFFSET EQUAL -1 OR
     NOT PLAN_DIAGNOSTIC MATCHES "candidates=\\[")
    message(FATAL_ERROR
      "registered variant ${VARIANT} lacked a complete v2 diagnostic: "
      "${PLAN_DIAGNOSTIC}")
  endif()
  if(NOT PLAN_STATUS EQUAL 0 AND NOT PLAN_STATUS EQUAL 1)
    message(FATAL_ERROR
      "registered variant ${VARIANT} returned unexpected status "
      "${PLAN_STATUS}: ${PLAN_DIAGNOSTIC}")
  endif()
endforeach()
