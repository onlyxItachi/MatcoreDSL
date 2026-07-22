if(NOT DEFINED MATCORE_PLAN)
  message(FATAL_ERROR "MATCORE_PLAN executable is required")
endif()

set(STABLE_VARIANTS
  cpu.reference.f32.v1
  cpu.tiled.f32.v1
  cpu.compiler-vectorized.avx2-fma.f32.v1
  cpu.external.openblas.f32.v1
  cpu.native-packed.avx2-fma.f32.v1
  cpu.native-packed.avx512-fma.f32.v1
  cpu.native-parallel.avx2-fma.f32.v1
  cpu.native-parallel.avx512-fma.f32.v1
)

set(EXPECTED_REQUEST 0)

foreach(VARIANT IN LISTS STABLE_VARIANTS)
  math(EXPR EXPECTED_REQUEST "${EXPECTED_REQUEST} + 1")
  execute_process(
    COMMAND "${MATCORE_PLAN}" --m 257 --k 259 --n 255 --alignment 64
      --threads 4 --variant "${VARIANT}"
    RESULT_VARIABLE PLAN_STATUS
    OUTPUT_VARIABLE PLAN_OUTPUT
    ERROR_VARIABLE PLAN_ERROR
  )
  string(CONCAT PLAN_DIAGNOSTIC "${PLAN_OUTPUT}" "${PLAN_ERROR}")
  if(PLAN_DIAGNOSTIC MATCHES "unsupported variant")
    message(FATAL_ERROR
      "registered variant ${VARIANT} was rejected by the CLI parser: "
      "${PLAN_DIAGNOSTIC}")
  endif()
  if(NOT PLAN_DIAGNOSTIC MATCHES
       "cpu-planner-v3 request-id=${EXPECTED_REQUEST} .*candidates=\\[")
    message(FATAL_ERROR
      "registered variant ${VARIANT} mapped to the wrong v3 request or "
      "lacked a complete diagnostic: "
      "${PLAN_DIAGNOSTIC}")
  endif()
  if(PLAN_STATUS EQUAL 0)
    string(FIND "${PLAN_DIAGNOSTIC}"
      "selected=${VARIANT} reason=explicit legal variant request"
      SELECTED_OFFSET)
    if(SELECTED_OFFSET EQUAL -1)
      message(FATAL_ERROR
        "forced legal variant ${VARIANT} did not select its exact stable ID: "
        "${PLAN_DIAGNOSTIC}")
    endif()
  elseif(PLAN_STATUS EQUAL 1)
    string(FIND "${PLAN_DIAGNOSTIC}" "${VARIANT}:rejected:reason="
      REJECTED_OFFSET)
    if(REJECTED_OFFSET EQUAL -1)
      message(FATAL_ERROR
        "forced unavailable variant ${VARIANT} lacked its actionable exact "
        "candidate rejection: ${PLAN_DIAGNOSTIC}")
    endif()
  else()
    message(FATAL_ERROR
      "registered variant ${VARIANT} returned unexpected status "
      "${PLAN_STATUS}: ${PLAN_DIAGNOSTIC}")
  endif()
  foreach(FIELD IN ITEMS required-hardware required-os required-compiler
      required-implementation runtime-validated shared-workspace
      per-worker-workspace)
    string(FIND "${PLAN_DIAGNOSTIC}" "${FIELD}=" FIELD_OFFSET)
    if(FIELD_OFFSET EQUAL -1)
      message(FATAL_ERROR
        "variant ${VARIANT} diagnostic omitted ${FIELD}: ${PLAN_DIAGNOSTIC}")
    endif()
  endforeach()
endforeach()
