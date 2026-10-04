set(arguments)
set(environment)
if(MODE STREQUAL "negative")
  list(APPEND arguments --negative-workspace-under-memcheck)
  set(environment "MDSLC_RESEARCH_CUDA_MEMCHECK_NEGATIVE_V1=1")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env ${environment} "${SANITIZER}"
  --tool memcheck --error-exitcode 86 --padding 32 "${EXECUTABLE}" "${IMAGE}" ${arguments}
  RESULT_VARIABLE status OUTPUT_VARIABLE out ERROR_VARIABLE error TIMEOUT 150)
set(log "${out}\n${error}")
if(MODE STREQUAL "negative")
  if(NOT status EQUAL 86 OR NOT log MATCHES "Invalid __global__ (write|read)" OR
     NOT log MATCHES "__matcore_strict_fused_gemm_f32_v1_kernel" OR
     NOT log MATCHES "no output exposed, no possibly-live resource reclaimed")
    message(FATAL_ERROR "Device instrumentation negative control failed: ${status}: ${log}")
  endif()
else()
  if(NOT status EQUAL 0 OR NOT log MATCHES "ERROR SUMMARY: 0 errors" OR NOT log MATCHES "PASS launched=")
    message(FATAL_ERROR "Strict pair memcheck failed: ${status}: ${log}")
  endif()
endif()
message(STATUS "GPU pair memcheck ${MODE}: ${status}: ${log}")
