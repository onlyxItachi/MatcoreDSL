if(NOT INPUT MATCHES "^(a|b|d)$")
  message(FATAL_ERROR "Unknown generated-load negative control")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "DEBUGINFOD_URLS="
  "ASAN_OPTIONS=halt_on_error=1:detect_leaks=0" "${EXECUTABLE}" "read-${INPUT}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "AddressSanitizer: heap-buffer-overflow" OR
   NOT error MATCHES "READ of size 4 at " OR
   NOT error MATCHES "#0[^\n]*(_mlir_ciface_)?__matcore_strict_fused_gemm_f32_v1")
  message(FATAL_ERROR "Emitted pair load was not instrumented: ${result}\n${output}\n${error}")
endif()
message(STATUS "Actual emitted pair ${INPUT} float load reported ASan heap OOB")
