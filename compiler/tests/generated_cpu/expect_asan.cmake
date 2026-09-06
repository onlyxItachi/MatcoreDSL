execute_process(COMMAND "${CMAKE_COMMAND}" -E env "DEBUGINFOD_URLS="
  "ASAN_OPTIONS=halt_on_error=1:detect_leaks=0" "${EXECUTABLE}" --oob
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "AddressSanitizer: heap-buffer-overflow" OR
   NOT error MATCHES "_mlir_ciface___matcore_strict_gemm_f32_v1|__matcore_strict_gemm_f32_v1")
  message(FATAL_ERROR "Generated-code ASan negative control did not detect kernel OOB: ${result}\n${output}\n${error}")
endif()
message(STATUS "Generated-kernel heap OOB was detected by AddressSanitizer")
