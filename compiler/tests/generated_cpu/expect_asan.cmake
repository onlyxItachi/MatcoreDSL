cmake_minimum_required(VERSION 3.20)
set(argument --oob)
if(DEFINED MODE)
  if(NOT MODE STREQUAL "simd")
    message(FATAL_ERROR "Unknown generated ASan negative-control MODE")
  endif()
  set(argument --oob-simd)
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "DEBUGINFOD_URLS="
  "ASAN_OPTIONS=halt_on_error=1:detect_leaks=0" "${EXECUTABLE}" "${argument}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "AddressSanitizer: heap-buffer-overflow" OR
   NOT error MATCHES "_mlir_ciface___matcore_strict_gemm_f32_v1|__matcore_strict_gemm_f32_v1")
  message(FATAL_ERROR "Generated-code ASan negative control did not detect kernel OOB: ${result}\n${output}\n${error}")
endif()
if(DEFINED MODE AND
   (NOT error MATCHES "READ of size (16|32|64) at " OR
    NOT error MATCHES "#0[^\n]*(_mlir_ciface___matcore_strict_gemm_f32_v1|__matcore_strict_gemm_f32_v1)"))
  message(FATAL_ERROR "Generated SIMD OOB did not demonstrate a wide instrumented read with a generated top frame:\n${error}")
endif()
message(STATUS "Generated-kernel ${argument} heap OOB was detected by AddressSanitizer")
