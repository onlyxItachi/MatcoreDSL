cmake_minimum_required(VERSION 3.20)
execute_process(COMMAND "${EXECUTABLE}" --oob-full RESULT_VARIABLE status
  OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60)
if(status STREQUAL "0" OR NOT error MATCHES "AddressSanitizer: heap-buffer-overflow" OR
   NOT error MATCHES "__matcore_strict_gemm_f32_v1")
  message(FATAL_ERROR "Expected issued full-tile leaf ASan failure, not an arbitrary failure: ${status}\n${output}\n${error}")
endif()
