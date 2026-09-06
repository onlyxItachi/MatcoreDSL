execute_process(COMMAND "${NM}" --defined-only "${OBJECT}"
  RESULT_VARIABLE status OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT status EQUAL 0 OR NOT symbols MATCHES " T __matcore_strict_gemm_f32_v1" OR
   NOT symbols MATCHES " T _mlir_ciface___matcore_strict_gemm_f32_v1")
  message(FATAL_ERROR "Missing real generated object entry points: ${symbols}\n${error}")
endif()
execute_process(COMMAND "${NM}" --undefined-only "${OBJECT}"
  RESULT_VARIABLE status OUTPUT_VARIABLE imports ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Cannot inspect generated object imports: ${error}")
endif()
string(REGEX REPLACE "[^\n]*[ \t]memset\n?" "" imports "${imports}")
if(SANITIZED MATCHES "fsanitize=.*address")
  # Runtime instrumentation imports are expected only in the instrumented lane.
  string(REGEX REPLACE "[^\n]*(__asan_|__start_asan_|__stop_asan_)[^\n]*\n?" "" imports "${imports}")
endif()
string(STRIP "${imports}" imports)
if(NOT imports STREQUAL "")
  message(FATAL_ERROR "Generated object acquired unknown allocation/provider/runtime imports: ${imports}")
endif()
execute_process(COMMAND "${OBJDUMP}" -d "${OBJECT}"
  RESULT_VARIABLE status OUTPUT_VARIABLE assembly ERROR_VARIABLE error)
if(NOT status EQUAL 0 OR NOT assembly MATCHES "file format elf64-x86-64" OR
   assembly MATCHES "[ \t]v?f(madd|msub|nmadd|nmsub)" OR
   NOT assembly MATCHES "[ \t]mulss" OR NOT assembly MATCHES "[ \t]addss")
  message(FATAL_ERROR "Generated baseline object lost separate scalar arithmetic/target: ${error}\n${assembly}")
endif()
message(STATUS "Generated object is baseline x86-64 with separate scalar multiply/add and bounded imports")
