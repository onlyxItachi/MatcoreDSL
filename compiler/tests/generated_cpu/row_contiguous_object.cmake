execute_process(COMMAND "${OBJDUMP}" -d "${OBJECT}"
  RESULT_VARIABLE status OUTPUT_VARIABLE assembly ERROR_VARIABLE error)
if(NOT status EQUAL 0 OR
   NOT assembly MATCHES "file format elf64-x86-64" OR
   NOT assembly MATCHES "[ \t]mulps" OR NOT assembly MATCHES "[ \t]addps" OR
   assembly MATCHES "[ \t]v?f(madd|msub|nmadd|nmsub)")
  message(FATAL_ERROR "Row-contiguous baseline object lacks separate SSE output-lane arithmetic: ${error}\n${assembly}")
endif()
file(READ "${MANIFEST}" manifest)
if(NOT manifest MATCHES "\nschedule=row-contiguous-mkn\n")
  message(FATAL_ERROR "SIMD object test did not select the reviewed M/K/N schedule")
endif()
message(STATUS "Observed baseline x86-64 SSE mulps/addps after Transform interchange and LLVM auto-vectorization; no timing claim")
