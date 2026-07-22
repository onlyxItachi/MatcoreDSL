if(NOT DEFINED MDSLC_OBJDUMP OR NOT DEFINED MDSLC_NM OR
   NOT DEFINED MDSLC_PACKED_ARTIFACT)
  message(FATAL_ERROR
    "MDSLC_OBJDUMP, MDSLC_NM, and MDSLC_PACKED_ARTIFACT are required")
endif()

set(MDSLC_MICROKERNEL
  "matcore_cpu_packed_avx2_4x16_microkernel_f32_v1")

execute_process(
  COMMAND "${MDSLC_NM}" --defined-only "${MDSLC_PACKED_ARTIFACT}"
  RESULT_VARIABLE MDSLC_NM_STATUS
  OUTPUT_VARIABLE MDSLC_SYMBOLS
  ERROR_VARIABLE MDSLC_NM_ERROR
)
if(NOT MDSLC_NM_STATUS EQUAL 0)
  message(FATAL_ERROR
    "nm failed for ${MDSLC_PACKED_ARTIFACT}: ${MDSLC_NM_ERROR}")
endif()
string(REGEX MATCH
  "[0-9A-Fa-f]+[ \t]+[Tt][ \t]+${MDSLC_MICROKERNEL}([\r\n]|$)"
  MDSLC_MICROKERNEL_DEFINITION
  "${MDSLC_SYMBOLS}"
)
if(MDSLC_MICROKERNEL_DEFINITION STREQUAL "")
  message(FATAL_ERROR
    "exact packed AVX2 microkernel symbol is absent from ${MDSLC_PACKED_ARTIFACT}")
endif()

execute_process(
  COMMAND "${MDSLC_OBJDUMP}" -d -M intel
          "--disassemble=${MDSLC_MICROKERNEL}" "${MDSLC_PACKED_ARTIFACT}"
  RESULT_VARIABLE MDSLC_OBJDUMP_STATUS
  OUTPUT_VARIABLE MDSLC_DISASSEMBLY
  ERROR_VARIABLE MDSLC_OBJDUMP_ERROR
)
if(NOT MDSLC_OBJDUMP_STATUS EQUAL 0)
  message(FATAL_ERROR
    "objdump failed for ${MDSLC_PACKED_ARTIFACT}: ${MDSLC_OBJDUMP_ERROR}")
endif()

string(TOLOWER "${MDSLC_DISASSEMBLY}" MDSLC_DISASSEMBLY)
string(FIND "${MDSLC_DISASSEMBLY}" "ymm" MDSLC_YMM_INSTRUCTION)
string(REGEX MATCHALL "vfmadd(132|213|231)ps" MDSLC_PACKED_FMA
  "${MDSLC_DISASSEMBLY}")
list(LENGTH MDSLC_PACKED_FMA MDSLC_PACKED_FMA_COUNT)
if(MDSLC_YMM_INSTRUCTION EQUAL -1)
  message(FATAL_ERROR
    "exact packed microkernel contains no YMM operation; scalar or XMM-only code is forbidden")
endif()
if(MDSLC_PACKED_FMA_COUNT LESS 1)
  message(FATAL_ERROR
    "exact packed microkernel contains no packed single-precision FMA")
endif()

message(STATUS
  "packed AVX2 microkernel contains YMM operations and ${MDSLC_PACKED_FMA_COUNT} packed FMA instruction sites")
