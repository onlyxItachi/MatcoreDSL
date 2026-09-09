cmake_minimum_required(VERSION 3.20)
execute_process(COMMAND "${NM}" --defined-only --extern-only "${OBJECT}"
  RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT result STREQUAL "0" OR
   NOT symbols MATCHES " T __matcore_reassociate_gemm_f32_avx2_v1" OR
   NOT symbols MATCHES " T _mlir_ciface___matcore_reassociate_gemm_f32_avx2_v1")
  message(FATAL_ERROR "Missing issued object definitions: ${symbols}\n${error}")
endif()
string(REGEX REPLACE "[^\n]* T (_mlir_ciface_)?__matcore_reassociate_gemm_f32_avx2_v1\n?" "" other "${symbols}")
if(SANITIZED)
  # Exact common bookkeeping symbol emitted by the pinned ASan ELF pass.
  string(REGEX REPLACE "[^\n]* C ___asan_globals_registered\n?" "" other "${other}")
endif()
string(STRIP "${other}" other)
if(NOT other STREQUAL "")
  message(FATAL_ERROR "Issued object acquired another exported definition: ${other}")
endif()
execute_process(COMMAND "${NM}" --undefined-only "${OBJECT}"
  RESULT_VARIABLE result OUTPUT_VARIABLE imports ERROR_VARIABLE error)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "Cannot inspect object imports: ${error}")
endif()
# LLVM may implement the already-verified C overwrite through conforming memset.
# This is not provider/candidate selection or authorization of arbitrary calls.
string(REGEX REPLACE "[^\n]*[ \t]memset\n?" "" imports "${imports}")
if(SANITIZED)
  string(REGEX REPLACE "[^\n]*(__asan_|__start_asan_|__stop_asan_)[^\n]*\n?" "" imports "${imports}")
endif()
string(STRIP "${imports}" imports)
if(NOT imports STREQUAL "")
  message(FATAL_ERROR "Unknown allocation/provider/software-FMA import: ${imports}")
endif()
execute_process(COMMAND "${OBJDUMP}" -d "${OBJECT}"
  RESULT_VARIABLE result OUTPUT_VARIABLE assembly ERROR_VARIABLE error)
if(NOT result STREQUAL "0" OR NOT assembly MATCHES "file format elf64-x86-64" OR
   NOT assembly MATCHES "vfmadd[^\n]*ymm" OR
   NOT assembly MATCHES "v?mulss" OR NOT assembly MATCHES "v?addss" OR
   assembly MATCHES "[% ]zmm[0-9]|[% ]tmm[0-9]")
  message(FATAL_ERROR "Expected full-tile YMM FMA/scalar-tail object was not established: ${error}")
endif()
message(STATUS "Issued x86-64 AVX2/FMA object has fused full tiles, scalar arithmetic and bounded imports")
