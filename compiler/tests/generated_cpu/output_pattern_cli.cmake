cmake_minimum_required(VERSION 3.20)
string(RANDOM LENGTH 14 ALPHABET 0123456789abcdef nonce)
set(work "${CMAKE_CURRENT_BINARY_DIR}/output-pattern-cli-${nonce}")
file(MAKE_DIRECTORY "${work}")
set(index 0)
foreach(arguments IN ITEMS
    "--output-tiles=0,4" "--output-tiles=4,0" "--output-tiles=-1,4"
    "--output-tiles=65,1" "--output-tiles=1,65" "--output-tiles=4"
    "--output-tiles=4,8,2" "--output-tiles=4,+8" "--output-tiles=4,8x"
    "--output-tiles=18446744073709551616,4" "--output-tiles=,4"
    "--output-tiles=4," "--output-tiles=3,5;--schedule=row-contiguous"
    "--output-tiles=3,5;--output-tiles=4,8"
    "--output-tiles=3,5;--candidate=reassociate-register"
    "--output-tiles=3,5;--target=linux-aarch64;--isa=avx2"
    "--output-tiles=3,5;--tile-k=4")
  math(EXPR index "${index}+1")
  set(output "${work}/bad-${index}.ll")
  execute_process(COMMAND "${ISSUER}" --output "${output}" ${arguments}
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result STREQUAL "2" OR EXISTS "${output}")
    message(FATAL_ERROR "Unclosed pattern accepted: ${arguments}: ${result}\n${stdout}\n${stderr}")
  endif()
endforeach()
foreach(tiles IN ITEMS "3,5" "4,16" "1,1" "64,64")
  set(output "${work}/good-${tiles}.ll")
  execute_process(COMMAND "${ISSUER}" --output "${output}" "--output-tiles=${tiles}"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result STREQUAL "0" OR NOT EXISTS "${output}.transform.mlir")
    message(FATAL_ERROR "Closed output pattern failed: ${result}\n${stdout}\n${stderr}")
  endif()
  file(READ "${output}.manifest" manifest)
  if(NOT manifest MATCHES "\nschedule=output-tiled-mkn\n")
    message(FATAL_ERROR "Output pattern is not identified by artifact manifest")
  endif()
endforeach()
message(STATUS "Closed output pattern CLI: 17 refusals and 4 issued parameter sets")
