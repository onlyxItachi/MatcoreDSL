cmake_minimum_required(VERSION 3.20)
string(RANDOM LENGTH 14 ALPHABET 0123456789abcdef nonce)
set(work "${CMAKE_CURRENT_BINARY_DIR}/output-peel-cli-${nonce}")
file(MAKE_DIRECTORY "${work}")
set(index 0)
foreach(arguments IN ITEMS
    "--output-tiles-peel-n=0,3" "--output-tiles-peel-n=5,0"
    "--output-tiles-peel-n=-1,3" "--output-tiles-peel-n=65,1"
    "--output-tiles-peel-n=1,65" "--output-tiles-peel-n=5"
    "--output-tiles-peel-n=5,3,1" "--output-tiles-peel-n=5,+3"
    "--output-tiles-peel-n=5,3x" "--output-tiles-peel-n=18446744073709551616,3"
    "--output-tiles-peel-n=,3" "--output-tiles-peel-n=5,"
    "--output-tiles-peel-n=5,3;--output-tiles=5,3"
    "--output-tiles=5,3;--output-tiles-peel-n=5,3"
    "--output-tiles-peel-n=5,3;--schedule=row-contiguous"
    "--output-tiles-peel-n=5,3;--output-tiles-peel-n=3,5"
    "--output-tiles-peel-n=5,3;--candidate=reassociate-register"
    "--output-tiles-peel-n=5,3;--isa=avx"
    "--output-tiles-peel-n=5,3;--isa=avx2"
    "--output-tiles-peel-n=5,3;--isa=avx512f"
    "--output-tiles-peel-n=5,3;--tile-k=4"
    "--output-tiles-peel-n=5,3;--input=untrusted.mlir")
  math(EXPR index "${index}+1")
  set(output "${work}/bad-${index}.ll")
  execute_process(COMMAND "${ISSUER}" --output "${output}" ${arguments}
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT result STREQUAL "2" OR EXISTS "${output}")
    message(FATAL_ERROR "Unclosed peeled pattern accepted: ${arguments}: ${result}\n${stdout}\n${stderr}")
  endif()
endforeach()
foreach(target IN ITEMS linux-x86_64 linux-aarch64)
  foreach(tiles IN ITEMS "5,3" "3,5" "1,1" "64,64")
    set(output "${work}/good-${target}-${tiles}.ll")
    execute_process(COMMAND "${ISSUER}" --output "${output}"
      "--output-tiles-peel-n=${tiles}" "--target=${target}"
      RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(NOT result STREQUAL "0" OR NOT EXISTS "${output}.transform.mlir")
      message(FATAL_ERROR "Closed peeled pattern failed: ${result}\n${stdout}\n${stderr}")
    endif()
    file(READ "${output}.manifest" manifest)
    if(NOT manifest MATCHES "\nschedule=output-tiled-peel-n-mkn\n" OR
       NOT manifest MATCHES "\npattern=matcore-gemm-output-tile-peel-n-v1\n" OR
       NOT manifest MATCHES "\npattern_source_selection=unavailable_private_issuer_only\n")
      message(FATAL_ERROR "Peeled private pattern identity/authority is missing")
    endif()
  endforeach()
endforeach()
message(STATUS "Closed peeled pattern CLI: ${index} refusals and 8 issued parameter/target sets")
