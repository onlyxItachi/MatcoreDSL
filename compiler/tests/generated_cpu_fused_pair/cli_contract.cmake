foreach(option IN ITEMS --input=forged.mlir --transform=forged.mlir --target=linux-aarch64
    --isa=avx2 --candidate=reassociate --asan-extra)
  execute_process(COMMAND "${ISSUER}" --output "${CMAKE_CURRENT_BINARY_DIR}/refused-pair.ll" "${option}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT result EQUAL 2 OR NOT error MATCHES "No source/MLIR/LLVM/Transform input")
    message(FATAL_ERROR "Closed pair issuer did not refuse ${option}: ${result}\n${output}\n${error}")
  endif()
endforeach()
execute_process(COMMAND "${ISSUER}" RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 2)
  message(FATAL_ERROR "Missing-arguments issuer did not fail cleanly: ${result}\n${output}\n${error}")
endif()
