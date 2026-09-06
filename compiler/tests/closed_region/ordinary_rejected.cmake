execute_process(
  COMMAND "${CLANGXX}" -x c++ -std=c++20 -fsyntax-only "${FIXTURE}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "inspection-only; ordinary compilation is unsupported")
  message(FATAL_ERROR "Private fixture did not reject ordinary compilation clearly: ${result}\n${output}\n${error}")
endif()
message(STATUS "Private fixture rejects ordinary compilation; no eager fallback")
