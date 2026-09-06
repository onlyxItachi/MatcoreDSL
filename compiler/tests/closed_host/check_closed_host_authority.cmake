# Independent negative controls. A successful ordinary link is required before
# treating a failed authority-forgery link as evidence.
foreach(required CXX NM SOURCE_DIR INCLUDE_DIR PRODUCTION_LIBRARY OUTPUT_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing closed-host authority test input: ${required}")
  endif()
endforeach()
if(NOT EXISTS "${PRODUCTION_LIBRARY}")
  message(FATAL_ERROR "Production closed-host library is missing")
endif()
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
separate_arguments(compile_flags NATIVE_COMMAND "${CXX_FLAGS}")
separate_arguments(link_flags NATIVE_COMMAND "${LINK_FLAGS}")

execute_process(
  COMMAND "${CXX}" ${compile_flags} -std=c++20 -UMDSLC_CLOSED_HOST_TESTING
    -I "${INCLUDE_DIR}" "${SOURCE_DIR}/production_link_control.cpp"
    "${PRODUCTION_LIBRARY}" ${link_flags} -o "${OUTPUT_DIR}/link-control"
  RESULT_VARIABLE control_result OUTPUT_VARIABLE control_stdout
  ERROR_VARIABLE control_stderr)
if(NOT control_result EQUAL 0)
  message(FATAL_ERROR "Positive production link control failed:\n${control_stdout}\n${control_stderr}")
endif()

execute_process(
  COMMAND "${CXX}" ${compile_flags} -std=c++20 -UMDSLC_CLOSED_HOST_TESTING
    -I "${INCLUDE_DIR}" -fsyntax-only "${SOURCE_DIR}/forged_value.cpp"
  RESULT_VARIABLE value_result OUTPUT_VARIABLE value_stdout
  ERROR_VARIABLE value_stderr)
if(value_result EQUAL 0 OR NOT value_stderr MATCHES "storage_" OR
   NOT value_stderr MATCHES "private")
  message(FATAL_ERROR "Forged value did not fail at its private-storage boundary:\n${value_stdout}\n${value_stderr}")
endif()

execute_process(
  COMMAND "${CXX}" ${compile_flags} -std=c++20 -UMDSLC_CLOSED_HOST_TESTING
    -I "${INCLUDE_DIR}" -fsyntax-only
    "${SOURCE_DIR}/production_candidate_injection.cpp"
  RESULT_VARIABLE setter_result OUTPUT_VARIABLE setter_stdout
  ERROR_VARIABLE setter_stderr)
if(setter_result EQUAL 0 OR NOT setter_stderr MATCHES "configureForTesting" OR
   NOT setter_stderr MATCHES "no member")
  message(FATAL_ERROR "Production candidate injection did not reject at declaration boundary:\n${setter_stdout}\n${setter_stderr}")
endif()

execute_process(
  COMMAND "${CXX}" ${compile_flags} -std=c++20
    -DMDSLC_CLOSED_HOST_TESTING=1 -I "${INCLUDE_DIR}"
    "${SOURCE_DIR}/production_candidate_injection.cpp"
    "${PRODUCTION_LIBRARY}" ${link_flags} -o "${OUTPUT_DIR}/forged-test-authority"
  RESULT_VARIABLE link_result OUTPUT_VARIABLE link_stdout
  ERROR_VARIABLE link_stderr)
if(link_result EQUAL 0 OR NOT link_stderr MATCHES "configureForTesting" OR
   NOT link_stderr MATCHES "undefined reference|undefined symbol|unresolved external")
  message(FATAL_ERROR "Macro-forged candidate injection did not reject at production link boundary:\n${link_stdout}\n${link_stderr}")
endif()
execute_process(
  COMMAND "${NM}" -C --defined-only "${PRODUCTION_LIBRARY}"
  RESULT_VARIABLE symbols_result OUTPUT_VARIABLE symbols_stdout
  ERROR_VARIABLE symbols_stderr)
if(NOT symbols_result EQUAL 0)
  message(FATAL_ERROR "Production symbol inspection failed:\n${symbols_stderr}")
endif()
if(symbols_stdout MATCHES "configureForTesting|allocationAttemptsForTesting")
  message(FATAL_ERROR "Production library exports test-control authority:\n${symbols_stdout}")
endif()
message(STATUS "Closed-host authority controls: positive link, three expected rejections and no test-control symbols passed")
