cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED DRIVER)
  message(FATAL_ERROR "Missing publication-read forwarding program driver")
endif()
if(NOT NM)
  find_program(NM NAMES llvm-nm-21 nm REQUIRED)
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(root "${CMAKE_CURRENT_BINARY_DIR}/publication-forwarding-program-${suffix}")
file(MAKE_DIRECTORY "${root}")
file(READ "${CMAKE_CURRENT_LIST_DIR}/publication_read_forwarding.mdsl" source)
string(REPLACE "int main(" "int math_main(" source "${source}")
file(WRITE "${root}/source.mdsl" "${source}")
file(WRITE "${root}/main.cpp"
  "extern int math_main(int,char**);\nint main(int argc,char **argv) { return math_main(argc,argv); }\n")
foreach(optimization IN ITEMS none publication-read-forwarding)
  execute_process(COMMAND "${DRIVER}" --program --host "${root}/main.cpp"
    --region "${root}/source.mdsl" forwarding_pipeline --candidate generated-strict
    --optimization "${optimization}" -o "${root}/${optimization}"
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT status EQUAL 0 OR NOT EXISTS "${root}/${optimization}" OR
      NOT stdout MATCHES "PUBLISHED authenticated multi-source program")
    message(FATAL_ERROR "Multi-source ${optimization} compilation failed: ${stdout}\n${stderr}")
  endif()
  execute_process(COMMAND "${NM}" --undefined-only --demangle "${root}/${optimization}"
    RESULT_VARIABLE symbol_status OUTPUT_VARIABLE symbols ERROR_VARIABLE symbol_error)
  if(NOT symbol_status EQUAL 0)
    message(FATAL_ERROR "Cannot inspect multi-source orchestration: ${symbol_error}")
  endif()
  if(optimization STREQUAL "publication-read-forwarding")
    if(NOT symbols MATCHES "closed_host_v1::SessionAbiV2::readForwarded")
      message(FATAL_ERROR "Optimized program driver did not connect the checked readForwarded call")
    endif()
  elseif(symbols MATCHES "closed_host_v1::SessionAbiV2::readForwarded")
    message(FATAL_ERROR "No-optimization program baseline unexpectedly calls readForwarded")
  endif()
  execute_process(COMMAND "${root}/${optimization}" --require-execution
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(NOT status EQUAL 0 OR NOT stdout MATCHES
      "^Publication forwarding source: [1-9][0-9]* checks; 0 failures; 6 executed cases; 0 explicit refusals\n$")
    message(FATAL_ERROR "Multi-source ${optimization} oracle failed: ${stdout}\n${stderr}")
  endif()
  if(optimization STREQUAL "none")
    set(baseline "${stdout}")
  else()
    set(optimized "${stdout}")
  endif()
  message(STATUS "program/${optimization}: ${stdout}")
endforeach()
if(NOT baseline STREQUAL optimized)
  message(FATAL_ERROR "Multi-source same-source realization outcomes differ")
endif()
execute_process(COMMAND "${DRIVER}" --program --host "${root}/main.cpp"
  --region "${root}/source.mdsl" forwarding_pipeline --optimization unknown -o "${root}/invalid"
  RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if(status EQUAL 0 OR EXISTS "${root}/invalid" OR NOT stderr MATCHES "unknown region optimization")
  message(FATAL_ERROR "Multi-source unknown optimization acquired authority: ${stdout}\n${stderr}")
endif()
execute_process(COMMAND "${DRIVER}" --program --host "${root}/main.cpp"
  --region "${root}/source.mdsl" forwarding_pipeline
  --optimization none --optimization publication-read-forwarding -o "${root}/duplicate"
  RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if(status EQUAL 0 OR EXISTS "${root}/duplicate" OR NOT stderr MATCHES "duplicate --optimization")
  message(FATAL_ERROR "Multi-source duplicate optimization was not rejected: ${stdout}\n${stderr}")
endif()
