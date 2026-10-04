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
file(WRITE "${root}/main.cpp"
  "extern int math_main(int,char**);\nint main(int argc,char **argv) { return math_main(argc,argv); }\n")
# Preserve the original fixture as a negative: Clang attaches nounwind strictfp
# to its direct protected call under FENV_ACCESS ON, unlike the canonical
# program-interface witness's nounwind-only call. Never relax that comparator.
file(WRITE "${root}/strict-direct.mdsl" "${source}")
foreach(optimization IN ITEMS none publication-read-forwarding)
  execute_process(COMMAND "${DRIVER}" --program --host "${root}/main.cpp"
    --region "${root}/strict-direct.mdsl" forwarding_pipeline --candidate generated-strict
    --optimization "${optimization}" -o "${root}/strict-direct-${optimization}"
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  if(status EQUAL 0 OR EXISTS "${root}/strict-direct-${optimization}" OR NOT stderr MATCHES
      "CALLSITE: region call ABI or effect promises differ from compiler witness")
    message(FATAL_ERROR "Original strictfp callsite did not retain its program-interface refusal: ${stdout}\n${stderr}")
  endif()
endforeach()
# Only the ordinary host oracle calls through this pragma-free noexcept wrapper.
# The region body and every region source line/column remain unchanged; run()
# retains FENV_ACCESS ON and its complete caller-FP before/after oracle.
set(wrapper [=[
static md::Result forwarding_host_call(
    md::Storage A, md::Storage B, md::Storage C, md::Storage D, md::Storage E,
    md::Shape m, md::Shape n, md::Shape k, md::Shape requested_rows,
    md::Shape requested_columns, md::Shape clobber) noexcept {
  return forwarding_pipeline(A,B,C,D,E,m,n,k,requested_rows,requested_columns,clobber);
}
]=])
string(REPLACE "#pragma STDC FENV_ACCESS ON" "${wrapper}\n#pragma STDC FENV_ACCESS ON" source "${source}")
string(REPLACE "auto result = forwarding_pipeline(" "auto result = forwarding_host_call(" source "${source}")
file(WRITE "${root}/source.mdsl" "${source}")
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
