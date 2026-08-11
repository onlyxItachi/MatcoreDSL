foreach(required IN ITEMS MDSLC_NINJA MDSLC_BINARY_DIR MDSLC_MSVC_ABI
                          MDSLC_CXX_COMPILER_ID)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "missing required -D${required}")
  endif()
endforeach()

execute_process(
  COMMAND "${MDSLC_NINJA}" -C "${MDSLC_BINARY_DIR}" -t commands
          matcore_cpu_backends_v1 matcore_runtime
  RESULT_VARIABLE command_status
  OUTPUT_VARIABLE commands
  ERROR_VARIABLE command_error
)
if(NOT command_status EQUAL 0)
  message(FATAL_ERROR "could not inspect runtime compile commands: ${command_error}")
endif()

string(REPLACE "\n" ";" command_lines "${commands}")
set(critical_compile_lines 0)
foreach(line IN LISTS command_lines)
  if((line MATCHES "matcore_cpu_backends_v1\\.dir.*\\.(c|cc|cpp|cxx)" OR
      line MATCHES "matcore_runtime\\.dir.*\\.(c|cc|cpp|cxx)") AND
     NOT line MATCHES "(^| )(:|cmake -E) ")
    math(EXPR critical_compile_lines "${critical_compile_lines} + 1")
    if(MDSLC_MSVC_ABI)
      foreach(required_option IN ITEMS "/fp:precise" "/FI" "fp_compile_contract_v1.h")
        string(FIND "${line}" "${required_option}" option_offset)
        if(option_offset EQUAL -1)
          message(FATAL_ERROR
            "runtime compile command lacks ${required_option}: ${line}")
        endif()
      endforeach()
    else()
      foreach(required_option IN ITEMS
          "-fno-fast-math"
          "-ffp-contract=fast"
          "fp_compile_contract_v1.h")
        string(FIND "${line}" "${required_option}" option_offset)
        if(option_offset EQUAL -1)
          message(FATAL_ERROR
            "runtime compile command lacks ${required_option}: ${line}")
        endif()
      endforeach()
      if(MDSLC_CXX_COMPILER_ID MATCHES "Clang")
        string(FIND "${line}" "-ffp-eval-method=source" option_offset)
        if(option_offset EQUAL -1)
          message(FATAL_ERROR
            "runtime compile command lacks -ffp-eval-method=source: ${line}")
        endif()
      endif()
    endif()
  endif()
endforeach()

if(critical_compile_lines LESS 2)
  message(FATAL_ERROR
    "compile-command inspection found too few critical runtime commands")
endif()

message(STATUS
  "authenticated precise FP options on ${critical_compile_lines} runtime compile commands")
