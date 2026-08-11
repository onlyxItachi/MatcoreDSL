foreach(required IN ITEMS MDSLC_CMAKE MDSLC_NINJA MDSLC_SOURCE_DIR
                          MDSLC_TEST_BINARY_DIR MDSLC_CXX_COMPILER)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "missing required -D${required}")
  endif()
endforeach()

set(unsafe_flags
  -ffast-math
  -ffp-eval-method=double
  -ffp-eval-method=extended
)
set(case_index 0)
foreach(unsafe_flag IN LISTS unsafe_flags)
  math(EXPR case_index "${case_index} + 1")
  set(case_binary_dir "${MDSLC_TEST_BINARY_DIR}-${case_index}")
  file(REMOVE_RECURSE "${case_binary_dir}")
  execute_process(
    COMMAND "${MDSLC_CMAKE}"
            -S "${MDSLC_SOURCE_DIR}"
            -B "${case_binary_dir}"
            -G Ninja
            "-DCMAKE_MAKE_PROGRAM=${MDSLC_NINJA}"
            "-DCMAKE_CXX_COMPILER=${MDSLC_CXX_COMPILER}"
            "-DCMAKE_CXX_FLAGS=${unsafe_flag}"
            -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF
            -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON
            -DMDSLC_ENABLE_OPENBLAS=OFF
            -DBUILD_TESTING=OFF
    RESULT_VARIABLE configure_status
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error
  )
  file(REMOVE_RECURSE "${case_binary_dir}")
  if(configure_status EQUAL 0)
    message(FATAL_ERROR
      "unsafe ${unsafe_flag} configure unexpectedly succeeded")
  endif()
  set(combined "${configure_output}\n${configure_error}")
  if(NOT combined MATCHES "rejects unsafe global floating-point option")
    message(FATAL_ERROR
      "unsafe ${unsafe_flag} configure failed for the wrong reason:\n${combined}")
  endif()
endforeach()

set(safe_binary_dir "${MDSLC_TEST_BINARY_DIR}-source")
file(REMOVE_RECURSE "${safe_binary_dir}")
execute_process(
  COMMAND "${MDSLC_CMAKE}"
          -S "${MDSLC_SOURCE_DIR}"
          -B "${safe_binary_dir}"
          -G Ninja
          "-DCMAKE_MAKE_PROGRAM=${MDSLC_NINJA}"
          "-DCMAKE_CXX_COMPILER=${MDSLC_CXX_COMPILER}"
          -DCMAKE_CXX_FLAGS=-ffp-eval-method=source
          -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF
          -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON
          -DMDSLC_ENABLE_OPENBLAS=OFF
          -DBUILD_TESTING=OFF
  RESULT_VARIABLE safe_status
  OUTPUT_VARIABLE safe_output
  ERROR_VARIABLE safe_error
)
file(REMOVE_RECURSE "${safe_binary_dir}")
if(NOT safe_status EQUAL 0)
  message(FATAL_ERROR
    "safe -ffp-eval-method=source configure failed:\n${safe_output}\n${safe_error}")
endif()

message(STATUS "unsafe global FP options reject and source evaluation accepts")
