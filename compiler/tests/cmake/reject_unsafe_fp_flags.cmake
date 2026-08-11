foreach(required IN ITEMS MDSLC_CMAKE MDSLC_NINJA MDSLC_SOURCE_DIR
                          MDSLC_TEST_BINARY_DIR MDSLC_CXX_COMPILER)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "missing required -D${required}")
  endif()
endforeach()

file(REMOVE_RECURSE "${MDSLC_TEST_BINARY_DIR}")
execute_process(
  COMMAND "${MDSLC_CMAKE}"
          -S "${MDSLC_SOURCE_DIR}"
          -B "${MDSLC_TEST_BINARY_DIR}"
          -G Ninja
          "-DCMAKE_MAKE_PROGRAM=${MDSLC_NINJA}"
          "-DCMAKE_CXX_COMPILER=${MDSLC_CXX_COMPILER}"
          -DCMAKE_CXX_FLAGS=-ffast-math
          -DMDSLC_ENABLE_NATIVE_FRONTEND=OFF
          -DMDSLC_ENABLE_BOOTSTRAP_FRONTEND=ON
          -DMDSLC_ENABLE_OPENBLAS=OFF
          -DBUILD_TESTING=OFF
  RESULT_VARIABLE configure_status
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error
)
file(REMOVE_RECURSE "${MDSLC_TEST_BINARY_DIR}")
if(configure_status EQUAL 0)
  message(FATAL_ERROR "unsafe -ffast-math configure unexpectedly succeeded")
endif()
set(combined "${configure_output}\n${configure_error}")
if(NOT combined MATCHES "rejects unsafe global floating-point option")
  message(FATAL_ERROR
    "unsafe configure failed for the wrong reason:\n${combined}")
endif()

message(STATUS "unsafe global floating-point option rejection PASS")
