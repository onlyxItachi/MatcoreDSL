add_executable(matcore-mlir main.cpp)
target_link_libraries(matcore-mlir PRIVATE matcore_mlir_semantics)
mdslc_target_enable_warnings(matcore-mlir WERROR)

if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter QUIET)
endif()
if(BUILD_TESTING AND Python3_Interpreter_FOUND)
  add_test(
    NAME mlir.semantic.cli
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/run_matcore_mlir_tests.py"
      --tool "$<TARGET_FILE:matcore-mlir>"
  )
  set_tests_properties(
    mlir.semantic.cli
    PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  )
endif()
