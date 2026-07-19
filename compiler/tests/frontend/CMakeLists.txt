cmake_minimum_required(VERSION 3.24)

project(MatcoreFrontendBootstrapTests LANGUAGES CXX)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../lib/ir"
                 "${CMAKE_CURRENT_BINARY_DIR}/ir")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../lib/frontend"
                 "${CMAKE_CURRENT_BINARY_DIR}/frontend")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../lib/codegen"
                 "${CMAKE_CURRENT_BINARY_DIR}/codegen")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../tools/matcore-extract"
                 "${CMAKE_CURRENT_BINARY_DIR}/matcore-extract")

enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(
  NAME mdslc_frontend_bootstrap
  COMMAND
    "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_LIST_DIR}/run_frontend_tests.py"
    --extractor "$<TARGET_FILE:matcore-extract>"
)
