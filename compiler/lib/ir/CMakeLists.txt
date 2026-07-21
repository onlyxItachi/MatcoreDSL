find_path(MATCORE_RAPIDJSON_INCLUDE_DIR rapidjson/prettywriter.h REQUIRED)

add_library(matcore_ir_v0 STATIC matcore_ir.cpp)
target_compile_features(matcore_ir_v0 PUBLIC cxx_std_20)
target_include_directories(matcore_ir_v0
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  PRIVATE
    "${MATCORE_RAPIDJSON_INCLUDE_DIR}"
)

add_library(
  matcore_ir_v1
  STATIC
    matcore_ir_v1.cpp
    matcore_ir_v1_bridge.cpp
    matcore_ir_v1_json.cpp
)
target_compile_features(matcore_ir_v1 PUBLIC cxx_std_20)
target_include_directories(
  matcore_ir_v1
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  PRIVATE "${MATCORE_RAPIDJSON_INCLUDE_DIR}"
)
target_link_libraries(matcore_ir_v1 PUBLIC matcore_ir_v0)
target_compile_options(matcore_ir_v1 PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  add_executable(
    matcore_ir_v1_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir/matcore_ir_v1_test.cpp"
  )
  target_link_libraries(matcore_ir_v1_tests PRIVATE matcore_ir_v1)
  target_compile_options(
    matcore_ir_v1_tests
    PRIVATE -Wall -Wextra -Wpedantic
  )
  target_compile_definitions(
    matcore_ir_v1_tests
    PRIVATE
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
      MDSLC_FRONTEND_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/frontend"
  )
  add_test(NAME ir.v1.core COMMAND matcore_ir_v1_tests)
endif()
