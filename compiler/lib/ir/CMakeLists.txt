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
)
target_compile_features(matcore_ir_v1 PUBLIC cxx_std_20)
target_include_directories(
  matcore_ir_v1
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  PRIVATE "${MATCORE_RAPIDJSON_INCLUDE_DIR}"
)
target_link_libraries(matcore_ir_v1 PUBLIC matcore_ir_v0)
target_compile_options(matcore_ir_v1 PRIVATE -Wall -Wextra -Wpedantic)
