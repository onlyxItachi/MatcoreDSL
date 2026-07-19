find_path(MATCORE_RAPIDJSON_INCLUDE_DIR rapidjson/document.h REQUIRED)

add_library(matcore_frontend_bootstrap STATIC ast_json_frontend.cpp)
target_compile_features(matcore_frontend_bootstrap PUBLIC cxx_std_20)
target_include_directories(matcore_frontend_bootstrap
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  PRIVATE
    "${MATCORE_RAPIDJSON_INCLUDE_DIR}"
)
target_link_libraries(matcore_frontend_bootstrap PUBLIC matcore_ir_v0)
