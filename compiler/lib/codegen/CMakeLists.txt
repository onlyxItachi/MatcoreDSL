add_library(matcore_codegen_v0 STATIC codegen.cpp)
target_compile_features(matcore_codegen_v0 PUBLIC cxx_std_20)
target_include_directories(matcore_codegen_v0
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(matcore_codegen_v0 PUBLIC matcore_ir_v0)
