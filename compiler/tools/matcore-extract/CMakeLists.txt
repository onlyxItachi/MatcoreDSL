add_executable(matcore-extract main.cpp)
target_compile_features(matcore-extract PRIVATE cxx_std_20)
target_include_directories(
  matcore-extract PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../generated")
target_link_libraries(matcore-extract
  PRIVATE matcore_frontend_bootstrap matcore_codegen_v0)
