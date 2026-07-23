add_executable(matcore-extract main.cpp)
target_compile_features(matcore-extract PRIVATE cxx_std_20)
target_include_directories(
  matcore-extract PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../generated")
target_link_libraries(matcore-extract PRIVATE
  matcore_codegen_v0
  matcore_ir_v1
  MatcoreDSL::PlatformSupportV1
)
if(MDSLC_ENABLE_NATIVE_FRONTEND)
  target_link_libraries(matcore-extract PRIVATE matcore_frontend_native)
  target_compile_definitions(matcore-extract PRIVATE MDSLC_HAS_NATIVE_FRONTEND=1)
else()
  target_compile_definitions(matcore-extract PRIVATE MDSLC_HAS_NATIVE_FRONTEND=0)
endif()
if(MDSLC_ENABLE_BOOTSTRAP_FRONTEND)
  target_link_libraries(matcore-extract PRIVATE matcore_frontend_bootstrap)
  target_compile_definitions(matcore-extract PRIVATE MDSLC_HAS_BOOTSTRAP_FRONTEND=1)
else()
  target_compile_definitions(matcore-extract PRIVATE MDSLC_HAS_BOOTSTRAP_FRONTEND=0)
endif()
