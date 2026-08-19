add_library(matcore_codegen_v0 STATIC codegen.cpp)
target_compile_features(matcore_codegen_v0 PUBLIC cxx_std_20)
target_include_directories(matcore_codegen_v0
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(matcore_codegen_v0 PUBLIC matcore_ir_v0)

if(BUILD_TESTING)
  add_executable(matcore_runtime_dispatch_backend_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/runtime_dispatch_backend_test.cpp"
  )
  target_compile_features(matcore_runtime_dispatch_backend_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_runtime_dispatch_backend_test
    PRIVATE matcore_codegen_v0
  )
  mdslc_target_enable_warnings(matcore_runtime_dispatch_backend_test WERROR)
  add_test(NAME codegen.runtime_dispatch_backend_v1
    COMMAND matcore_runtime_dispatch_backend_test
  )

  add_executable(matcore_static_aot_specialization_execution_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/static_aot_specialization_execution_test.cpp"
  )
  target_compile_features(matcore_static_aot_specialization_execution_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_static_aot_specialization_execution_test
    PRIVATE matcore_codegen_v0 matcore_runtime_c_abi_v0
  )
  mdslc_target_enable_warnings(matcore_static_aot_specialization_execution_test WERROR)
  add_test(NAME codegen.static_aot_specialization_execution
    COMMAND matcore_static_aot_specialization_execution_test
  )
endif()
