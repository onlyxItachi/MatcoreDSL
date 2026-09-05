# Match the narrow RapidJSON 1.1 sanitizer boundary used by the IR parsers.
set_source_files_properties(
  ast_json_frontend.cpp
  PROPERTIES COMPILE_OPTIONS
    "${MDSLC_RAPIDJSON_NO_POINTER_OVERFLOW_OPTION};${MDSLC_RAPIDJSON_COMPAT_COMPILE_OPTIONS}"
)

add_library(matcore_frontend_common STATIC frontend_common.cpp)
target_compile_features(matcore_frontend_common PUBLIC cxx_std_20)
target_include_directories(matcore_frontend_common
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(matcore_frontend_common PUBLIC matcore_ir_v0)

if(MDSLC_ENABLE_BOOTSTRAP_FRONTEND)
  add_library(matcore_frontend_bootstrap STATIC ast_json_frontend.cpp)
  target_compile_features(matcore_frontend_bootstrap PUBLIC cxx_std_20)
  target_include_directories(matcore_frontend_bootstrap
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    PRIVATE
      "${MATCORE_RAPIDJSON_INCLUDE_DIR}"
  )
  target_link_libraries(matcore_frontend_bootstrap PUBLIC matcore_frontend_common)
endif()

if(MDSLC_ENABLE_NATIVE_FRONTEND)
  add_library(matcore_frontend_native STATIC
    native_frontend_evidence.cpp
    native_frontend.cpp
    recovered_gemm_recognizer.cpp
  )
  target_compile_features(matcore_frontend_native PUBLIC cxx_std_20)
  target_include_directories(matcore_frontend_native
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  )
  target_include_directories(matcore_frontend_native SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${CLANG_INCLUDE_DIRS}
  )
  target_compile_definitions(matcore_frontend_native PRIVATE ${LLVM_DEFINITIONS})
  mdslc_target_match_llvm_rtti(matcore_frontend_native)
  target_link_libraries(matcore_frontend_native
    PUBLIC matcore_frontend_common
    PRIVATE ${MDSLC_CLANG_TOOLING_LIBRARIES} MatcoreDSL::PlatformSupportV1
  )
  if(BUILD_TESTING AND NOT WIN32)
    execute_process(COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
      OUTPUT_VARIABLE MDSLC_REGION_FRONTEND_TEST_RESOURCE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    add_executable(matcore_two_gemm_region_frontend_tests
      "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/native/two_gemm_region_frontend_test.cpp")
    target_link_libraries(matcore_two_gemm_region_frontend_tests PRIVATE
      matcore_frontend_native)
    mdslc_target_enable_warnings(matcore_two_gemm_region_frontend_tests)
    mdslc_target_match_llvm_rtti(matcore_two_gemm_region_frontend_tests)
    add_test(NAME frontend.native.two_gemm_region
      COMMAND matcore_two_gemm_region_frontend_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/../../.."
        "${MDSLC_CLANGXX_EXECUTABLE}"
        "${MDSLC_REGION_FRONTEND_TEST_RESOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include/matcore/mdsl.h")
  endif()
endif()
