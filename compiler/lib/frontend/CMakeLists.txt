find_path(MATCORE_RAPIDJSON_INCLUDE_DIR rapidjson/document.h REQUIRED)

# Match the narrow RapidJSON 1.1 sanitizer boundary used by the IR parsers.
set_source_files_properties(
  ast_json_frontend.cpp
  PROPERTIES COMPILE_OPTIONS -fno-sanitize=pointer-overflow
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
  add_library(matcore_frontend_native STATIC native_frontend.cpp)
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
  target_link_libraries(matcore_frontend_native
    PUBLIC matcore_frontend_common
    PRIVATE clang-cpp LLVM
  )
endif()
