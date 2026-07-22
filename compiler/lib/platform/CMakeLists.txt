add_library(matcore_platform_v1 STATIC platform.cpp)
add_library(MatcoreDSL::PlatformV1 ALIAS matcore_platform_v1)

target_compile_features(matcore_platform_v1 PUBLIC cxx_std_20)
target_include_directories(
  matcore_platform_v1
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

if(BUILD_TESTING)
  add_executable(
    matcore_platform_v1_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/platform/platform_test.cpp"
  )
  target_compile_features(matcore_platform_v1_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_platform_v1_test PRIVATE MatcoreDSL::PlatformV1)
  add_test(NAME platform.record.v1 COMMAND matcore_platform_v1_test)
endif()
