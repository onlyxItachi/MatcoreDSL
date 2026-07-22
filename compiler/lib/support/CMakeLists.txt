cmake_minimum_required(VERSION 3.24)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(MatcoreDSLPlatformSupport VERSION 1.0.0 LANGUAGES CXX)
  include(CTest)
endif()

find_package(Threads REQUIRED)

set(MATCORE_PLATFORM_SUPPORT_SOURCES
  platform_support_common.cpp
)
if(WIN32)
  list(APPEND MATCORE_PLATFORM_SUPPORT_SOURCES platform_support_windows.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND MATCORE_PLATFORM_SUPPORT_SOURCES platform_support_linux.cpp)
else()
  message(FATAL_ERROR
    "Matcore platform support v1 has no backend for ${CMAKE_SYSTEM_NAME}")
endif()

add_library(matcore_platform_support_v1 STATIC
  ${MATCORE_PLATFORM_SUPPORT_SOURCES}
)
add_library(MatcoreDSL::PlatformSupportV1 ALIAS matcore_platform_support_v1)
target_compile_features(matcore_platform_support_v1 PUBLIC cxx_std_20)
set_target_properties(matcore_platform_support_v1 PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(
  matcore_platform_support_v1
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(matcore_platform_support_v1 PUBLIC Threads::Threads)

if(MSVC)
  target_compile_options(matcore_platform_support_v1 PRIVATE /W4 /WX /permissive-)
else()
  target_compile_options(matcore_platform_support_v1
    PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(BUILD_TESTING)
  add_executable(matcore_platform_support_v1_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/support/platform_support_test.cpp"
  )
  target_compile_features(matcore_platform_support_v1_test PRIVATE cxx_std_20)
  set_target_properties(matcore_platform_support_v1_test
    PROPERTIES CXX_EXTENSIONS OFF)
  target_link_libraries(matcore_platform_support_v1_test
    PRIVATE MatcoreDSL::PlatformSupportV1)
  if(MSVC)
    target_compile_options(matcore_platform_support_v1_test
      PRIVATE /W4 /WX /permissive-)
  else()
    target_compile_options(matcore_platform_support_v1_test
      PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME support.platform.v1 COMMAND matcore_platform_support_v1_test)
endif()
