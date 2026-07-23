add_library(matcore_platform_v1 STATIC
  platform.cpp
  cpu_capability_v2.cpp
  cpu_topology_v1.cpp
  windows_cpu_topology_v1.cpp
  thread_affinity_v1.cpp
)
add_library(MatcoreDSL::PlatformV1 ALIAS matcore_platform_v1)

target_compile_features(matcore_platform_v1 PUBLIC cxx_std_20)
target_include_directories(
  matcore_platform_v1
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

mdslc_target_enable_warnings(matcore_platform_v1 WERROR)

if(BUILD_TESTING)
  add_executable(
    matcore_platform_v1_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/platform/platform_test.cpp"
  )
  target_compile_features(matcore_platform_v1_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_platform_v1_test PRIVATE MatcoreDSL::PlatformV1)
  add_test(NAME platform.record.v1 COMMAND matcore_platform_v1_test)

  add_executable(
    matcore_cpu_capability_v2_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/platform/cpu_capability_v2_test.cpp"
  )
  target_compile_features(matcore_cpu_capability_v2_test PRIVATE cxx_std_20)
  mdslc_target_enable_warnings(matcore_cpu_capability_v2_test WERROR)
  target_link_libraries(matcore_cpu_capability_v2_test
    PRIVATE MatcoreDSL::PlatformV1
  )
  add_test(NAME platform.cpu_capability.v2
    COMMAND matcore_cpu_capability_v2_test
  )

  add_executable(
    matcore_cpu_topology_v1_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/platform/cpu_topology_v1_test.cpp"
  )
  target_compile_features(matcore_cpu_topology_v1_test PRIVATE cxx_std_20)
  mdslc_target_enable_warnings(matcore_cpu_topology_v1_test WERROR)
  target_link_libraries(matcore_cpu_topology_v1_test
    PRIVATE MatcoreDSL::PlatformV1
  )
  add_test(NAME platform.cpu_topology.v1
    COMMAND matcore_cpu_topology_v1_test
  )

  add_executable(
    matcore_thread_affinity_v1_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/platform/thread_affinity_v1_test.cpp"
  )
  target_compile_features(matcore_thread_affinity_v1_test PRIVATE cxx_std_20)
  mdslc_target_enable_warnings(matcore_thread_affinity_v1_test WERROR)
  target_link_libraries(matcore_thread_affinity_v1_test
    PRIVATE MatcoreDSL::PlatformV1
  )
  add_test(NAME platform.thread_affinity.v1
    COMMAND matcore_thread_affinity_v1_test
  )
endif()
