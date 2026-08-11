add_library(matcore_platform_v1 STATIC
  platform.cpp
  fp_environment_v1.cpp
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

# The inspector must observe, not normalize or assume, the current thread's
# floating-point environment. Keep the Windows backend under the documented
# strict MSVC floating-point model; GNU-mode Clang/GCC use the equivalent
# environment-sensitive compilation flags for this one translation unit.
if(MDSLC_USING_MSVC_ABI_FRONTEND)
  set_source_files_properties(fp_environment_v1.cpp PROPERTIES
    COMPILE_OPTIONS "/fp:strict"
  )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set_source_files_properties(fp_environment_v1.cpp PROPERTIES
    COMPILE_OPTIONS "-frounding-math;-ftrapping-math;-ffp-contract=off"
  )
endif()

if(BUILD_TESTING)
  add_executable(
    matcore_platform_v1_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/platform/platform_test.cpp"
  )
  target_compile_features(matcore_platform_v1_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_platform_v1_test PRIVATE MatcoreDSL::PlatformV1)
  add_test(NAME platform.record.v1 COMMAND matcore_platform_v1_test)

  add_executable(
    matcore_fp_environment_v1_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/platform/fp_environment_v1_test.cpp"
  )
  target_compile_features(matcore_fp_environment_v1_test PRIVATE cxx_std_20)
  mdslc_target_enable_warnings(matcore_fp_environment_v1_test WERROR)
  if(MDSLC_USING_MSVC_ABI_FRONTEND)
    target_compile_options(matcore_fp_environment_v1_test PRIVATE /fp:strict)
  endif()
  target_link_libraries(matcore_fp_environment_v1_test
    PRIVATE MatcoreDSL::PlatformV1
  )
  add_test(NAME platform.fp_environment.v1
    COMMAND matcore_fp_environment_v1_test
  )

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
