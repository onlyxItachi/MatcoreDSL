add_library(matcore_cpu_planner_v1 INTERFACE)
add_library(MatcoreDSL::CpuPlannerV1 ALIAS matcore_cpu_planner_v1)
target_compile_features(matcore_cpu_planner_v1 INTERFACE cxx_std_20)
target_include_directories(matcore_cpu_planner_v1 INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)

add_library(matcore_cpu_planner_v3 STATIC
  cpu_planner_v3.cpp
)
add_library(MatcoreDSL::CpuPlannerV3 ALIAS matcore_cpu_planner_v3)
target_compile_features(matcore_cpu_planner_v3 PUBLIC cxx_std_20)
target_include_directories(matcore_cpu_planner_v3 PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(matcore_cpu_planner_v3
  PUBLIC MatcoreDSL::CpuPlannerV1 MatcoreDSL::PlatformV1
)
set_target_properties(matcore_cpu_planner_v3 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(BUILD_TESTING)
  add_executable(matcore_cpu_planner_v2_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/cpu_planner_v2_test.cpp"
  )
  target_compile_features(matcore_cpu_planner_v2_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_cpu_planner_v2_test
    PRIVATE MatcoreDSL::CpuPlannerV1
  )
  add_test(NAME planner.cpu.v2.resources
    COMMAND matcore_cpu_planner_v2_test
  )

  add_executable(matcore_cpu_planner_v3_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/runtime/cpu_planner_v3_test.cpp"
  )
  target_compile_features(matcore_cpu_planner_v3_test PRIVATE cxx_std_20)
  mdslc_target_enable_warnings(matcore_cpu_planner_v3_test WERROR)
  target_link_libraries(matcore_cpu_planner_v3_test
    PRIVATE MatcoreDSL::CpuPlannerV3
  )
  add_test(NAME planner.cpu.v3.advanced
    COMMAND matcore_cpu_planner_v3_test
  )
endif()
