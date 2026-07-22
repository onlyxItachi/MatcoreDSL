add_library(matcore_cpu_planner_v1 INTERFACE)
add_library(MatcoreDSL::CpuPlannerV1 ALIAS matcore_cpu_planner_v1)
target_compile_features(matcore_cpu_planner_v1 INTERFACE cxx_std_20)
target_include_directories(matcore_cpu_planner_v1 INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
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
endif()
