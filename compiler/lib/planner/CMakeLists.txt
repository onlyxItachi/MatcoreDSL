add_library(matcore_cpu_planner_v1 INTERFACE)
add_library(MatcoreDSL::CpuPlannerV1 ALIAS matcore_cpu_planner_v1)
target_compile_features(matcore_cpu_planner_v1 INTERFACE cxx_std_20)
target_include_directories(matcore_cpu_planner_v1 INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
