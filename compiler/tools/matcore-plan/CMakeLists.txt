add_executable(matcore-plan main.cpp)
target_compile_features(matcore-plan PRIVATE cxx_std_20)
target_compile_options(matcore-plan PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(matcore-plan PRIVATE MatcoreDSL::CpuPlannerV1)
set_property(TARGET matcore-plan PROPERTY EXPORT_NAME PlanInspector)

if(BUILD_TESTING)
  add_test(
    NAME planner.cpu.cli.reference
    COMMAND matcore-plan --m 2 --k 3 --n 2 --variant reference
  )
  set_tests_properties(
    planner.cpu.cli.reference
    PROPERTIES
      PASS_REGULAR_EXPRESSION
        "selected=cpu.reference.f32.v1.*reason=explicit legal variant request"
  )
endif()
