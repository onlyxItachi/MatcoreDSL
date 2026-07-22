add_executable(matcore-plan main.cpp)
target_compile_features(matcore-plan PRIVATE cxx_std_20)
target_compile_options(matcore-plan PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(matcore-plan PRIVATE
  MatcoreDSL::CpuPlannerV1
  MatcoreDSL::PlatformV1
)
set_property(TARGET matcore-plan PROPERTY EXPORT_NAME PlanInspector)

if(BUILD_TESTING)
  add_test(NAME planner.platform.cli
    COMMAND matcore-plan --platform-info
  )
  set_tests_properties(
    planner.platform.cli
    PROPERTIES PASS_REGULAR_EXPRESSION
      "platform-record-v1.*platform=.*object=.*executable="
  )

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

  add_test(
    NAME planner.cpu.cli.automatic
    COMMAND matcore-plan --m 33 --k 35 --n 37 --alignment 64
  )
  set_tests_properties(
    planner.cpu.cli.automatic
    PROPERTIES
      PASS_REGULAR_EXPRESSION
        "status=selected.*selected=cpu\\..*candidates=\\["
  )

  add_test(
    NAME planner.cpu.cli.invalid_alignment
    COMMAND
      "${CMAKE_COMMAND}"
      "-DMATCORE_PLAN=$<TARGET_FILE:matcore-plan>"
      -P
      "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cmake/expect_invalid_plan_cli.cmake"
  )
endif()
