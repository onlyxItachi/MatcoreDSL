add_executable(matcore-plan main.cpp)
target_compile_features(matcore-plan PRIVATE cxx_std_20)
target_compile_options(matcore-plan PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(matcore-plan PRIVATE
  MatcoreDSL::CpuBackendsV1
  MatcoreDSL::PlatformV1
  MatcoreDSL::Runtime
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
    COMMAND matcore-plan --m 2 --k 3 --n 2 --threads 1
      --variant cpu.reference.f32.v1
  )
  set_tests_properties(
    planner.cpu.cli.reference
    PROPERTIES
      PASS_REGULAR_EXPRESSION
        "cpu-planner-v3.*selected=cpu.reference.f32.v1.*reason=explicit legal variant request"
  )

  add_test(
    NAME planner.cpu.cli.automatic
    COMMAND matcore-plan --m 33 --k 35 --n 37 --alignment 64
  )
  set_tests_properties(
    planner.cpu.cli.automatic
    PROPERTIES
      PASS_REGULAR_EXPRESSION
        "cpu-planner-v3.*selected=cpu\\..*candidates=\\["
  )

  add_test(
    NAME planner.cpu.cli.registry_v3
    COMMAND
      "${CMAKE_COMMAND}"
      "-DMATCORE_PLAN=$<TARGET_FILE:matcore-plan>"
      -P
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_registry_v3.cmake"
  )

  add_test(
    NAME planner.cpu.cli.deterministic_v3
    COMMAND
      "${CMAKE_COMMAND}"
      "-DMATCORE_PLAN=$<TARGET_FILE:matcore-plan>"
      -P
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_deterministic_v3.cmake"
  )

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(
      NAME planner.cpu.cli.compact_affinity_v3
      COMMAND
        "${CMAKE_COMMAND}"
        "-DMATCORE_PLAN=$<TARGET_FILE:matcore-plan>"
        -P
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_compact_affinity_v3.cmake"
    )
  endif()

  add_test(
    NAME planner.cpu.cli.invalid_threads
    COMMAND
      "${CMAKE_COMMAND}"
      "-DMATCORE_PLAN=$<TARGET_FILE:matcore-plan>"
      -P
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_invalid_threads.cmake"
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
