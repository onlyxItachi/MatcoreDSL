add_library(matcore_codegen_v0 STATIC codegen.cpp)
target_compile_features(matcore_codegen_v0 PUBLIC cxx_std_20)
target_include_directories(matcore_codegen_v0
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(matcore_codegen_v0 PUBLIC matcore_ir_v0)
mdslc_target_enable_warnings(matcore_codegen_v0)

if(BUILD_TESTING)
  add_executable(matcore_runtime_dispatch_backend_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/runtime_dispatch_backend_test.cpp"
  )
  target_compile_features(matcore_runtime_dispatch_backend_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_runtime_dispatch_backend_test
    PRIVATE matcore_codegen_v0
  )
  mdslc_target_enable_warnings(matcore_runtime_dispatch_backend_test WERROR)
  add_test(NAME codegen.runtime_dispatch_backend_v1
    COMMAND matcore_runtime_dispatch_backend_test
  )

  add_executable(matcore_static_aot_specialization_execution_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/static_aot_specialization_execution_test.cpp"
  )
  target_compile_features(matcore_static_aot_specialization_execution_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_static_aot_specialization_execution_test
    PRIVATE matcore_codegen_v0 matcore_runtime
  )
  target_include_directories(matcore_static_aot_specialization_execution_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  )
  mdslc_target_enable_warnings(matcore_static_aot_specialization_execution_test WERROR)
  add_test(NAME codegen.static_aot_specialization_execution
    COMMAND matcore_static_aot_specialization_execution_test
  )
  add_executable(matcore_numerical_semantics_adversarial_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/numerical_semantics_adversarial_test.cpp"
  )
  target_compile_features(matcore_numerical_semantics_adversarial_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_numerical_semantics_adversarial_test
    PRIVATE matcore_codegen_v0 matcore_runtime
  )
  target_include_directories(matcore_numerical_semantics_adversarial_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  )
  mdslc_target_enable_warnings(matcore_numerical_semantics_adversarial_test WERROR)
  add_test(NAME codegen.numerical_semantics_adversarial
    COMMAND matcore_numerical_semantics_adversarial_test
  )
  add_executable(matcore_aliasing_and_alignment_policy_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/aliasing_and_alignment_policy_test.cpp"
  )
  target_compile_features(matcore_aliasing_and_alignment_policy_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_aliasing_and_alignment_policy_test
    PRIVATE matcore_codegen_v0 matcore_runtime
  )
  target_include_directories(matcore_aliasing_and_alignment_policy_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  )
  mdslc_target_enable_warnings(matcore_aliasing_and_alignment_policy_test WERROR)
  add_test(NAME codegen.aliasing_and_alignment_policy
    COMMAND matcore_aliasing_and_alignment_policy_test
  )
  find_program(LLVM_OBJDUMP_EXECUTABLE
    NAMES llvm-objdump llvm-objdump.exe llvm-objdump-21 llvm-objdump-21.1.8
    HINTS "C:/Users/hamza/tools/llvm-21.1.8/bin"
    ENV PATH
  )
  find_program(Python3_EXECUTABLE
    NAMES python python3 py
    ENV PATH
  )
  if(Python3_EXECUTABLE AND LLVM_OBJDUMP_EXECUTABLE)
    add_test(NAME codegen.zero_runtime_call_and_spill_proof
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/run_zero_runtime_call_and_spill_proof_test.py"
        --clangxx "${MDSLC_CLANGXX_EXECUTABLE}"
        --objdump "${LLVM_OBJDUMP_EXECUTABLE}"
        --include-dir "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    )
    add_test(NAME codegen.binary_bloat_and_compile_time_stress
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/run_binary_bloat_and_compile_time_stress_test.py"
        --clangxx "${MDSLC_CLANGXX_EXECUTABLE}"
        --include-dir "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    )
  endif()
  add_executable(matcore_real_performance_crossover_bench
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/real_performance_crossover_bench.cpp"
  )
  target_compile_features(matcore_real_performance_crossover_bench PRIVATE cxx_std_20)
  target_link_libraries(matcore_real_performance_crossover_bench
    PRIVATE matcore_codegen_v0 matcore_runtime
  )
  target_include_directories(matcore_real_performance_crossover_bench
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  )
  add_test(NAME codegen.real_performance_crossover
    COMMAND matcore_real_performance_crossover_bench
  )
  add_executable(matcore_prepacked_b_hoisting_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/prepacked_b_hoisting_test.cpp"
  )
  target_compile_features(matcore_prepacked_b_hoisting_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_prepacked_b_hoisting_test
    PRIVATE matcore_codegen_v0 matcore_runtime
  )
  target_include_directories(matcore_prepacked_b_hoisting_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  )
  mdslc_target_enable_warnings(matcore_prepacked_b_hoisting_test WERROR)
  add_test(NAME codegen.prepacked_b_hoisting
    COMMAND matcore_prepacked_b_hoisting_test
  )
  add_executable(matcore_triway_specialization_comparison_test
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/codegen/triway_specialization_comparison_test.cpp"
  )
  target_compile_features(matcore_triway_specialization_comparison_test PRIVATE cxx_std_20)
  target_link_libraries(matcore_triway_specialization_comparison_test
    PRIVATE matcore_codegen_v0 matcore_runtime
  )
  target_include_directories(matcore_triway_specialization_comparison_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  )
  mdslc_target_enable_warnings(matcore_triway_specialization_comparison_test)
  add_test(NAME codegen.triway_specialization_comparison
    COMMAND matcore_triway_specialization_comparison_test
  )
endif()
