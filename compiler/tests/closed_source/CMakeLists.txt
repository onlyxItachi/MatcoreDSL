# Private authenticated-source execution proof: retain the strict-native control
# and, when linked, separately execute the trusted MLIR-generated mathematical leaf.
add_library(matcore_closed_host_emitter STATIC
  ../../lib/codegen/ClosedHostEmitter.cpp)
target_compile_features(matcore_closed_host_emitter PUBLIC cxx_std_20)
target_include_directories(matcore_closed_host_emitter PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/codegen")
target_link_libraries(matcore_closed_host_emitter PUBLIC
  matcore_closed_region_admission)
mdslc_target_enable_warnings(matcore_closed_host_emitter)
mdslc_target_match_llvm_rtti(matcore_closed_host_emitter)

add_executable(matcore_closed_source_execution_tests
  closed_source_execution_test.cpp)
target_link_libraries(matcore_closed_source_execution_tests PRIVATE
  matcore_closed_host_emitter MatcoreDSL::PlatformSupportV1 MLIRParser)
mdslc_target_enable_warnings(matcore_closed_source_execution_tests)
mdslc_target_match_llvm_rtti(matcore_closed_source_execution_tests)

# Instrument the actual child-compiled runtime/orchestration when the parent is
# sanitized. Linking an instrumented parent is not instrumentation of its child.
set(closed_source_child_flags)
execute_process(COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
  OUTPUT_VARIABLE closed_source_resource_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=([^ ]+)")
  list(APPEND closed_source_child_flags "-fsanitize=${CMAKE_MATCH_1}"
    -fno-omit-frame-pointer)
endif()
add_test(NAME frontend.closed_source_execution_v1
  COMMAND matcore_closed_source_execution_tests
    "${MDSLC_CLANGXX_EXECUTABLE}"
    "${closed_source_resource_dir}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    ${closed_source_child_flags})
set_tests_properties(frontend.closed_source_execution_v1 PROPERTIES TIMEOUT 180)

# Exercise the same admitted source programs through the real compile-trusted
# registry. Its archive already contains the issued LLVM object: linking another
# copy would conceal an integration error behind duplicate kernel definitions.
if(TARGET matcore_closed_candidates_production_v1)
  add_dependencies(matcore_closed_source_execution_tests
    matcore_closed_candidates_production_v1 matcore_generated_cpu_asan_test)
  add_test(NAME frontend.closed_source_generated_execution_v1
    COMMAND matcore_closed_source_execution_tests
      "${MDSLC_CLANGXX_EXECUTABLE}"
      "${closed_source_resource_dir}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../.."
      --registry "$<TARGET_FILE:matcore_closed_candidates_production_v1>"
      --runtime "$<TARGET_FILE:matcore_runtime>"
      --asan-control "$<TARGET_FILE:matcore_generated_cpu_asan_test>"
      --kernel "${CMAKE_BINARY_DIR}/tests/generated_cpu/strict-normal.o"
      --asan-kernel "${CMAKE_BINARY_DIR}/tests/generated_cpu/strict-asan.o"
      ${closed_source_child_flags})
  set_tests_properties(frontend.closed_source_generated_execution_v1
    PROPERTIES TIMEOUT 180)
endif()
