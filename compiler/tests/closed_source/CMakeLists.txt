# Private authenticated-source execution proof. Generated C++ orchestrates the
# new strict native adapter; this does not claim an MLIR-generated math kernel.
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
