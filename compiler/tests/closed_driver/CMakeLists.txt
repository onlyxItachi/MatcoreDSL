# Original-host code generation mechanics, not yet an installed entry point.
add_library(matcore_frozen_host_codegen STATIC ../../lib/codegen/FrozenHostCodegen.cpp)
target_compile_features(matcore_frozen_host_codegen PUBLIC cxx_std_20)
target_include_directories(matcore_frozen_host_codegen PUBLIC ../../lib/codegen)
target_include_directories(matcore_frozen_host_codegen SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_libraries(matcore_frozen_host_codegen PUBLIC matcore_closed_region_admission)
mdslc_target_enable_warnings(matcore_frozen_host_codegen)
mdslc_target_match_llvm_rtti(matcore_frozen_host_codegen)
add_executable(matcore_frozen_host_codegen_test frozen_host_test.cpp)
target_include_directories(matcore_frozen_host_codegen_test SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_libraries(matcore_frozen_host_codegen_test PRIVATE
  matcore_frozen_host_codegen MatcoreDSL::PlatformSupportV1)
mdslc_target_match_llvm_rtti(matcore_frozen_host_codegen_test)
execute_process(COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
  OUTPUT_VARIABLE closed_driver_resource_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
add_test(NAME frontend.frozen_host_codegen
  COMMAND matcore_frozen_host_codegen_test
  "${MDSLC_CLANGXX_EXECUTABLE}" "${closed_driver_resource_dir}")

add_executable(matcore_frozen_host_independent_test frozen_host_independent_test.cpp)
target_include_directories(matcore_frozen_host_independent_test SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_libraries(matcore_frozen_host_independent_test PRIVATE
  matcore_frozen_host_codegen MatcoreDSL::PlatformSupportV1)
mdslc_target_match_llvm_rtti(matcore_frozen_host_independent_test)
add_test(NAME frontend.frozen_host_independent
  COMMAND matcore_frozen_host_independent_test
  "${MDSLC_CLANGXX_EXECUTABLE}" "${closed_driver_resource_dir}")

# Private composition proof until the separately reviewed installation boundary.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/runtime/closed_host_v1.h"
  MDSLC_PRIVATE_REGION_RUNTIME_HEADER)
configure_file("../../lib/codegen/ExperimentalRegionRuntimeHeader.inc.in"
  "ExperimentalRegionRuntimeHeader.inc" @ONLY)
add_library(matcore_experimental_region_compiler STATIC
  ../../lib/codegen/ExperimentalRegionEmitter.cpp
  ../../lib/codegen/ExperimentalRegionCompiler.cpp)
target_compile_features(matcore_experimental_region_compiler PUBLIC cxx_std_20)
target_include_directories(matcore_experimental_region_compiler PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(matcore_experimental_region_compiler SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_libraries(matcore_experimental_region_compiler PUBLIC
  matcore_frozen_host_codegen matcore_closed_host_emitter
  matcore_authenticated_host_thunk PRIVATE LLVMAsmParser)
mdslc_target_enable_warnings(matcore_experimental_region_compiler WERROR)
mdslc_target_match_llvm_rtti(matcore_experimental_region_compiler)

add_executable(matcore_experimental_compile_test experimental_compile_test.cpp)
target_link_libraries(matcore_experimental_compile_test PRIVATE
  matcore_experimental_region_compiler MatcoreDSL::PlatformSupportV1)
mdslc_target_enable_warnings(matcore_experimental_compile_test WERROR)
mdslc_target_match_llvm_rtti(matcore_experimental_compile_test)
add_dependencies(matcore_experimental_compile_test matcore_closed_candidates_production_v1)
set(experimental_compiler_test_flags)
if(CMAKE_CXX_FLAGS MATCHES "fsanitize=.*address")
  list(APPEND experimental_compiler_test_flags --asan)
endif()
add_test(NAME frontend.experimental_source_generated_execution
  COMMAND matcore_experimental_compile_test
  "${MDSLC_CLANGXX_EXECUTABLE}" "${closed_driver_resource_dir}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  "$<TARGET_FILE:matcore_closed_candidates_production_v1>"
  "$<TARGET_FILE:matcore_runtime>" ${experimental_compiler_test_flags})
set_tests_properties(frontend.experimental_source_generated_execution PROPERTIES TIMEOUT 180)
