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
