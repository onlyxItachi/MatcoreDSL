add_executable(matcore_spirv_issuer_test issuer_test.cpp)
target_link_libraries(matcore_spirv_issuer_test PRIVATE matcore_spirv_gemm_candidate LLVM)
mdslc_target_match_llvm_rtti(matcore_spirv_issuer_test)
add_test(NAME generated_spirv.issuer COMMAND matcore_spirv_issuer_test)
add_executable(matcore_spirv_adversarial_test adversarial_test.cpp)
target_link_libraries(matcore_spirv_adversarial_test PRIVATE
  matcore_spirv_gemm_candidate MLIRParser LLVM)
mdslc_target_match_llvm_rtti(matcore_spirv_adversarial_test)
add_test(NAME generated_spirv.adversarial COMMAND matcore_spirv_adversarial_test)

# Test harness only, never a source/runtime candidate or installed component.
# Physical runs require an explicit device and are not automatic CTest dispatch.
option(MDSLC_BUILD_SPIRV_VULKAN_PROBE "Build the private Vulkan correctness probe" OFF)
if(MDSLC_BUILD_SPIRV_VULKAN_PROBE)
  find_path(MDSLC_SPIRV_VULKAN_INCLUDE_DIR vulkan/vulkan.h REQUIRED)
  add_executable(matcore_spirv_vulkan_probe vulkan_probe.cpp)
  target_include_directories(matcore_spirv_vulkan_probe SYSTEM PRIVATE
    "${MDSLC_SPIRV_VULKAN_INCLUDE_DIR}")
  target_link_libraries(matcore_spirv_vulkan_probe PRIVATE
    matcore_spirv_gemm_candidate LLVM ${CMAKE_DL_LIBS})
  mdslc_target_match_llvm_rtti(matcore_spirv_vulkan_probe)
  mdslc_target_enable_warnings(matcore_spirv_vulkan_probe)
  add_test(NAME generated_spirv.oracle COMMAND matcore_spirv_vulkan_probe --oracle-only)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME generated_spirv.receipt_validator
  COMMAND "${Python3_EXECUTABLE}" -B
    "${CMAKE_CURRENT_SOURCE_DIR}/physical_qualification.py" --self-test)
# External validation is a separate optional gate; its absence is not a pass.
find_program(MDSLC_SPIRV_VAL NAMES spirv-val)
find_program(MDSLC_SPIRV_DIS NAMES spirv-dis)
if(MDSLC_SPIRV_VAL AND MDSLC_SPIRV_DIS)
  add_test(NAME generated_spirv.external_validation
    COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/external_validation.py"
      --issuer "$<TARGET_FILE:matcore-spirv-gemm-candidate>"
      --validator "${MDSLC_SPIRV_VAL}" --disassembler "${MDSLC_SPIRV_DIS}"
      --output "${CMAKE_CURRENT_BINARY_DIR}/external-validation")
  set_tests_properties(generated_spirv.external_validation PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
else()
  message(STATUS "SPIRV-Tools absent: external SPIR-V validation NOT registered/qualified")
endif()
