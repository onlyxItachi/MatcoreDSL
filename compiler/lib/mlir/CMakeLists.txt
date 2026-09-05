set(LLVM_TARGET_DEFINITIONS MatcoreDialect.td)
mlir_tablegen(MatcoreDialect.h.inc -gen-dialect-decls -dialect=mdsl
  EXTRA_INCLUDES "${MLIR_INCLUDE_DIRS}")
mlir_tablegen(MatcoreDialect.cpp.inc -gen-dialect-defs -dialect=mdsl
  EXTRA_INCLUDES "${MLIR_INCLUDE_DIRS}")
add_public_tablegen_target(MatcoreDialectIncGen)

set(LLVM_TARGET_DEFINITIONS MatcoreOps.td)
mlir_tablegen(MatcoreOps.h.inc -gen-op-decls
  EXTRA_INCLUDES "${MLIR_INCLUDE_DIRS}")
mlir_tablegen(MatcoreOps.cpp.inc -gen-op-defs
  EXTRA_INCLUDES "${MLIR_INCLUDE_DIRS}")
add_public_tablegen_target(MatcoreOpsIncGen)

add_library(
  matcore_mlir_semantics
  STATIC
    MatcoreDialect.cpp
    MatcoreGemmSemanticBuilder.cpp
    MatcoreOps.cpp
    MatcoreV1Bridge.cpp
    MatcoreCpuRuntimeLowering.cpp
)
add_dependencies(
  matcore_mlir_semantics
  MatcoreDialectIncGen
  MatcoreOpsIncGen
)
target_compile_features(matcore_mlir_semantics PUBLIC cxx_std_20)
target_include_directories(
  matcore_mlir_semantics
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
)
target_include_directories(
  matcore_mlir_semantics
  SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>"
    "$<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>"
  PRIVATE
    "${LLVM_INCLUDE_DIRS}"
    "${MLIR_INCLUDE_DIRS}"
)
target_link_libraries(
  matcore_mlir_semantics
  PUBLIC
    matcore_ir_v1
  PRIVATE
    MLIRIR
    MLIRFuncDialect
    MLIRDestinationStyleOpInterface
    MLIRSideEffectInterfaces
)
mdslc_target_enable_warnings(matcore_mlir_semantics)
mdslc_target_match_llvm_rtti(matcore_mlir_semantics)

# The first structured projection is intentionally a separate internal layer:
# semantic WHAT remains independent of Linalg/Arith transformation machinery.
add_library(
  matcore_mlir_structured_handoff
  STATIC
    MatcoreContractionModel.cpp
    MatcoreStructuredHandoffCertificate.cpp
    MatcoreStructuredGemmHandoff.cpp
)
target_compile_features(matcore_mlir_structured_handoff PUBLIC cxx_std_20)
target_include_directories(
  matcore_mlir_structured_handoff
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_include_directories(
  matcore_mlir_structured_handoff
  SYSTEM PRIVATE
    "${LLVM_INCLUDE_DIRS}"
    "${MLIR_INCLUDE_DIRS}"
)
target_link_libraries(
  matcore_mlir_structured_handoff
  PUBLIC
    matcore_mlir_semantics
  PRIVATE
    MLIRArithDialect
    MLIRLinalgDialect
)
mdslc_target_enable_warnings(matcore_mlir_structured_handoff)
mdslc_target_match_llvm_rtti(matcore_mlir_structured_handoff)

# Bufferization is a separate, inspection-only proof boundary. It consumes the
# exact structured handoff through upstream One-Shot Bufferize without making
# the semantic or runtime layers depend on buffer transformation machinery.
add_library(
  matcore_mlir_bufferized_handoff
  STATIC
    MatcoreBufferizedGemmHandoff.cpp
)
target_compile_features(matcore_mlir_bufferized_handoff PUBLIC cxx_std_20)
target_include_directories(
  matcore_mlir_bufferized_handoff
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_include_directories(
  matcore_mlir_bufferized_handoff
  SYSTEM PRIVATE
    "${LLVM_INCLUDE_DIRS}"
    "${MLIR_INCLUDE_DIRS}"
)
target_link_libraries(
  matcore_mlir_bufferized_handoff
  PUBLIC
    matcore_mlir_structured_handoff
  PRIVATE
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRTensorTransforms
)
mdslc_target_enable_warnings(matcore_mlir_bufferized_handoff)
mdslc_target_match_llvm_rtti(matcore_mlir_bufferized_handoff)

if(MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS)
  # This is a downstream, inspection-only proof seam. The implementation runs
  # an explicit upstream Transform dialect schedule against a verified
  # structured handoff; it is not part of semantic WHAT, the core Matcore MLIR
  # dependency baseline, or the executable runtime route.
  add_library(
    matcore_mlir_vector_readiness
    STATIC
      MatcoreStructuredGemmVectorReadiness.cpp
  )
  target_compile_features(matcore_mlir_vector_readiness PUBLIC cxx_std_20)
  target_include_directories(
    matcore_mlir_vector_readiness
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  )
  target_include_directories(
    matcore_mlir_vector_readiness
    SYSTEM PRIVATE
      "${LLVM_INCLUDE_DIRS}"
      "${MLIR_INCLUDE_DIRS}"
  )
  target_link_libraries(
    matcore_mlir_vector_readiness
    PUBLIC
      matcore_mlir_structured_handoff
    PRIVATE
      MLIRLinalgTransformOps
      MLIRParser
      MLIRTensorDialect
      MLIRTransformDialect
      MLIRTransformDialectTransforms
      MLIRUBDialect
      MLIRVectorDialect
  )
  mdslc_target_enable_warnings(matcore_mlir_vector_readiness)
  if(COMMAND mdslc_target_match_llvm_rtti)
    mdslc_target_match_llvm_rtti(matcore_mlir_vector_readiness)
  elseif(DEFINED LLVM_ENABLE_RTTI AND NOT LLVM_ENABLE_RTTI)
    message(FATAL_ERROR
      "matcore_mlir_vector_readiness requires the project LLVM RTTI matching "
      "helper when the selected package disables RTTI")
  endif()
endif()

if(MDSLC_ENABLE_NATIVE_FRONTEND AND NOT WIN32)
  # Recovery analysis is kept in a separate internal library so the general
  # semantic dialect and explicit-v1 bridge never depend on Clang frontend
  # implementation details.
  add_library(
    matcore_mlir_recovered_analysis
    STATIC
      MatcoreRecoveredGemmBridge.cpp
  )
  target_compile_features(matcore_mlir_recovered_analysis PUBLIC cxx_std_20)
  target_include_directories(
    matcore_mlir_recovered_analysis
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  )
  target_include_directories(
    matcore_mlir_recovered_analysis
    SYSTEM PRIVATE
      "${LLVM_INCLUDE_DIRS}"
      "${MLIR_INCLUDE_DIRS}"
  )
  target_link_libraries(
    matcore_mlir_recovered_analysis
    PUBLIC
      matcore_mlir_semantics
      matcore_frontend_native
  )
  mdslc_target_enable_warnings(matcore_mlir_recovered_analysis)
  mdslc_target_match_llvm_rtti(matcore_mlir_recovered_analysis)
endif()

if(BUILD_TESTING)
  add_executable(
    matcore_mlir_contraction_model_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_contraction_model_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_contraction_model_tests
    PRIVATE
      matcore_mlir_structured_handoff
      MLIRParser
  )
  target_include_directories(
    matcore_mlir_contraction_model_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  mdslc_target_enable_warnings(matcore_mlir_contraction_model_tests)
  add_test(NAME mlir.structured.contraction-model-v1
    COMMAND matcore_mlir_contraction_model_tests)

  add_executable(
    matcore_mlir_semantics_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_mlir_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_semantics_tests
    PRIVATE
      matcore_mlir_semantics
      MLIRParser
      MLIRTransforms
  )
  target_include_directories(
    matcore_mlir_semantics_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_semantics_tests
    PRIVATE
      MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_semantics_tests)
  add_test(NAME mlir.semantic.core COMMAND matcore_mlir_semantics_tests)

  add_executable(
    matcore_mlir_map_domain_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_map_domain_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_map_domain_tests
    PRIVATE
      matcore_mlir_semantics
      MLIRParser
  )
  target_include_directories(
    matcore_mlir_map_domain_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_map_domain_tests
    PRIVATE
      MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_map_domain_tests)
  add_test(NAME mlir.semantic.map-domain COMMAND matcore_mlir_map_domain_tests)

  add_executable(
    matcore_mlir_structured_gemm_handoff_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_structured_gemm_handoff_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_structured_gemm_handoff_tests
    PRIVATE
      matcore_mlir_structured_handoff
      MLIRParser
  )
  target_include_directories(
    matcore_mlir_structured_gemm_handoff_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_structured_gemm_handoff_tests
    PRIVATE
      MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_structured_gemm_handoff_tests)
  add_test(NAME mlir.structured.gemm-handoff-v1
    COMMAND matcore_mlir_structured_gemm_handoff_tests)

  if(MDSLC_ENABLE_EXPERIMENTAL_MLIR_VECTOR_READINESS)
    add_executable(
      matcore_mlir_structured_gemm_vector_readiness_tests
      "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_structured_gemm_vector_readiness_test.cpp"
    )
    target_link_libraries(
      matcore_mlir_structured_gemm_vector_readiness_tests
      PRIVATE
        matcore_mlir_vector_readiness
        MLIRParser
    )
    target_include_directories(
      matcore_mlir_structured_gemm_vector_readiness_tests
      SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
    )
    target_compile_definitions(
      matcore_mlir_structured_gemm_vector_readiness_tests
      PRIVATE
        MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
        MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
    )
    mdslc_target_enable_warnings(
      matcore_mlir_structured_gemm_vector_readiness_tests)
    add_test(NAME mlir.structured.gemm-vector-readiness-v1
      COMMAND matcore_mlir_structured_gemm_vector_readiness_tests)
  endif()

  add_executable(
    matcore_mlir_derived_handoff_certificate_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_derived_handoff_certificate_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_derived_handoff_certificate_tests
    PRIVATE
      matcore_mlir_structured_handoff
  )
  target_include_directories(
    matcore_mlir_derived_handoff_certificate_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  mdslc_target_enable_warnings(matcore_mlir_derived_handoff_certificate_tests)
  mdslc_target_match_llvm_rtti(matcore_mlir_derived_handoff_certificate_tests)
  add_test(NAME mlir.structured.derived-certificate-v1
    COMMAND matcore_mlir_derived_handoff_certificate_tests)

  add_executable(
    matcore_mlir_bufferized_gemm_handoff_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_bufferized_gemm_handoff_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_bufferized_gemm_handoff_tests
    PRIVATE
      matcore_mlir_bufferized_handoff
      MLIRParser
  )
  target_include_directories(
    matcore_mlir_bufferized_gemm_handoff_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_bufferized_gemm_handoff_tests
    PRIVATE
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_bufferized_gemm_handoff_tests)
  mdslc_target_match_llvm_rtti(matcore_mlir_bufferized_gemm_handoff_tests)
  add_test(NAME mlir.bufferization.gemm-handoff-v1
    COMMAND matcore_mlir_bufferized_gemm_handoff_tests)

  add_executable(
    matcore_mlir_cpu_runtime_lowering_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_cpu_runtime_lowering_test.cpp"
  )
  target_link_libraries(
    matcore_mlir_cpu_runtime_lowering_tests
    PRIVATE
      matcore_mlir_semantics
      MLIRParser
  )
  target_include_directories(
    matcore_mlir_cpu_runtime_lowering_tests
    SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
  )
  target_compile_definitions(
    matcore_mlir_cpu_runtime_lowering_tests
    PRIVATE
      MDSLC_MLIR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir"
      MDSLC_IR_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
  )
  mdslc_target_enable_warnings(matcore_mlir_cpu_runtime_lowering_tests)
  add_test(NAME mlir.cpu.runtime_dispatch_lowering_v1
    COMMAND matcore_mlir_cpu_runtime_lowering_tests
  )

  if(MDSLC_ENABLE_NATIVE_FRONTEND AND NOT WIN32)
    execute_process(
      COMMAND "${MDSLC_CLANGXX_EXECUTABLE}" -print-resource-dir
      OUTPUT_VARIABLE MDSLC_RECOVERED_TEST_RESOURCE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE MDSLC_RECOVERED_TEST_RESOURCE_STATUS
    )
    if(NOT MDSLC_RECOVERED_TEST_RESOURCE_STATUS EQUAL 0 OR
       NOT IS_DIRECTORY "${MDSLC_RECOVERED_TEST_RESOURCE_DIR}")
      message(FATAL_ERROR
        "recovered MLIR tests require the configured Clang resource directory")
    endif()
    add_executable(
      matcore_mlir_recovered_gemm_tests
      "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/mlir/matcore_recovered_gemm_bridge_test.cpp"
    )
    target_link_libraries(
      matcore_mlir_recovered_gemm_tests
      PRIVATE
        matcore_mlir_recovered_analysis
    )
    target_include_directories(
      matcore_mlir_recovered_gemm_tests
      SYSTEM PRIVATE "${LLVM_INCLUDE_DIRS}" "${MLIR_INCLUDE_DIRS}"
    )
    target_compile_definitions(
      matcore_mlir_recovered_gemm_tests
      PRIVATE
        MDSLC_RECOVERED_TEST_CLANG="${MDSLC_CLANGXX_EXECUTABLE}"
        MDSLC_RECOVERED_TEST_RESOURCE_DIR="${MDSLC_RECOVERED_TEST_RESOURCE_DIR}"
        MDSLC_RECOVERED_TEST_PUBLIC_HEADER="${CMAKE_CURRENT_SOURCE_DIR}/../../include/matcore/mdsl.h"
        MDSLC_RECOVERED_TEST_FRONTEND_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/frontend/recovered"
        MDSLC_RECOVERED_TEST_IR_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ir"
    )
    mdslc_target_enable_warnings(matcore_mlir_recovered_gemm_tests)
    add_test(NAME mlir.semantic.recovered-gemm
      COMMAND matcore_mlir_recovered_gemm_tests)
  endif()
endif()
