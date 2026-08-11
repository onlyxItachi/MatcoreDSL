cmake_minimum_required(VERSION 3.24)

project(MatcoreDSLInstalledConsumer LANGUAGES CXX)

find_package(MatcoreDSL 0.1 CONFIG REQUIRED)

set(MATCOREDSL_CONSUMER_SEMANTIC_PIPELINE "" CACHE STRING
    "Optional semantic-pipeline override for package validation")
set(MATCOREDSL_CONSUMER_FRONTEND "native" CACHE STRING
    "Frontend selected by the installed-consumer validation")
set(MATCOREDSL_EXPECT_MATCORE_MLIR_AVAILABLE "" CACHE STRING
    "Expected installed Matcore MLIR capability for package validation")
set(MATCOREDSL_EXPECT_DEFAULT_SEMANTIC_PIPELINE "" CACHE STRING
    "Expected installed default semantic pipeline for package validation")

if(NOT "${MATCOREDSL_EXPECT_MATCORE_MLIR_AVAILABLE}" STREQUAL "" AND
   NOT "${MatcoreDSL_MATCORE_MLIR_AVAILABLE}" STREQUAL
       "${MATCOREDSL_EXPECT_MATCORE_MLIR_AVAILABLE}")
  message(FATAL_ERROR
    "installed Matcore MLIR capability mismatch: expected "
    "${MATCOREDSL_EXPECT_MATCORE_MLIR_AVAILABLE}, got "
    "${MatcoreDSL_MATCORE_MLIR_AVAILABLE}")
endif()
if(NOT "${MATCOREDSL_EXPECT_DEFAULT_SEMANTIC_PIPELINE}" STREQUAL "" AND
   NOT "${MatcoreDSL_DEFAULT_SEMANTIC_PIPELINE}" STREQUAL
       "${MATCOREDSL_EXPECT_DEFAULT_SEMANTIC_PIPELINE}")
  message(FATAL_ERROR
    "installed default semantic pipeline mismatch: expected "
    "${MATCOREDSL_EXPECT_DEFAULT_SEMANTIC_PIPELINE}, got "
    "${MatcoreDSL_DEFAULT_SEMANTIC_PIPELINE}")
endif()

set(MATCOREDSL_CONSUMER_SEMANTIC_ARGUMENTS)
if(NOT "${MATCOREDSL_CONSUMER_SEMANTIC_PIPELINE}" STREQUAL "")
  list(APPEND MATCOREDSL_CONSUMER_SEMANTIC_ARGUMENTS
       SEMANTIC_PIPELINE "${MATCOREDSL_CONSUMER_SEMANTIC_PIPELINE}")
endif()

matcoredsl_add_executable(
  matcore_consumer
  SOURCE consumer.mdsl
  CXX_STANDARD 20
  MATCORE_TARGET cpu
  FRONTEND "${MATCOREDSL_CONSUMER_FRONTEND}"
  ${MATCOREDSL_CONSUMER_SEMANTIC_ARGUMENTS}
)

enable_testing()
add_test(NAME matcore_consumer.runs COMMAND matcore_consumer)
if(WIN32)
  set_tests_properties(matcore_consumer.runs PROPERTIES
    ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$<TARGET_FILE_DIR:MatcoreDSL::Runtime>"
  )
endif()
