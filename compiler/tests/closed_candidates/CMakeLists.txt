# Tests share the production registry definition; only dedicated variants
# enable injection or intentionally omit linked candidates.
if(NOT TARGET matcore-cpu-gemm-candidate)
  return()
endif()
set(closed_openblas "${MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS}")
mdslc_add_closed_candidate_runtime(matcore_closed_candidates_testing_v1 TESTING)
mdslc_add_closed_candidate_runtime(matcore_closed_candidates_native_only_v1 NATIVE_ONLY)
foreach(mode IN ITEMS production testing native_only)
  set(target matcore_closed_candidates_${mode}_v1)
  add_executable(matcore_closed_candidates_${mode}_test candidate_test.cpp)
  target_link_libraries(matcore_closed_candidates_${mode}_test PRIVATE ${target})
  target_compile_options(matcore_closed_candidates_${mode}_test PRIVATE -ffp-contract=off -frounding-math)
  if(mode STREQUAL "native_only")
    target_compile_definitions(matcore_closed_candidates_${mode}_test PRIVATE EXPECT_NATIVE_ONLY)
  endif()
  if(closed_openblas)
    target_compile_definitions(matcore_closed_candidates_${mode}_test PRIVATE EXPECT_OPENBLAS)
  endif()
  add_test(NAME closed_candidates.${mode} COMMAND matcore_closed_candidates_${mode}_test)
endforeach()

add_test(NAME closed_candidates.production_symbols COMMAND "${CMAKE_COMMAND}"
  "-DNM=${MDSLC_CANDIDATE_NM}"
  "-DARCHIVE=$<TARGET_FILE:matcore_closed_candidates_production_v1>"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/symbol_contract.cmake")

# Keep all original adapter falsification fixtures against the expanded registry.
add_executable(matcore_closed_candidates_adapter_regression ../closed_host/closed_host_v1_test.cpp)
target_link_libraries(matcore_closed_candidates_adapter_regression PRIVATE matcore_closed_candidates_testing_v1)
target_compile_options(matcore_closed_candidates_adapter_regression PRIVATE -ffp-contract=off -frounding-math)
add_test(NAME closed_candidates.adapter_regression COMMAND matcore_closed_candidates_adapter_regression)

add_executable(matcore_closed_candidates_provider_negative
  provider_negative.cpp ../../lib/runtime/closed_host_v1.cpp)
target_include_directories(matcore_closed_candidates_provider_negative PRIVATE ../../lib/runtime ../../include)
target_compile_features(matcore_closed_candidates_provider_negative PRIVATE cxx_std_20)
target_compile_options(matcore_closed_candidates_provider_negative PRIVATE -fno-fast-math -ffp-contract=off -frounding-math)
target_compile_definitions(matcore_closed_candidates_provider_negative PRIVATE
  MDSLC_CLOSED_HOST_LEGACY_CANDIDATES MDSLC_CLOSED_HOST_OPENBLAS)
foreach(mode IN ITEMS wrong_numeric wrong_variant wrong_threads control_corruption partial_failure)
  add_test(NAME closed_candidates.provider_negative.${mode}
    COMMAND matcore_closed_candidates_provider_negative ${mode})
endforeach()

if(closed_openblas)
  find_package(Threads REQUIRED)
  add_executable(matcore_closed_candidates_independent_test independent_arithmetic_test.cpp)
  target_link_libraries(matcore_closed_candidates_independent_test PRIVATE
    matcore_closed_candidates_production_v1 Threads::Threads)
  target_compile_options(matcore_closed_candidates_independent_test PRIVATE
    -fno-fast-math -ffp-contract=off -frounding-math)
  add_test(NAME closed_candidates.independent_arithmetic_concurrency
    COMMAND matcore_closed_candidates_independent_test)
endif()
