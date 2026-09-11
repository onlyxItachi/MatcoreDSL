add_test(NAME private_runtime.symbol_contract
  COMMAND "${CMAKE_COMMAND}"
    "-DLIBRARY=$<TARGET_FILE:matcore_closed_candidates_isolated_v1>"
    "-DEXPORTS=${CMAKE_CURRENT_SOURCE_DIR}/../../lib/regions/private_runtime_v2.exports"
    "-DNM=${CMAKE_NM}" "-DREADELF=${CMAKE_READELF}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/symbol_contract.cmake")

foreach(kind IN ITEMS candidates values results)
  if(kind STREQUAL "candidates")
    set(source ../closed_candidates/candidate_test.cpp ../../lib/platform/closed_cpu_isa_v1.cpp)
  elseif(kind STREQUAL "values")
    set(source ../closed_host/private_value_independent_test.cpp)
  else()
    set(source ../experimental_region/result_test.cpp)
  endif()
  add_executable(matcore_isolated_runtime_${kind}_test ${source})
  target_link_libraries(matcore_isolated_runtime_${kind}_test PRIVATE
    matcore_closed_candidates_isolated_v1)
  target_compile_options(matcore_isolated_runtime_${kind}_test PRIVATE
    -fno-fast-math -ffp-contract=off -frounding-math)
  if(MDSLC_CLOSED_CANDIDATES_HAS_OPENBLAS AND kind STREQUAL "candidates")
    target_compile_definitions(matcore_isolated_runtime_${kind}_test PRIVATE EXPECT_OPENBLAS)
  endif()
  add_test(NAME private_runtime.${kind} COMMAND matcore_isolated_runtime_${kind}_test)
endforeach()

foreach(spelling IN ITEMS assembler named)
  foreach(kind IN ITEMS isolated archive)
    set(target matcore_private_cleanup_${kind}_${spelling})
    add_executable(${target} weak_cleanup.cpp)
    if(spelling STREQUAL "named")
      target_compile_definitions(${target} PRIVATE NAMED_DEFINITION)
      set(expected 72)
    else()
      set(expected 71)
    endif()
    if(kind STREQUAL "isolated")
      target_link_libraries(${target} PRIVATE matcore_closed_candidates_isolated_v1)
      add_test(NAME private_runtime.cleanup.${spelling} COMMAND ${target})
    else()
      target_link_libraries(${target} PRIVATE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,matcore_closed_candidates_production_v1>")
      set(required_reproduction ON)
      set(control_name archive_negative)
      if(MDSLC_CLOSED_REGION_CXX_FLAGS MATCHES "fsanitize=")
        # The actual ASan lowering currently inlines the failure cleanup: the
        # Release weak-symbol selection counterexample is not universal codegen.
        set(required_reproduction OFF)
        set(control_name archive_observation)
      endif()
      add_test(NAME private_runtime.${control_name}.${spelling}
        COMMAND "${CMAKE_COMMAND}" "-DPROGRAM=$<TARGET_FILE:${target}>"
          "-DEXPECTED=${expected}" "-DREQUIRED_REPRODUCTION=${required_reproduction}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/archive_negative.cmake")
    endif()
  endforeach()
endforeach()
