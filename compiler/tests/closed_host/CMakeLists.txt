# The production variant has no candidate injection exports. Compile tests that
# require fault injection against a separate variant rather than exposing an
# arbitrary callback in the trusted runtime interface.
foreach(target IN ITEMS matcore_closed_host_v1 matcore_closed_host_test_v1)
  add_library(${target} STATIC ../../lib/runtime/closed_host_v1.cpp)
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_include_directories(${target} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/runtime"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
  target_compile_options(${target} PRIVATE
    -frounding-math -ftrapping-math -ffp-contract=off)
  target_link_libraries(${target} PRIVATE m)
  mdslc_target_enable_warnings(${target} WERROR)
endforeach()
target_compile_definitions(matcore_closed_host_test_v1 PRIVATE
  MDSLC_CLOSED_HOST_TESTING=1)

function(mdslc_closed_host_test target source library name)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE ${library})
  target_compile_options(${target} PRIVATE
    -frounding-math -ftrapping-math -ffp-contract=off)
  if(library STREQUAL "matcore_closed_host_test_v1")
    target_compile_definitions(${target} PRIVATE MDSLC_CLOSED_HOST_TESTING=1)
  endif()
  mdslc_target_enable_warnings(${target} WERROR)
  add_test(NAME ${name} COMMAND ${target})
endfunction()

mdslc_closed_host_test(matcore_closed_host_tests closed_host_v1_test.cpp
  matcore_closed_host_test_v1 runtime.closed_host.contract_v1)
mdslc_closed_host_test(matcore_experimental_region_result_tests
  ../experimental_region/result_test.cpp matcore_closed_host_v1
  runtime.experimental_region.owning_result)
mdslc_closed_host_test(matcore_experimental_region_result_reverse_tests
  ../experimental_region/result_test.cpp matcore_closed_host_v1
  runtime.experimental_region.header_order)
target_compile_definitions(matcore_experimental_region_result_reverse_tests PRIVATE
  MDSLC_RESULT_REVERSE_HEADERS=1)
add_library(matcore_experimental_region_result_producer OBJECT
  ../experimental_region/mixed_configuration_producer.cpp)
target_link_libraries(matcore_experimental_region_result_producer PRIVATE matcore_closed_host_v1)
mdslc_target_enable_warnings(matcore_experimental_region_result_producer WERROR)
find_package(Threads REQUIRED)
add_executable(matcore_experimental_region_mixed_consumer
  ../experimental_region/mixed_configuration_consumer.cpp
  $<TARGET_OBJECTS:matcore_experimental_region_result_producer>)
target_link_libraries(matcore_experimental_region_mixed_consumer PRIVATE matcore_closed_host_v1 Threads::Threads)
target_compile_definitions(matcore_experimental_region_mixed_consumer PRIVATE
  _GLIBCXX_DEBUG=1 _GLIBCXX_USE_CXX11_ABI=0)
mdslc_target_enable_warnings(matcore_experimental_region_mixed_consumer WERROR)
add_test(NAME runtime.experimental_region.mixed_configuration_ownership
  COMMAND matcore_experimental_region_mixed_consumer)
mdslc_closed_host_test(matcore_experimental_region_independent_retirement
  ../experimental_region/independent_retirement_test.cpp matcore_closed_host_test_v1
  runtime.experimental_region.independent_retirement)
# Deliberately tests a self-move expression, not an accidental implementation move.
target_compile_options(matcore_experimental_region_independent_retirement PRIVATE -Wno-self-move)
add_library(matcore_experimental_region_independent_producer OBJECT
  ../experimental_region/independent_result_producer.cpp)
target_link_libraries(matcore_experimental_region_independent_producer PRIVATE matcore_closed_host_v1)
mdslc_target_enable_warnings(matcore_experimental_region_independent_producer WERROR)
add_executable(matcore_experimental_region_independent_consumer
  ../experimental_region/independent_result_consumer.cpp
  $<TARGET_OBJECTS:matcore_experimental_region_independent_producer>)
target_link_libraries(matcore_experimental_region_independent_consumer PRIVATE matcore_closed_host_v1 Threads::Threads)
target_compile_definitions(matcore_experimental_region_independent_consumer PRIVATE
  _GLIBCXX_DEBUG=1 _GLIBCXX_USE_CXX11_ABI=0)
mdslc_target_enable_warnings(matcore_experimental_region_independent_consumer WERROR)
add_test(NAME runtime.experimental_region.independent_mixed_configuration
  COMMAND matcore_experimental_region_independent_consumer)
add_test(NAME frontend.experimental_region.ordinary_link_rejected
  COMMAND "${CMAKE_COMMAND}"
    "-DCXX=${CMAKE_CXX_COMPILER}"
    "-DCXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DLINK_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
    "-DINCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    "-DRUNTIME=$<TARGET_FILE:matcore_closed_host_v1>"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/experimental-ordinary"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/../experimental_region/ordinary_link_fails.cmake")
mdslc_closed_host_test(matcore_closed_host_production_tests
  closed_host_production_test.cpp matcore_closed_host_v1
  runtime.closed_host.production_v1)
mdslc_closed_host_test(matcore_closed_host_independent_tests
  closed_host_independent_test.cpp matcore_closed_host_test_v1
  runtime.closed_host.independent_v1)
mdslc_closed_host_test(matcore_closed_host_allocation_tests
  closed_host_allocation_test.cpp matcore_closed_host_v1
  runtime.closed_host.allocation_v1)
add_test(NAME runtime.closed_host.authority_v1
  COMMAND "${CMAKE_COMMAND}"
    "-DCXX=${CMAKE_CXX_COMPILER}"
    "-DNM=${CMAKE_NM}"
    "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
    "-DINCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../lib/runtime"
    "-DPUBLIC_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    "-DPRODUCTION_LIBRARY=$<TARGET_FILE:matcore_closed_host_v1>"
    "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/authority"
    "-DCXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DLINK_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/check_closed_host_authority.cmake")
