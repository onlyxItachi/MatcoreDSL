cmake_minimum_required(VERSION 3.24)

# Existing ordinary C++ consumers and the opt-in installed source driver have
# separate oracles. An archive consumer alone does not authenticate source.
include("${CMAKE_CURRENT_LIST_DIR}/experimental_regions_consumer_identity.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/experimental_regions_consumer_link.cmake")
foreach(required IN ITEMS BINARY_DIR SOURCE_DIR CXX INSTALL_LIBDIR INSTALL_INCLUDEDIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing package-test input ${required}")
  endif()
endforeach()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(prefix "${BINARY_DIR}/experimental-region-package-${suffix}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BINARY_DIR}" --prefix "${prefix}"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Package install failed: ${output}\n${error}")
endif()
set(include "${prefix}/${INSTALL_INCLUDEDIR}")
set(lib "${prefix}/${INSTALL_LIBDIR}")
set(private "${lib}/mdslc/experimental-regions")
if(NOT DEFINED INSTALL_BINDIR)
  set(INSTALL_BINDIR bin)
endif()
set(driver "${prefix}/${INSTALL_BINDIR}/mdslc-region")
foreach(header IN ITEMS mdsl.h runtime_c.h)
  if(NOT EXISTS "${include}/matcore/${header}")
    message(FATAL_ERROR "Missing legacy installed header ${header}")
  endif()
endforeach()
file(READ "${lib}/cmake/MatcoreDSL/MatcoreDSLTargets.cmake" exports)
if(exports MATCHES "MLIR|LLVM|matcore_closed_|matcore_cpu_gemm_candidate")
  message(FATAL_ERROR "Compiler-private dependency leaked into consumer target export")
endif()
if(NOT ENABLED)
  foreach(path IN ITEMS "${include}/matcore/region.h"
      "${include}/matcore/detail" "${private}" "${driver}")
    if(EXISTS "${path}")
      message(FATAL_ERROR "Feature-OFF package leaked experimental artifact ${path}")
    endif()
  endforeach()
  message(STATUS "Feature-OFF install preserves legacy headers and excludes experimental artifacts")
  return()
endif()
matcore_installed_provider_link_flags("${HAS_OPENBLAS}" "${PROVIDER}"
  provider_link_flags canonical_provider)
# The driver is bound to the configured spelling, whose directory can differ
# from a symlink's physical target used by the manual consumers.
set(driver_provider)
if(HAS_OPENBLAS)
  set(driver_provider "${PROVIDER}")
endif()
set(archive "${private}/libmatcore_closed_candidates_production_v1.a")
set(candidates "${private}/libmatcore_closed_candidates_isolated_v1.so")
foreach(path IN ITEMS "${include}/matcore/region.h"
    "${include}/matcore/detail/region_storage.h"
    "${private}/include/closed_host_v1.h" "${archive}" "${candidates}" "${driver}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Missing feature-ON artifact ${path}")
  endif()
endforeach()
# Run the small falsification controls in the existing feature-ON CTest lane;
# they must not depend on a separately registered test or generated test binary.
set(link_regression_scratch "${prefix}/consumer-link-regression")
file(MAKE_DIRECTORY "${link_regression_scratch}")
foreach(regression IN ITEMS experimental_regions_install_identity_test experimental_regions_consumer_link_test)
  execute_process(COMMAND "${CMAKE_COMMAND}" "-DSCRATCH=${link_regression_scratch}"
    -P "${CMAKE_CURRENT_LIST_DIR}/${regression}.cmake"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed-consumer regression ${regression} failed: ${output}\n${error}")
  endif()
  message(STATUS "${output}")
endforeach()
execute_process(COMMAND "${NM}" --defined-only --extern-only "${archive}"
  RESULT_VARIABLE status OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Cannot inspect installed registry archive: ${error}")
endif()
string(REGEX MATCHALL "[ \t]T[ \t]+_mlir_ciface___matcore_strict_gemm_f32_v1[\r\n]"
  definitions "${symbols}\n")
list(LENGTH definitions definition_count)
if(NOT definition_count EQUAL 1 OR symbols MATCHES
    "configureForTesting|allocationAttemptsForTesting|[ \t]T[ \t]+(openblas_|cblas_)")
  message(FATAL_ERROR "Installed registry violates leaf, test-authority or provider-owner contract")
endif()
separate_arguments(compile_flags NATIVE_COMMAND "${CXX_FLAGS}")
separate_arguments(link_flags NATIVE_COMMAND "${LINK_FLAGS}")
set(consumer_link_flags "-L${lib}" -lmatcore_runtime ${provider_link_flags}
  -lm -pthread -Xlinker -rpath -Xlinker "${lib}" ${link_flags})
set(provider_flag)
if(HAS_OPENBLAS)
  set(provider_flag -DEXPECT_OPENBLAS)
endif()
foreach(test IN ITEMS result candidates private_value)
  if(test STREQUAL "result")
    set(source "${SOURCE_DIR}/tests/experimental_region/result_test.cpp")
  elseif(test STREQUAL "candidates")
    set(source "${SOURCE_DIR}/tests/closed_candidates/candidate_test.cpp")
  else()
    set(source "${SOURCE_DIR}/tests/closed_host/private_value_independent_test.cpp")
  endif()
  set(executable "${prefix}/${test}")
  execute_process(COMMAND "${CXX}" -std=c++20 ${compile_flags}
    -ffp-contract=off -frounding-math ${provider_flag}
    "-I${include}" "-I${private}/include" "${source}" "${archive}"
    ${consumer_link_flags} -o "${executable}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed ${test} consumer failed without LLVM/MLIR: ${output}\n${error}")
  endif()
  execute_process(COMMAND "${executable}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed ${test} consumer failed: ${output}\n${error}")
  endif()
  matcore_expect_installed_consumer_output("${test}" "${output}")
  message(STATUS "Installed ${test}: ${output}")
endforeach()
# Exercise actual installed owning handles across differing host STL settings.
# Only the consumer changes configuration; both use installed header bytes and
# the same installed production archive. No compiler-private build include leaks.
foreach(kind IN ITEMS result value)
  if(kind STREQUAL "result")
    set(test_dir "${SOURCE_DIR}/tests/experimental_region")
    set(stem mixed_configuration)
  else()
    set(test_dir "${SOURCE_DIR}/tests/closed_host")
    set(stem private_value)
  endif()
  set(object "${prefix}/${kind}-producer.o")
  execute_process(COMMAND "${CXX}" -std=c++20 ${compile_flags}
    "-I${include}" "-I${private}/include" -c "${test_dir}/${stem}_producer.cpp"
    -o "${object}" RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed ${kind} producer failed: ${output}\n${error}")
  endif()
  set(executable "${prefix}/${kind}-mixed")
  execute_process(COMMAND "${CXX}" -std=c++20 ${compile_flags}
    -D_GLIBCXX_DEBUG=1 -D_GLIBCXX_USE_CXX11_ABI=0
    "-I${include}" "-I${private}/include" "${test_dir}/${stem}_consumer.cpp"
    "${object}" "${archive}" ${consumer_link_flags} -o "${executable}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed mixed ${kind} consumer failed to link: ${output}\n${error}")
  endif()
  execute_process(COMMAND "${executable}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed mixed ${kind} ownership failed: ${output}\n${error}")
  endif()
  message(STATUS "Installed mixed ${kind} ownership: PASS ${output}")
endforeach()
matcore_serialize_installed_link_flags(abi_link_flags ${consumer_link_flags})
execute_process(COMMAND "${CMAKE_COMMAND}"
  "-DCXX=${CXX}" "-DNM=${NM}" "-DOBJCOPY=${OBJCOPY}"
  "-DSOURCE_DIR=${SOURCE_DIR}/tests/closed_host"
  "-DINCLUDE_DIR=${private}/include" "-DPUBLIC_INCLUDE_DIR=${include}"
  "-DRUNTIME=${archive}" "-DOUTPUT_DIR=${prefix}/abi-control"
  "-DCXX_FLAGS=${CXX_FLAGS}" "-DLINK_FLAGS=${abi_link_flags}"
  -P "${SOURCE_DIR}/tests/closed_host/check_private_value_abi.cmake"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Installed private Value revision gate failed: ${output}\n${error}")
endif()
message(STATUS "${output}")
if(CXX_FLAGS MATCHES "fsanitize=.*address")
  set(executable "${prefix}/installed-leaf-asan-control")
  execute_process(COMMAND "${CXX}" -std=c++20 ${compile_flags}
    -ffp-contract=off "${SOURCE_DIR}/tests/generated_cpu/execution_test.cpp"
    "${archive}" ${consumer_link_flags} -o "${executable}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed leaf ASan control did not link: ${output}\n${error}")
  endif()
  execute_process(COMMAND "${CMAKE_COMMAND}" "-DEXECUTABLE=${executable}"
    -P "${SOURCE_DIR}/tests/generated_cpu/expect_asan.cmake"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed leaf was not actually instrumented: ${output}\n${error}")
  endif()
  message(STATUS "${output}")
endif()
set(driver_sanitized OFF)
if(CXX_FLAGS MATCHES "fsanitize=.*address")
  set(driver_sanitized ON)
endif()
execute_process(COMMAND "${CMAKE_COMMAND}"
  "-DDRIVER=${driver}"
  "-DSOURCE=${SOURCE_DIR}/examples/experimental/two_gemm.mdsl"
  "-DOUTPUT_ROOT=${prefix}" "-DSANITIZED=${driver_sanitized}"
  "-DPROVIDER=${driver_provider}"
  -P "${SOURCE_DIR}/tests/closed_driver/driver_contract.cmake"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Installed source compiler contract failed: ${output}\n${error}")
endif()
message(STATUS "${output}")
message(STATUS "Feature-ON install: authenticated source-to-executable driver and isolated candidate DSO; separate archive consumers; no public LLVM/MLIR dependency")
