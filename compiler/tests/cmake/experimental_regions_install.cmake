# Build/install contract only: these ordinary C++ consumers exercise the private
# adapter and issued leaf, not a new source compiler or public execution syntax.
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
      "${include}/matcore/detail" "${private}")
    if(EXISTS "${path}")
      message(FATAL_ERROR "Feature-OFF package leaked experimental artifact ${path}")
    endif()
  endforeach()
  message(STATUS "Feature-OFF install preserves legacy headers and excludes experimental artifacts")
  return()
endif()
set(archive "${private}/libmatcore_closed_candidates_production_v1.a")
foreach(path IN ITEMS "${include}/matcore/region.h"
    "${include}/matcore/detail/region_storage.h"
    "${private}/include/closed_host_v1.h" "${archive}")
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Missing feature-ON artifact ${path}")
  endif()
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
    "-L${lib}" -lmatcore_runtime -lm -pthread "-Wl,-rpath,${lib}"
    ${link_flags} -o "${executable}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed ${test} consumer failed without LLVM/MLIR: ${output}\n${error}")
  endif()
  execute_process(COMMAND "${executable}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Installed ${test} consumer failed: ${output}\n${error}")
  endif()
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
    "${object}" "${archive}" "-L${lib}" -lmatcore_runtime -lm -pthread
    "-Wl,-rpath,${lib}" ${link_flags} -o "${executable}"
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
set(abi_link_flags "${LINK_FLAGS} -L\"${lib}\" -lmatcore_runtime -lm -pthread -Wl,-rpath,\"${lib}\"")
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
    "${archive}" "-L${lib}" -lmatcore_runtime -lm -pthread
    "-Wl,-rpath,${lib}" ${link_flags} -o "${executable}"
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
message(STATUS "Feature-ON install: one issued leaf, no injection exports, no public LLVM/MLIR dependency")
