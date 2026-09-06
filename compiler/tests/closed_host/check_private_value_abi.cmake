foreach(required CXX NM OBJCOPY SOURCE_DIR INCLUDE_DIR PUBLIC_INCLUDE_DIR RUNTIME OUTPUT_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing private Value ABI input: ${required}")
  endif()
endforeach()
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
separate_arguments(compile_flags NATIVE_COMMAND "${CXX_FLAGS}")
separate_arguments(link_flags NATIVE_COMMAND "${LINK_FLAGS}")
execute_process(COMMAND "${NM}" -C --defined-only "${RUNTIME}"
  RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR
   symbols MATCHES "closed_host_v1::(Value|Session|Observation|ValueStorage)[^A-Za-z0-9_]" OR
   NOT symbols MATCHES "SessionAbiV2" OR NOT symbols MATCHES "ValueAbiV2")
  message(FATAL_ERROR "Runtime private owning-record linkage is not version-separated: ${symbols} ${error}")
endif()
execute_process(COMMAND "${CXX}" ${compile_flags} -std=c++20
  -I "${INCLUDE_DIR}" -I "${PUBLIC_INCLUDE_DIR}"
  "${SOURCE_DIR}/private_value_link_control.cpp" "${RUNTIME}" ${link_flags}
  -o "${OUTPUT_DIR}/positive" RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Matching-header/runtime positive link failed: ${error}")
endif()
execute_process(COMMAND "${OUTPUT_DIR}/positive" RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Matching-header/runtime positive execution failed: ${error}")
endif()
# Rename only the revision symbol in an independent test artifact. This retains
# all real runtime definitions and sanitizer dependencies; a failed link cannot
# be explained by an empty fake library or missing ordinary runtime functions.
execute_process(COMMAND "${OBJCOPY}"
  --redefine-sym matcore_closed_host_private_value_abi_v2=matcore_closed_host_private_value_abi_v1
  "${RUNTIME}" "${OUTPUT_DIR}/incompatible.a"
  RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Could not construct incompatible revision artifact: ${error}")
endif()
execute_process(COMMAND "${NM}" --defined-only "${OUTPUT_DIR}/incompatible.a"
  RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR symbols MATCHES "matcore_closed_host_private_value_abi_v2" OR
   NOT symbols MATCHES "matcore_closed_host_private_value_abi_v1")
  message(FATAL_ERROR "Incompatible artifact revision was not established: ${symbols} ${error}")
endif()
execute_process(COMMAND "${CXX}" ${compile_flags} -std=c++20
  -I "${INCLUDE_DIR}" -I "${PUBLIC_INCLUDE_DIR}"
  "${SOURCE_DIR}/private_value_link_control.cpp" "${OUTPUT_DIR}/incompatible.a" ${link_flags}
  -o "${OUTPUT_DIR}/negative" RESULT_VARIABLE result ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "matcore_closed_host_private_value_abi_v2" OR
   NOT error MATCHES "undefined reference|undefined symbol")
  message(FATAL_ERROR "Header/runtime revision mismatch did not fail specifically at link gate: ${error}")
endif()
execute_process(COMMAND "${CXX}" ${compile_flags} -std=c++20 -O0
  -c "${SOURCE_DIR}/private_value_stale_constructor.cpp"
  -o "${OUTPUT_DIR}/stale-constructor.o" RESULT_VARIABLE result ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Could not compile stale inline-constructor adversary: ${error}")
endif()
execute_process(COMMAND "${NM}" --defined-only "${OUTPUT_DIR}/stale-constructor.o"
  RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT symbols MATCHES "W _ZN7matcore5mdslc7runtime14closed_host_v17SessionC2Ev")
  message(FATAL_ERROR "Stale weak constructor was not actually emitted: ${symbols} ${error}")
endif()
foreach(optimization 0 2)
  execute_process(COMMAND "${CXX}" ${compile_flags} -std=c++20 "-O${optimization}"
    -I "${INCLUDE_DIR}" -I "${PUBLIC_INCLUDE_DIR}"
    -c "${SOURCE_DIR}/private_value_link_control.cpp" -o "${OUTPUT_DIR}/consumer.o"
    RESULT_VARIABLE result ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Actual new-header consumer compile failed: ${error}")
  endif()
  foreach(order stale_first stale_last)
    if(order STREQUAL "stale_first")
      set(objects "${OUTPUT_DIR}/stale-constructor.o" "${OUTPUT_DIR}/consumer.o")
    else()
      set(objects "${OUTPUT_DIR}/consumer.o" "${OUTPUT_DIR}/stale-constructor.o")
    endif()
    execute_process(COMMAND "${CXX}" ${compile_flags} ${objects} "${RUNTIME}" ${link_flags}
      -o "${OUTPUT_DIR}/mixed-positive" RESULT_VARIABLE result ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Compatible mixed-object positive link failed: ${error}")
    endif()
    execute_process(COMMAND "${OUTPUT_DIR}/mixed-positive" RESULT_VARIABLE result ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Compatible mixed-object execution failed: ${error}")
    endif()
    execute_process(COMMAND "${CXX}" ${compile_flags} ${objects} "${OUTPUT_DIR}/incompatible.a" ${link_flags}
      -o "${OUTPUT_DIR}/mixed-negative" RESULT_VARIABLE result ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT error MATCHES "matcore_closed_host_private_value_abi_v2" OR
       NOT error MATCHES "undefined reference|undefined symbol")
      message(FATAL_ERROR "Stale inline constructor bypassed private ABI revision at O${optimization}/${order}: ${error}")
    endif()
  endforeach()
endforeach()
message(STATUS "Private Value ABI: matching artifact executes; mismatch rejects including stale COMDAT first/last at O0/O2")
