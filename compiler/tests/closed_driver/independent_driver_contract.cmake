foreach(required IN ITEMS DRIVER CLANG SOURCE_ROOT OUTPUT_ROOT)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing independent driver test input ${required}")
  endif()
endforeach()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(root "${OUTPUT_ROOT}/independent-driver-${suffix}")
file(MAKE_DIRECTORY "${root}/tools")
set(fixtures "${SOURCE_ROOT}/compiler/tests/closed_driver")
configure_file("${fixtures}/independent_minimal.mdsl" "${root}/source.mdsl" COPYONLY)
configure_file("${fixtures}/independent_weak_runtime.mdsl" "${root}/weak.mdsl" COPYONLY)
execute_process(COMMAND "${CLANG}" --no-default-config -std=c++20
  "${fixtures}/independent_linker.cpp" -o "${root}/tools/ld"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Cannot build independent fake linker: ${output}\n${error}")
endif()

# Establish that this Clang would select our fake linker without the driver's
# explicit linker binding. This is a real control, not an assumed search rule.
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH=${root}/tools:/usr/bin:/bin"
  "${CLANG}" --no-default-config "-###" -x c++ /dev/null -o "${root}/unused"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
string(FIND "${error}" "${root}/tools/ld" selected)
if(NOT status EQUAL 0 OR selected EQUAL -1)
  message(FATAL_ERROR "Fake-linker PATH negative control was not selected: ${output}\n${error}")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH=${root}/tools:/usr/bin:/bin"
  "${DRIVER}" "${root}/source.mdsl" --region pipeline -o "${root}/program"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Pinned-linker compilation failed: ${output}\n${error}")
endif()
file(READ "${root}/program" magic LIMIT 4 HEX)
if(NOT magic STREQUAL "7f454c46")
  message(FATAL_ERROR "Driver published non-ELF fake-linker output: ${magic}")
endif()
execute_process(COMMAND "${root}/program"
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "Pinned-linker generated program failed: ${output}\n${error}")
endif()

# Failure must not issue an object either: compile-only is not an escape from
# source/runtime ownership. Never execute the rejected hostile program here.
foreach(mode IN ITEMS executable object)
  set(options)
  if(mode STREQUAL "object")
    list(APPEND options -c)
  endif()
  foreach(spelling IN ITEMS assembler named)
  set(definition)
  if(spelling STREQUAL "named")
    set(definition -- -DMDSLC_INDEPENDENT_NAMED_RUNTIME_DEFINITION=1)
  endif()
  execute_process(COMMAND "${DRIVER}" "${root}/weak.mdsl" --region pipeline
    ${options} -o "${root}/weak-${mode}-${spelling}" ${definition}
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(status EQUAL 0 OR EXISTS "${root}/weak-${mode}-${spelling}")
    message(FATAL_ERROR "Source weak-runtime ownership violation published ${mode}/${spelling}: ${output}\n${error}")
  endif()
  if(NOT error MATCHES "symbol|ownership|assembler|reserved")
    message(FATAL_ERROR "Weak-runtime case failed for an unrelated reason: ${error}")
  endif()
  endforeach()
endforeach()
message(STATUS "Independent driver: real PATH-selected-linker control bypassed; real generated math and observation pass; weak-runtime definition rejects executable and object publication")
