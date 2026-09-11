cmake_minimum_required(VERSION 3.24)

if(DEFINED QUALIFICATION_CHILD)
  include("${SOURCE_DIR}/cmake/MatcoreDSLGpuQualification.cmake")
  mdslc_check_rocdl_sanitizer_qualification("${ROCDL}" "${PROFILE_FLAGS}")
  message(STATUS "GPU sanitizer profile admitted")
  return()
endif()

function(check_profile name rocdl flags expected_status)
  execute_process(COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_DIR=${SOURCE_DIR}" -DQUALIFICATION_CHILD=ON
    "-DROCDL=${rocdl}" "-DPROFILE_FLAGS=${flags}"
    -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
  if(NOT "${status}" STREQUAL "${expected_status}")
    message(FATAL_ERROR "${name}: expected exit ${expected_status}, got ${status}: ${output}\n${error}")
  endif()
  if(expected_status EQUAL 0)
    if(NOT output MATCHES "GPU sanitizer profile admitted")
      message(FATAL_ERROR "${name}: missing explicit admission marker: ${output}\n${error}")
    endif()
  elseif(NOT error MATCHES "global host AddressSanitizer is unqualified" OR
         output MATCHES "GPU sanitizer profile admitted")
    message(FATAL_ERROR "${name}: did not fail at the exact qualification gate: ${output}\n${error}")
  endif()
endfunction()

# These run the actual configure gate, not a shadow regex. No compiler, SDK or
# vendor DSO is needed. Target-local mocked ASan is deliberately not a global
# host profile and remains exercised by the Release GPU regression targets.
check_profile(release ON "-O3 -DNDEBUG" 0)
check_profile(ubsan_only ON "-O1 -g -fsanitize=undefined" 0)
check_profile(ubsan_explicit_no_address ON "-fsanitize=undefined -fno-sanitize=address" 0)
check_profile(cpu_only_asan OFF "-fsanitize=address,undefined" 0)
check_profile(issuer_only_asan OFF "-fsanitize=address" 0)
check_profile(address_substring ON "-Daddress=1 -fsanitize=undefined -Dlabel=address" 0)
check_profile(hwaddress_is_not_address ON "-fsanitize=hwaddress" 0)
check_profile(no_sanitizer ON "" 0)
check_profile(address_only ON "-fsanitize=address" 1)
check_profile(address_first ON "-fsanitize=address,undefined" 1)
check_profile(address_last ON "-fsanitize=undefined,address" 1)
check_profile(address_middle ON "-fsanitize=undefined,address,float-divide-by-zero" 1)
check_profile(separate_flags ON "-fsanitize=undefined -fsanitize=address" 1)
check_profile(quoted_flag ON "-O1 \"-fsanitize=address,undefined\" -g" 1)
check_profile(whitespace ON "  -O1\t-fsanitize=address,undefined\n-g  " 1)
# Contradictory global profiles are not an authorized disable-ASan workaround.
check_profile(later_disable ON "-fsanitize=address -fno-sanitize=address" 1)
message(STATUS "GPU sanitizer qualification: 16/16 vendor-free controls passed")
