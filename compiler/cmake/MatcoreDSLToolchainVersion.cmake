include_guard(GLOBAL)

# Extract the version token that immediately follows Clang's stable
# "clang version" marker.  Callers must compare the returned token with
# STREQUAL: substring matches are not a toolchain admission boundary.
function(mdslc_extract_clang_version_token version_output result_variable)
  string(REGEX MATCH "clang version ([^ \t\r\n(]+)"
    mdslc_clang_version_match "${version_output}")
  if(mdslc_clang_version_match STREQUAL "")
    set(${result_variable} "" PARENT_SCOPE)
  else()
    set(${result_variable} "${CMAKE_MATCH_1}" PARENT_SCOPE)
  endif()
endfunction()
