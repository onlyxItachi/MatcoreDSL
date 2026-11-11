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

# GPU tools have different stable markers from Clang. Recognize complete
# version lines, not an incidental version elsewhere in a multi-line banner.
# LLVM/LLD permit the upstream, Ubuntu and Debian spellings; horizontal
# whitespace/indentation and LF/CRLF line endings are accepted. NVIDIA's token
# must be on its "Cuda compilation tools" line in a ptxas banner. Metadata is
# allowed only after whitespace: punctuation and prerelease/build suffixes
# attached to the version remain part of the token and fail exact comparison.
# Multiple authoritative version lines are ambiguous and fail closed.
function(mdslc_extract_gpu_tool_version_token tool version_output result_variable)
  set(${result_variable} "" PARENT_SCOPE)
  if(tool STREQUAL "llc")
    set(marker "(^|[\r\n])[ \t]*((Ubuntu|Debian)[ \t]+)?LLVM[ \t]+version[ \t]+")
    set(token_group 4)
  elseif(tool STREQUAL "lld")
    set(marker "(^|[\r\n])[ \t]*((Ubuntu|Debian)[ \t]+)?LLD[ \t]+")
    set(token_group 4)
  elseif(tool STREQUAL "ptxas")
    if(NOT version_output MATCHES
       "^[ \t\r\n]*ptxas:[ \t]+NVIDIA[ \t]+\\(R\\)[ \t]+Ptx[ \t]+optimizing[ \t]+assembler([ \t]*[\r\n]|$)")
      return()
    endif()
    set(marker "(^|[\r\n])[ \t]*Cuda[ \t]+compilation[ \t]+tools,[ \t]+release[ \t]+[0-9]+\\.[0-9]+,[ \t]+V")
    set(token_group 2)
  else()
    message(FATAL_ERROR "Unknown GPU version-token tool '${tool}'")
  endif()

  string(REGEX MATCHALL "${marker}" markers "${version_output}")
  list(LENGTH markers marker_count)
  if(NOT marker_count EQUAL 1)
    return()
  endif()
  string(REGEX MATCH "${marker}([^ \t\r\n]+)" version_match "${version_output}")
  if(NOT version_match STREQUAL "")
    set(${result_variable} "${CMAKE_MATCH_${token_group}}" PARENT_SCOPE)
  endif()
endfunction()

# This is the configure-time admission gate shared by the single and fused GPU
# recipes. Version banner checks are qualification checks, not authentication of
# an executable capable of lying about its identity.
function(mdslc_require_gpu_tool_version tool required_version version_output)
  mdslc_extract_gpu_tool_version_token("${tool}" "${version_output}" reported_version)
  if(NOT reported_version STREQUAL required_version)
    if(reported_version STREQUAL "")
      set(reported_version "<missing or ambiguous supported version token>")
    endif()
    message(FATAL_ERROR
      "GPU ${tool} qualification requires exactly ${required_version}; "
      "reported '${reported_version}'")
  endif()
endfunction()
