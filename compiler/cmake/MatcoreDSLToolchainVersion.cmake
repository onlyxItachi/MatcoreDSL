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

# Parse only a supported leading header. A quoted dependency in the banner
# tail cannot supply a missing version. Reject a second version record rather
# than choosing between them. Keep NVIDIA's independent release field so the
# gate can diagnose an inconsistent pair without discarding the actual token.
function(mdslc_extract_gpu_tool_version_token tool version_output result_variable release_variable)
  set(${result_variable} "" PARENT_SCOPE)
  set(${release_variable} "" PARENT_SCOPE)
  string(REPLACE "\r\n" "\n" banner "${version_output}")
  if(banner MATCHES "\r")
    return()
  endif()
  set(next_line "[ \t]*\n[ \t\n]*")
  if(tool STREQUAL "llc")
    set(marker "((Ubuntu|Debian)[ \t]+)?LLVM[ \t]+version")
    set(header "(LLVM[ \t]+\\(http://llvm\\.org/\\):${next_line})?${marker}[ \t]+")
    set(token_group 4)
  elseif(tool STREQUAL "lld")
    set(marker "((Ubuntu|Debian)[ \t]+)?LLD")
    set(header "${marker}[ \t]+")
    set(token_group 3)
  elseif(tool STREQUAL "ptxas")
    set(marker "Cuda[ \t]+compilation[ \t]+tools,")
    set(header "ptxas:[ \t]+NVIDIA[ \t]+\\(R\\)[ \t]+Ptx[ \t]+optimizing[ \t]+assembler${next_line}")
    string(APPEND header
      "(Copyright[ \t]+\\(c\\)[ \t]+[0-9]+(-[0-9]+)?[ \t]+NVIDIA[ \t]+Corporation${next_line})?"
      "(Built[ \t]+on[ \t]+[^\n \t]+${next_line})?"
      "${marker}[ \t]+release[ \t]+([0-9]+\\.[0-9]+),[ \t]+V")
    set(token_group 5)
  else()
    message(FATAL_ERROR "Unknown GPU version-token tool '${tool}'")
  endif()

  if(NOT banner MATCHES "^[ \t\n]*${header}([^ \t\n]+)")
    return()
  endif()
  set(token "${CMAKE_MATCH_${token_group}}")
  if(tool STREQUAL "ptxas")
    set(release "${CMAKE_MATCH_4}")
  endif()
  string(LENGTH "${CMAKE_MATCH_0}" header_length)
  string(SUBSTRING "${banner}" ${header_length} -1 tail)
  if(tail MATCHES "\n[ \t]*${marker}([ \t\n]|$)")
    return()
  endif()
  set(${result_variable} "${token}" PARENT_SCOPE)
  if(tool STREQUAL "ptxas")
    set(${release_variable} "${release}" PARENT_SCOPE)
  endif()
endfunction()

# This is the configure-time admission gate shared by the single and fused GPU
# recipes. Version banner checks are qualification checks, not authentication of
# an executable capable of lying about its identity.
function(mdslc_require_gpu_tool_version tool required_version version_output)
  mdslc_extract_gpu_tool_version_token("${tool}" "${version_output}" reported_version release)
  set(detail "")
  if(tool STREQUAL "ptxas" AND NOT reported_version STREQUAL "")
    string(FIND "${reported_version}" "${release}." release_position)
    if(NOT release_position EQUAL 0)
      set(detail " with conflicting release '${release}'")
    endif()
  endif()
  if(NOT reported_version STREQUAL required_version OR NOT detail STREQUAL "")
    if(reported_version STREQUAL "")
      set(reported_version "<missing or ambiguous supported version token>")
    endif()
    message(FATAL_ERROR
      "GPU ${tool} qualification requires exactly ${required_version}; "
      "reported '${reported_version}'${detail}")
  endif()
endfunction()
