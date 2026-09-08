# Test-only manual linking: a configured provider is a file, never a -l lookup.
function(matcore_installed_provider_link_flags enabled configured_provider output_flags output_provider)
  set(${output_flags} "" PARENT_SCOPE)
  set(${output_provider} "" PARENT_SCOPE)
  if(NOT enabled)
    return()
  endif()
  if(configured_provider STREQUAL "" OR NOT IS_ABSOLUTE "${configured_provider}" OR
      NOT EXISTS "${configured_provider}" OR IS_DIRECTORY "${configured_provider}")
    message(FATAL_ERROR
      "MDSLC_INSTALL_PROVIDER_FILE: provider-enabled consumers require an existing absolute provider file")
  endif()
  file(REAL_PATH "${configured_provider}" canonical_provider)
  get_filename_component(provider_directory "${canonical_provider}" DIRECTORY)
  set(${output_provider} "${canonical_provider}" PARENT_SCOPE)
  # Executable RUNPATH is not used for Runtime's transitive dependencies. Keep
  # the provider direct under --as-needed, then restore the caller's link state.
  set(${output_flags} -Xlinker --push-state -Xlinker --no-as-needed
    "${canonical_provider}" -Xlinker --pop-state
    -Xlinker -rpath -Xlinker "${provider_directory}" PARENT_SCOPE)
endfunction()

# check_private_value_abi.cmake accepts a native-command string and parses it
# with separate_arguments(NATIVE_COMMAND). Quote each complete argument, not a
# comma-joined -Wl payload, so paths survive that additional serialization edge.
function(matcore_serialize_installed_link_flags output)
  set(serialized "")
  foreach(argument IN LISTS ARGN)
    string(REPLACE "\\" "\\\\" escaped "${argument}")
    string(REPLACE "\"" "\\\"" escaped "${escaped}")
    if(NOT serialized STREQUAL "")
      string(APPEND serialized " ")
    endif()
    string(APPEND serialized "\"${escaped}\"")
  endforeach()
  set(${output} "${serialized}" PARENT_SCOPE)
endfunction()
