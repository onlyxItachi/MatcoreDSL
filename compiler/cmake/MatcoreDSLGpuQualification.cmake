# Build-profile qualification only: never discover or load a vendor runtime.
function(mdslc_check_rocdl_sanitizer_qualification rocdl_enabled host_flags)
  if(NOT rocdl_enabled)
    return()
  endif()
  separate_arguments(profile_flags UNIX_COMMAND "${host_flags}")
  foreach(flag IN LISTS profile_flags)
    if(flag MATCHES "^-fsanitize=(.+)$")
      string(REPLACE "," ";" sanitizers "${CMAKE_MATCH_1}")
      if("address" IN_LIST sanitizers)
        message(FATAL_ERROR
          "Experimental ROCDL with global host AddressSanitizer is unqualified: "
          "HIP 7.2.70201 exhibited alternate-signal-stack teardown failure. "
          "Use MDSLC_ENABLE_EXPERIMENTAL_ROCDL=OFF for host ASan builds, or a "
          "separate Release GPU build without global -fsanitize=address. "
          "Target-local mocked-driver ASan tests remain available; this does "
          "not qualify real HIP under ASan.")
      endif()
    endif()
  endforeach()
endfunction()
