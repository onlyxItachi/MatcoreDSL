# This check is keyed by the requested label, independently of the branch that
# selected/compiled a source file. A successful wrong executable must not pass.
function(matcore_expect_installed_consumer_output consumer actual_output)
  set(expected_result "^Experimental owning Result: [1-9][0-9]* checks, 0 failures$")
  set(expected_candidates "^[1-9][0-9]* candidate checks, 0 failures$")
  set(expected_private_value "^Independent private Value: [1-9][0-9]* checks, 32000 ownership cycles, 0 failures$")
  set(expected_regex "${expected_${consumer}}")
  string(STRIP "${actual_output}" actual_summary)
  if(expected_regex STREQUAL "" OR NOT actual_summary MATCHES "${expected_regex}")
    message(FATAL_ERROR
      "Installed ${consumer} output identity mismatch: expected '${expected_regex}', got '${actual_summary}'")
  endif()
endfunction()
