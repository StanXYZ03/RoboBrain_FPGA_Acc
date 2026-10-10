if {[info exists ::env(Q8Q8_XSIM_NO_SIGNAL_HANDLERS)] &&
        $::env(Q8Q8_XSIM_NO_SIGNAL_HANDLERS) eq "1"} {
    xsim {q8q8_gemm} -nosignalhandlers -tclbatch {q8q8_gemm.tcl}
} else {
    xsim {q8q8_gemm} -tclbatch {q8q8_gemm.tcl}
}
exit
