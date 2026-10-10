VITIS_HLS ?= /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls

.PHONY: csim csynth cosim cosim-native export-impl all clean

csim:
	$(VITIS_HLS) -f scripts/run_hls.tcl -tclargs csim

csynth:
	$(VITIS_HLS) -f scripts/run_hls.tcl -tclargs csynth

cosim:
	bash scripts/run_cosim_xsim_2023_2.sh

cosim-native:
	$(VITIS_HLS) -f scripts/run_hls.tcl -tclargs cosim

export-impl:
	$(VITIS_HLS) -f scripts/run_hls.tcl -tclargs export_impl

all:
	bash scripts/run_cosim_xsim_2023_2.sh

clean:
	$(RM) -r build vitis_hls.log vitis_hls.jou
