VITIS_HLS ?= /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls

.PHONY: csim csynth cosim cosim-native export clean

csim:
	$(VITIS_HLS) -f scripts/run_hls.tcl csim

csynth:
	$(VITIS_HLS) -f scripts/run_hls.tcl csynth

cosim:
	bash scripts/run_cosim_xsim_2023_2.sh

cosim-native:
	$(VITIS_HLS) -f scripts/run_hls.tcl cosim

export:
	mkdir -p artifacts
	$(VITIS_HLS) -f scripts/run_hls.tcl export_impl

clean:
	$(RM) -r build .Xil vitis_hls.log artifacts
