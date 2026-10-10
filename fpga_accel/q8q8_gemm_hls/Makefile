VITIS_HLS ?= /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls

.PHONY: csim csynth cosim all clean

csim:
	$(VITIS_HLS) -f scripts/run_hls.tcl -tclargs csim

csynth:
	$(VITIS_HLS) -f scripts/run_hls.tcl -tclargs csynth

cosim:
	$(VITIS_HLS) -f scripts/run_hls.tcl -tclargs cosim

all:
	$(VITIS_HLS) -f scripts/run_hls.tcl -tclargs all

clean:
	$(RM) -r build vitis_hls.log vitis_hls.jou

