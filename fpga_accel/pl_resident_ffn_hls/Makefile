VITIS_HLS ?= /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls

.PHONY: csim csynth export clean

csim:
	VITIS_HLS_BIN=$(VITIS_HLS) bash scripts/run_stage.sh csim

csynth:
	VITIS_HLS_BIN=$(VITIS_HLS) bash scripts/run_stage.sh csynth

export:
	VITIS_HLS_BIN=$(VITIS_HLS) bash scripts/run_stage.sh export_impl

clean:
	$(RM) -r build .Xil vitis_hls.log vitis_hls.jou
