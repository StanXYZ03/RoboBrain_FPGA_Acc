VITIS_HLS ?= /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls

.PHONY: csim csynth cosim summarize export export_impl summarize_impl

csim:
	VITIS_HLS_BIN=$(VITIS_HLS) bash scripts/run_stage.sh csim

csynth:
	VITIS_HLS_BIN=$(VITIS_HLS) bash scripts/run_stage.sh csynth

cosim:
	VITIS_HLS_BIN=$(VITIS_HLS) bash scripts/run_stage.sh cosim

summarize:
	python3 scripts/summarize_stream6.py

export:
	VITIS_HLS_BIN=$(VITIS_HLS) bash scripts/run_stage.sh export

export_impl:
	VITIS_HLS_BIN=$(VITIS_HLS) bash scripts/run_stage.sh export_impl

summarize_impl:
	python3 scripts/summarize_impl.py
