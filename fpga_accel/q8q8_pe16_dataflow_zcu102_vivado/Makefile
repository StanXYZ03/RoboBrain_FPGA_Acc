VIVADO ?= /home/xilinx/Vivado/2023.2/bin/vivado

.PHONY: validate synth impl bitstream fsbl

validate:
	./scripts/run_stage.sh validate

synth:
	./scripts/run_stage.sh synth

impl:
	./scripts/run_stage.sh impl

bitstream:
	./scripts/run_bitstream.sh

fsbl:
	./scripts/build_dataflow_fsbl.sh
