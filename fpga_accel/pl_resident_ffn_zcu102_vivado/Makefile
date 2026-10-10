VIVADO ?= /home/xilinx/Vivado/2023.2/bin/vivado

.PHONY: validate synth impl

validate:
	VIVADO=$(VIVADO) ./scripts/run_stage.sh validate

synth:
	VIVADO=$(VIVADO) ./scripts/run_stage.sh synth

impl:
	VIVADO=$(VIVADO) ./scripts/run_stage.sh impl
