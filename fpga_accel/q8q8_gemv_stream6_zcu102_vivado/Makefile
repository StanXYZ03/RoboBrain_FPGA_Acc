VIVADO ?= /home/xilinx/Vivado/2023.2/bin/vivado

.PHONY: validate synth impl

validate:
	VIVADO_BIN=$(VIVADO) bash scripts/run_stage.sh validate

synth:
	VIVADO_BIN=$(VIVADO) bash scripts/run_stage.sh synth

impl:
	VIVADO_BIN=$(VIVADO) bash scripts/run_stage.sh impl
