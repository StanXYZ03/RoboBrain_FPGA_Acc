VIVADO ?= /home/xilinx/Vivado/2023.2/bin/vivado

.PHONY: validate synth impl bitstream

validate:
	$(VIVADO) -mode batch -source scripts/build_system.tcl -tclargs validate

synth:
	$(VIVADO) -mode batch -source scripts/build_system.tcl -tclargs synth

impl:
	$(VIVADO) -mode batch -source scripts/build_system.tcl -tclargs impl

bitstream:
	$(VIVADO) -mode batch -source scripts/build_system.tcl -tclargs bitstream
