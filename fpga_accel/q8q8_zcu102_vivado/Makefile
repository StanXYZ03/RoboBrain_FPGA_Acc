VIVADO ?= /home/xilinx/Vivado/2023.2/bin/vivado

.PHONY: validate synth impl bitstream fsbl

validate:
	$(VIVADO) -mode batch -source scripts/build_system.tcl -tclargs validate

synth:
	$(VIVADO) -mode batch -source scripts/build_system.tcl -tclargs synth

impl:
	$(VIVADO) -mode batch -source scripts/build_system.tcl -tclargs impl

bitstream:
	$(VIVADO) -mode batch -source scripts/build_system.tcl -tclargs bitstream

fsbl:
	@test -r artifacts/q8q8_zcu102.xsa || \
		{ echo "missing artifacts/q8q8_zcu102.xsa; run make bitstream first" >&2; exit 1; }
	sh scripts/build_accel_fsbl.sh
