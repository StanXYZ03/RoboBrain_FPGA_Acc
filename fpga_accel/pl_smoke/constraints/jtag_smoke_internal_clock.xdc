# STARTUPE3 CFGMCLK is the UltraScale+ internal configuration oscillator.
# Its nominal simulation period is 20 ns (50 MHz). Constrain the source pin so
# that both the smoke-test logic and the VIO/debug hub are timed.
create_clock -name cfgmclk -period 20.000 -waveform {0.000 10.000} \
    [get_pins -hier -filter {REF_PIN_NAME == CFGMCLK}]
