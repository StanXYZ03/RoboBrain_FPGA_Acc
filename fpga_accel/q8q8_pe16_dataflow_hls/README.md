# Q8×Q8 PE16 ping-pong DATAFLOW candidate

This is an independent branch of the board-validated `q8q8_pe16_hls`
kernel. It does not modify that frozen baseline and is not yet authorized for
bitstream generation or board programming.

For every stationary 64-column weight tile, two activation banks and two
accumulator banks form a coarse pipeline:

```text
load activation tile t+1
        || compute tile t
        || store output tile t-1
```

Prime and drain operations remain explicit. Decoder `M=1` therefore should
not gain from this change; the target is multi-row Prefill and Vision GEMM.

Run C simulation and synthesis:

```bash
./scripts/run_stage.sh csim
./scripts/run_stage.sh csynth
```

Gates are frozen in `artifacts/dataflow_stage_contract.json`. A bitstream may
only be considered after C simulation, HLS synthesis, directed RTL validation
and full-system routed timing pass.
