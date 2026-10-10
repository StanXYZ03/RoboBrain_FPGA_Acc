# Q8_0 x Q8_0 weight-stationary HLS v2

This experimental v2 keeps one complete `TN=64` by `K` weight tile in BRAM
and reuses it across every `TM=16` activation tile. Weight and activation
loads are row-major so HLS can infer contiguous AXI bursts. It preserves the
v1 top-level function signature and GGML Q8_0 arithmetic.

Validation order:

```sh
make csim
make csynth
make cosim
make export
```

No target programs the board.
