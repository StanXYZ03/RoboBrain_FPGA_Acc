# Runtime FFN activation capture

The profiling-only GGML CPU backend can capture one named, contiguous F32 node
after all CPU worker threads finish it. The hook is disabled unless both are set:

```bash
GGML_CPU_CAPTURE_NAME=ffn_inp-0
GGML_CPU_CAPTURE_PATH=/new/path/ffn_inp_0.bin
```

The writer uses exclusive-create mode and never overwrites an existing capture.
The binary stores a fixed little-endian header followed by contiguous F32 data.
It rejects non-F32/non-contiguous matches. This is a CPU diagnostic path only;
it does not load CUDA, access ZCU102, or change model weights.

Validate the parser:

```bash
make -C fpga_accel/runtime_activation_capture test
```

After capturing `ffn_inp-0`, compare the real runtime rows against the proposed
Q8/PWL PL arithmetic path with real layer-0 weights:

```bash
make -C fpga_accel/runtime_activation_capture evaluate \
  PYTHON=/path/to/python MODEL=/path/to/model.gguf \
  CAPTURE=/path/to/ffn_inp_0.bin OUTPUT_DIR=/new/artifact/directory
```

The evaluator refuses to overwrite its output directory. Passing this local
numeric gate covers one captured layer-0 activation only; it is not an
end-to-end accuracy or board-performance result.
