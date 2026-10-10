# Q8×Q8 ZCU102 PS runtime and deterministic self-test

This directory contains the first PS-side runtime for the routed
`q8q8_dataflow` accelerator. It does not generate or program a bitstream.

The Linux MVP uses `generic-uio` with two mappings:

- map0: accelerator AXI4-Lite control at `0xA0000000`, 64 KiB;
- map1: a 64 MiB `no-map` DDR arena at `0xB7C000000`.

The arena is the top 64 MiB of the validated 14 GiB DDR_HIGH region and is
outside normal Linux allocation. UIO maps physical memory non-cached, so the
driver uses barriers and does not pretend that an ordinary process virtual
address is a PL-visible address. The arena is an MVP for functional tests;
production integration should use a kernel DMA API or a reviewed reserved
memory allocator.

Run all host-side checks:

```sh
make all
```

This runs the native unit test, ASan/UBSan, AArch64 cross-build, device-tree
overlay test, and a comparison between the runtime register offsets and the
header exported inside the synthesized HLS IP.

The deterministic board test is `M=2, N=4, K=32`, uses physical addresses
above 4 GiB, and expects exactly:

```text
[64, 0, -32, 0, -96, 0, 48, 0]
```

After a future bitstream and matching device tree are loaded, run on ZCU102:

```sh
/opt/robobrain/q8q8-selftest --timeout-ms 5000
```

The large-matrix benchmark uses the same UIO device and a bounded bump
allocator for the 64 MiB arena. It generates deterministic Q8_0 operands,
checks representative outputs against a scalar golden model, and reports
kernel-only time separately from data preparation:

```sh
/opt/robobrain/q8q8-benchmark --case language_prefill --repeat 3
/opt/robobrain/q8q8-benchmark --case vision_encoder --repeat 3
/opt/robobrain/q8q8-benchmark --case decoder_gemv --repeat 3
```

`language_prefill` deliberately remains the frozen `M=512` one-launch shape
used to compare PE8, PE16 and DATAFLOW kernels.  It must not be called a full
image Prompt.  The separate `visual_prefill` case represents the observed
`M=1080, N=9728, K=2560` image-prompt workload and is the like-for-like input
for fixed-512 versus automatic DMA chunking:

```sh
/opt/robobrain/q8q8-benchmark --case visual_prefill --chunk-m 512 --repeat 3
/opt/robobrain/q8q8-benchmark --case visual_prefill --auto-chunk --repeat 3
/opt/robobrain/q8q8-benchmark --case vision_encoder --chunk-m 512 --repeat 3
/opt/robobrain/q8q8-benchmark --case vision_encoder --auto-chunk --repeat 3
```

The optional `--apm traffic|latency` switch programs the hardened ZynqMP
DDR AXI Performance Monitor at `0xFD0B0000` and brackets every accelerator
launch with real hardware counters:

```sh
/opt/robobrain/q8q8-benchmark --case language_prefill --apm traffic
/opt/robobrain/q8q8-benchmark --case language_prefill --apm latency
```

DDR APM slots 1 and 2 are the two striped CCI-to-DDR ports, so they observe
the combined HPC0/HPC1 traffic and must be summed; they cannot be labelled as
one HLS bundle each. Slot 3 observes the direct HP0 DDR path used for scales
and output. Traffic mode records bytes, transactions, read completions and
write beats. Latency mode records total/max read latency and the two AXI4-MM
idle metrics exposed by the hardened APM. All counters are reset per chunk,
overflow is fatal, and the raw APM clock count is reported so its frequency is
measured rather than assumed.

With the frozen policy, `visual_prefill` executes `M=1080` as `512+512+56`
and `vision_encoder` executes `M=4320` as eight 512-row jobs plus one 224-row
job.  `--auto-chunk` selects the largest aligned M chunk that fits the same
64 MiB arena (currently planned as `976+104` and `3584+736`, respectively).
The reported GOPS always use the full `2*M*N*K` work. The activation
quantizer is not part of the accelerator or benchmark timing.

Timeout does not imply that an in-flight AXI transaction can safely be
aborted. The runtime latches a fault and prevents another launch until the IP
reports idle or the PL reset domain is reset.

The optional RAM-Linux integration is disabled by default. A future matching
hardware image can enable it with `ROBOBRAIN_PL_ACCEL_MODE=1` when running
`deployment/scripts/build_ram_linux.sh`; this builds UIO into the kernel,
applies the reserved-memory overlay, and stages the AArch64 self-test. It does
not itself create or program a PL bitstream.

The DATAFLOW RAM image also stages a versioned, fail-fast board protocol at
`/opt/robobrain/q8q8-dataflow-measure`. It performs a pre/post numerical
self-test, the frozen M=512 kernel comparison, fixed-512 versus auto-chunk A/B
at identical full visual shapes, and traffic/latency APM profiles. It checks
`ap_idle` after every command and rejects concurrent runs. Running the script
does not reset or program the board, but its results are valid only after the
matching bitstream/XSA/FSBL/RAM-Linux JTAG gate has passed.
