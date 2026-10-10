# RoboBrain2.5-4B ZCU102 RAM deployment acceptance

Date: 2026-07-21

## Result

- DDR probe: 17179869184 bytes, 2 ranks.
- Linux DT memory: 16776192 KiB; `/proc/meminfo` MemTotal: 14310760 KiB after ZynqMP reserved memory and CMA.
- RAM-only Linux boot: PASS; SD/flash were not written.
- PS Ethernet: `192.168.50.2/24`, 1 Gbps/full duplex, 3/3 final ping, 0.091-0.108 ms.
- Model transfer: `RoboBrain2.5-4B.Q8_0.gguf` and allow-listed runtime artifacts fetched from `192.168.50.1:8000` into DDR and SHA-256 verified.
- llama-server: `GET /health` returned `{"status":"ok"}`.
- Post-chat SD attestation: `SD_FINAL_PASS state=absent ... write_counters_unchanged=1`.

## Language acceptance

| Prompt | Response | Elapsed |
|---|---|---:|
| `Reply with only the word OK.` | `OK` | 14.084 s |
| `用两个汉字回答：天空通常是什么颜色？` | `蓝色` | 14.097 s |
| `What is 1+1? Answer with only the number.` | `2` | 16.739 s |
| deterministic repeat | `OK` | 2.086 s |

Overall result: `CHAT_TEST_PASS`.

## Vision acceptance

- Task: single-image affordance prediction on `app/assets/demo/affordance.jpg`.
- Request path: JPEG base64 over PS Ethernet to the OpenAI-compatible API.
- Runtime: Q8 model plus Q8 mmproj, CPU-only, 4 Cortex-A53 cores.
- Context/batch: `ctx=2048`, `batch=2048`, `ubatch=512`, image tokens 1024.
- Response: `[(920, 548)]`, a valid normalized 0..1000 coordinate close to the
  server CPU reference result.
- End-to-end elapsed: 1725.623 s (28 min 45.6 s).
- Server metrics: 1142 prompt tokens in 1709.905 s (0.668 token/s); 11 output
  tokens in 15.436 s (0.713 token/s).
- During inference the independent read-only status endpoint remained healthy,
  all four CPU cores were saturated, and more than 7 GiB remained available.
- Post-inference health: PASS. Post-inference SD attestation:
  `SD_FINAL_PASS state=absent ... write_counters_unchanged=1`.

Overall result: `VISION_TEST_PASS`. The measured bottleneck is the CPU vision
encoder/prompt processing, not Ethernet or DDR capacity. Increasing the LLM
batch to its context-limited maximum preserved correctness but did not
materially accelerate the vision encoder.

Artifacts:

- `deployment/reports/board-chat-result.json`
- `deployment/reports/board-vision-affordance-observed-result.json`
- `deployment/reports/2026-07-21-zcu102-uart-robobrain4b.log`
- `deployment/reports/2026-07-21-zcu102-uart-vision-batch2048.log`
- `deployment/reports/2026-07-21-zcu102-runtime-vision-batch2048.log`

The service remains volatile: a board reset or power cycle removes Linux, model files, and the running server from DDR.
