# RoboBrain2.5 / ZCU102 continuation rules

1. Before making changes, read `CURRENT_STATUS.md` and
   `docs/PL_ACCELERATION_RESEARCH_PLAN_2027.md`.
2. Treat this directory as the canonical active workspace:
   `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.
   Do not overwrite or automatically synchronize it with
   `/home/zhaoxiangyu/RoboBrain2.5`, which is an older independent copy.
3. Never use a revealed `blind` or `validation` manifest for training,
   hyperparameter selection, or algorithm design. Reproduce frozen results only
   when the user explicitly requests it.
4. Before any CUDA model load, inspect GPU memory and process ownership. Do not
   kill, pause, or modify another user's process.
5. Do not reset, program, or power-cycle ZCU102 unless the current task requires
   board access and the user has authorized that action.
6. Keep raw experiment records append-only and resumable. Store machine-readable
   JSON/JSONL plus a Chinese Markdown report. Do not claim accuracy or speedup
   beyond the gate documented in `CURRENT_STATUS.md`.
7. After every completed or stopped milestone, update `CURRENT_STATUS.md` with
   the decision, evidence paths, verification result, and exact next command.

