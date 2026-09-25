# 2026-09-25 frozen-method gate: reading order

1. `STUDENT_B_FREEZE.md` — accepted `CONVERGED` epoch-53 checkpoint registration; never retrain or overwrite.
2. `AUDIT_PROTOCOL.md` — the transfer/benchmark/perturbation rules committed **before** new measurements.
3. `KNOWLEDGE_TRANSFER.md` — 16-plant point → graph → path audit, including the two Diagnostic path losses.
4. `EFFICIENCY_ROBUSTNESS.md` — same-3090 measurement and pre-fixed perturbation results.
5. `FINAL_METHOD_GATE.md` — the A/B/C scientific decision. This is the entry point for project reporting.

`audit_transfer.py` reconstructs frozen val-only points/graph/path decisions. `run_frozen_diagnostics.py` runs `--mode preflight`, `benchmark` or `robustness` for one `--method Teacher-direct|Student-B` per process. `summarize_diagnostics.py` validates N/hardware and aggregates raw records. `test_frozen_diagnostics.py` verifies source-coordinate geometry against the frozen pilot evaluator. The scripts do not use human GT for timing/robustness or read V4 test.

Inputs/outputs with per-point GT associations and deblinded method detail are intentionally outside Git at `experiment/phenotype_pilot_protocol/runtime/method_gate_20260925/`. In particular, `sources/teacher_points.csv` is the exact frozen Route B val-only Teacher point source, SHA-256 `1ed017a6d6ba6f360bf216c44b01ffedaef9c2a14f18d46d75b1f640763b1814`. Student points remain in the saved frozen `evaluation_outputs/core_dinov2_v4_phenotype_roi_val/points.csv`. The public script checks both hashes. Do **not** replace either source with an unverified new inference file.

All GPU runs were on `cv` in `/media/neaucs2/evs/envs/adaptive_kp` after `unset LD_LIBRARY_PATH`; the benchmark runs Teacher and Student in separate processes. Full per-image raw CSV/JSON are retained locally under the ignored `measurements/` directory. No subsequent training, threshold tuning or test opening is authorized by this folder. After reporting the gate to the user, V4 test evaluation requires a **separate preregistration**.
