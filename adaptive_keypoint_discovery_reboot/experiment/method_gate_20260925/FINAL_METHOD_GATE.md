# Final frozen-method gate — 2026-09-25

## Decision: **B — learned surrogate / knowledge-distilled adaptive keypoint detector**

This is the decision on the **locked 16-plant V4 val pilot**, not a claim of final V4 test accuracy or external generalization. Student-B is the final frozen epoch-53 checkpoint registered in `STUDENT_B_FREEZE.md`; further Student-B training and method adjustment are prohibited. The same frozen point-conditioned graph, local decoder and phenotype geometry are used downstream of Teacher-direct and Student-B. The only differing source is generated points.

| Permitted gate | Decision | Reason |
| --- | --- | --- |
| A. Clear Student phenotype benefit | **Not supported** | Core has the same 20/24 complete-GT path matches; conditional length error is 3.51% Student versus 4.39% Teacher, but the paired plant-cluster effect is −0.49 percentage points with 95% CI [−1.95, 1.46], spanning zero and smaller than the 3.72% human MDC95. Diagnostic Student matches 7/9 versus Teacher 9/9. |
| B. Comparable Core phenotype with clear efficiency, stability or deployment value | **Selected** | Core recall is equal and no reliable phenotype advantage is claimed; on the same RTX 3090 the Student's in-memory end-to-end median is 42.33 ms versus Teacher 720.26 ms (`N=160` per method), paired median Teacher/Student ratio 17.75×. Point generation alone is 50.94× by the paired ratio. This is a clear measured single-image deployment value. |
| C. No phenotype or practical value | **Not supported** | The reproducible runtime gain is substantial, although it does not remove the Diagnostic structural losses or make Student a proven accuracy winner. |

## What the Student learned and what it lost

Across 16 plants, Teacher has 98 points and Student 68; 44 point pairs match within the pre-locked `0.025D` radius (51 pairs at the descriptive `0.05D` sensitivity). Student preserves 34 matched point pairs participating in the same GT-matched phenotype paths. Six strictly identifiable Teacher-only non-path points support a cautious `noise_suppression` reading, not a claim that every missing Teacher point was noise. At the same time, six Teacher-only points lie on GT paths Student missed; this `rare_structure_loss` includes the basal support of the two Diagnostic failures. One Student-only point supports a newly recovered GT path, and two Student-only points appear on unmatched predictions. These are explanatory categories, not point-wise anatomical labels.

The two Diagnostic GT paths lost by Student are both on `v4_val_0002`. Student's candidate near the common base is shifted and rejected by the unchanged graph projection (`9.87 px` to skeleton against `0.025D = 8.03 px`); the decoder then has no shoot-side basal node and returns zero paths. This is a point-location/projection bottleneck propagated into topology, **not** a phenotype-matching mistake. The mirror case `v4_val_0004` shows Student recovering two GT paths Teacher misses; `v4_val_0030` shows Student missing two Teacher-matched paths. No case was repaired after viewing GT.

## Efficiency, memory and stability qualification

The same-hardware benchmark starts from an already prepared 518×518 ROI image in memory; file I/O and ROI construction are excluded. Graph+decoder medians are close (27.10 ms Teacher, 24.86 ms Student); this stage is **not** the Student's distinctive advantage. End-to-end peak CUDA **allocated** memory is 137.3 MiB Teacher and 122.3 MiB Student, while process peak RSS is 1301.2 MiB Teacher and 1338.9 MiB Student. Student is not universally more memory-efficient; these counters are not whole-device peak occupancy.

The locked perturbation diagnostic shows mixed behavior. For Core +10° rotation, median point F1 is .857 Student versus .586 Teacher, exact path count is 11/12 versus 5/12, and original-path correspondence is 20/22 versus 16/24. Under Diagnostic +10°, Student's exact path count is only 1/4 versus Teacher's 2/4; matched-path length changes are also larger (8.23% versus 1.28%, conditional on 7 and 9 matches respectively). Brightness ×.85 stabilizes Student point counts, but cannot compensate for the frozen identity-image Diagnostic loss. These 16 images and three mild transforms support a **perturbation robustness** statement only, not external generalization or guaranteed phenotype reliability.

## Placement and non-negotiable boundaries

The appropriate description is **“knowledge-distilled, variable-count adaptive keypoint detector serving as a fast learned surrogate of the automatic Teacher”**. Teacher-direct remains the strong full-consensus reference and a visible counterexample on difficult structures. Do not write that Student is more accurate, that all Teacher removals are beneficial, or that Student has proved equivalence on rare morphology. For quality-sensitive use, the Diagnostic loss is an explicit limitation; the gate does not authorize a new hybrid, per-image exception or threshold repair.

The three objectives stay separate:

- **Optimization target:** frozen Route B Teacher pseudo-targets on the 182-image training portion.
- **Checkpoint-selection target:** 34-image internal-validation objective; epoch 53 has best loss `0.28449777762095135`, independently audited `CONVERGED`.
- **Scientific success target:** independently frozen phenotype GT with Teacher-direct comparison; it adjudicates method value, never checkpoint choice.

No new training, architecture/loss/threshold/NMS/ROI/graph/decoder/geometry/GT/Core–Diagnostic changes or V4 test reading occurred in this gate. The 16-image pilot cannot substitute for a final untouched test. **Stop here and report to the user; only after their review may a separate V4 test evaluation protocol be preregistered.**

Supporting evidence: `KNOWLEDGE_TRANSFER.md`, `EFFICIENCY_ROBUSTNESS.md`, `AUDIT_PROTOCOL.md`, `STUDENT_B_FREEZE.md`, and the pre-existing `experiment/phenotype_pilot_protocol/冻结方法表型先导比较_20260924.md`. Deblinded GT-linked per-point data and raw timing/robustness files are under Git-ignored `experiment/phenotype_pilot_protocol/runtime/method_gate_20260925/` and are not for GitHub publication.
