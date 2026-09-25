# Frozen Teacher → Student-B knowledge-transfer audit

Scope: the pre-locked 12 Core and 4 Diagnostic V4 **val** plants. This is an explanation of existing frozen predictions, not a retraining/selection step. The Teacher pseudo-target is the **optimization target**; the 34-image internal-validation loss selected Student-B epoch 53; independent human phenotype GT supplies the **scientific target**. The three roles are not interchangeable. V4 test was not read.

The analysis uses unchanged saved points and phenotype matching. The evaluator reconstructed all 16 Teacher and Student graph-node/path counts exactly (zero count discrepancies). Correspondence is one-to-one Hungarian assignment in the 518-pixel model image, primary radius `0.025D`; `0.05D` is sensitivity only. Here `D` is the automatic-support bounding-box diagonal. These spatial correspondences are **not** anatomical keypoint ground truth.

| Group | Teacher → Student points | Spatial pairs 0.025D / 0.05D | Teacher-only / Student-only at 0.025D | Accepted graph nodes T → S | Path-participating points T → S | Decoded paths T → S | Human-GT matched paths T → S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Core 12 | 72 → 51 | 32 / 37 | 40 / 19 | 68 → 45 | 63 → 43 | 24 → 22 | 20 → 20 |
| Diagnostic 4 | 26 → 17 | 12 / 14 | 14 / 5 | 26 → 15 | 26 → 13 | 12 → 9 | 9 → 7 |

The provisional, non-exclusive per-point explanations found 34 matched point **pairs** supporting the same GT-matched path (`true_point_preservation` is present on both members, 68 records), 6 Teacher-only non-path points with no attributable GT loss (`noise_suppression`), 6 Teacher-only points on GT paths that Student missed (`rare_structure_loss`), 1 Student-only point on a newly GT-matched path (`new_useful_point`), 2 Student-only points on unmatched predicted paths (`new_false_point`), and 4 records where terminal/base changes propagated to path topology (`topology_consequence`). Other unmatched points remain unclassified; 54 Teacher-only points must **not** all be called noise. Labels are operational hypotheses, not manual point truth. In particular, `rare_structure_loss` membership does not establish that every such point was individually necessary.

## Per-plant ledger

Each paired cell is Teacher/Student. `pairs` gives 0.025D / 0.05D, `nodes` accepted learned graph nodes, and `used` graph points lying on a decoded path. Full per-plant GT correspondence stays in the private audit output and is not listed here.

| Val plant | Group | Points | Pairs | Nodes | Used | Paths |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0004 | Core | 2/3 | 2/2 | 2/3 | 0/3 | 0/2 |
| 0005 | Core | 6/4 | 2/3 | 4/3 | 3/3 | 1/1 |
| 0006 | Core | 4/4 | 2/3 | 4/3 | 3/2 | 1/1 |
| 0009 | Core | 8/4 | 3/3 | 6/3 | 6/3 | 1/1 |
| 0011 | Core | 5/8 | 3/4 | 5/8 | 5/7 | 2/2 |
| 0016 | Core | 8/3 | 3/3 | 8/3 | 8/3 | 3/2 |
| 0020 | Core | 7/4 | 3/4 | 7/4 | 7/4 | 4/3 |
| 0027 | Core | 8/3 | 3/3 | 8/3 | 8/3 | 2/2 |
| 0028 | Core | 8/4 | 3/3 | 8/4 | 7/4 | 3/2 |
| 0030 | Core | 5/4 | 2/3 | 5/3 | 5/3 | 3/1 |
| 0035 | Core | 7/6 | 3/3 | 7/5 | 7/5 | 3/4 |
| 0037 | Core | 4/4 | 3/3 | 4/3 | 4/3 | 1/1 |
| 0002 | Diagnostic | 5/3 | 2/3 | 5/2 | 5/0 | 2/0 |
| 0018 | Diagnostic | 9/5 | 5/5 | 9/5 | 9/5 | 4/4 |
| 0023 | Diagnostic | 6/4 | 2/3 | 6/3 | 6/3 | 3/2 |
| 0034 | Diagnostic | 6/5 | 3/3 | 6/5 | 6/5 | 3/3 |

## Both Diagnostic missing paths: `v4_val_0002`

Both of the Student's Diagnostic losses are on the **same** plant and share a base failure; they are not two independent failure events. Teacher generated 5 points, all 5 entered the graph. Teacher point `p02` at model `(391,279)` was an accepted common base for both complete paths, which matched both GT traces. Student generated only 3 points. Two accepted points correspond to distal Teacher points, but Student basal candidate `p02` shifted to `(399.87,274.66)`. Its nearest-skeleton projection distance was **9.87 px**, beyond the unchanged `0.025D = 8.03 px` limit (`D=321.08 px`), so the graph rejected it as `projection_too_far`. At the looser *diagnostic-only* `0.05D`, this candidate is spatially paired with the Teacher basal point; that does not make it graph-eligible. The surviving two nodes do not supply a shoot-side basal-transition node. The unchanged decoder reports exactly `no_learned_node_near_shoot_side_transition` and returns 0 paths. Hence phenotype matching has 0 candidates and both previously matched GT paths are missed.

The first *observed* decisive failure is point filtering/projection after an upstream point-location shift. Graph construction carries the missing base forward; decoder's empty output is a consequence, and phenotype matching is not the cause. This reconstruction does **not** prove a counterfactual repair would restore both paths; no threshold, NMS, graph, decoder or model was altered to test that.

The complementary Core failures matter: at `v4_val_0004`, Student's extra basal point allows 2/2 GT paths while Teacher yields none; at `v4_val_0030`, Teacher matches 3/3 while Student matches only 1/3. Thus fewer Student points can suppress some non-path candidates, but cannot be equated with uniformly better structure preservation.

Private per-point records, GT associations, rejection distances and full deblinded identities remain under `phenotype_pilot_protocol/runtime/method_gate_20260925/transfer/` and are excluded from Git. The public audit script checks source hashes before reading those files. It does not run a model or access test.
