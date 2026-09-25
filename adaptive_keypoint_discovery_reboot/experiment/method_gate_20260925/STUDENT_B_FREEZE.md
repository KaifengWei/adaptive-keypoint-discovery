# Final frozen Student-B registration

Date: 2026-09-25. Status: **final frozen Student-B**. This registration does not copy or overwrite the checkpoint.

| Item | Frozen value |
| --- | --- |
| Checkpoint | `experiment/training_outputs/core_dinov2_v4_phenotype_roi/best.pt` |
| Checkpoint SHA-256 | `bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2` |
| Full frozen training configuration | `experiment/training_outputs/core_dinov2_v4_phenotype_roi/resolved_config.json` (SHA-256 below); retained in Git, not rewritten |
| Seed | `20260718` |
| Original internal split | 182 train / 34 internal validation, hash split of the 216 usable Route B train pseudo-target images; validation fraction `0.15` |
| Best checkpoint epoch / logged batch step | 53 / 2438 |
| Best internal-validation objective | `0.28449777762095135` |
| Completed budget | 80 epochs / 3680 logged minibatches |
| Training history SHA-256 | `9ab8e3b42dbf555cd3af0daecc5b4db528a5d70e8d19e264a1b59c4b5875a1dc` |
| Resolved configuration SHA-256 | `87929e7b5e318631d2b9dfa4ab87d981b30491144dbd0fb07fce59786b9a89b1` |
| Frozen training pseudo-targets SHA-256 | `1b1ae7e376f127d6336ae13a3ccd926aa1f595dcd63244864e436b4cc5d8f5b8` |
| Optimization recipe | DINOv2 ViT-S/14-Reg frozen; dynamic heatmap head only; AdamW, head LR `0.001` constant, weight decay `0.01`, batch 4, gradient accumulation 2, weighted BCE + 0.02 count Smooth-L1 |

The checkpoint metadata, 80-row history, and training summary agree exactly on epoch 53 and the minimum validation loss. The source code selects `best.pt` only when the internal-validation loss strictly decreases; it has no scheduler or early stopping. The last 27 epochs do not improve the minimum. The 2026-09-24 sufficiency audit was accepted by the user as **CONVERGED** on this frozen objective and recipe. No further Student-B training is permitted.

Audit sources retained without modification: `experiment/training_outputs/core_dinov2_v4_phenotype_roi/history.csv`, `resolved_config.json`, `training_summary.json` and checkpoint metadata. The epoch-53 loss is the minimum of all 80 internal-validation records; epoch 80 reports train loss `0.2646624769853509` and internal-validation loss `0.2922690858443578`, so additional epochs after 53 did not beat the selected checkpoint. This is a sufficiency decision on the Teacher pseudo-target objective, not a phenotype-GT stopping rule.

The 3680 logged `global_step` value counts minibatches, not confirmed successful optimizer updates. Gradient accumulation schedules 1840 calls to `GradScaler.step`; AMP skip events were not logged.

Three targets remain separate: optimization uses frozen Teacher pseudo-targets; checkpoint selection uses the 34-image internal-validation loss; scientific success uses independent phenotype GT and Teacher-direct comparison. The latter cannot alter the checkpoint. Core 12 / Diagnostic 4 and V4 test remain locked.
