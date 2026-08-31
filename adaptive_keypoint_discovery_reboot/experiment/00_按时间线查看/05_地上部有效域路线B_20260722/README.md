# 05 地上部有效域与路线B（2026-07-22）

## 当时回答的问题

在保留完整整株档案的同时，怎样让地上部表型链排除颖果、根须和错误基部影响。

## 权威产物

- [人工路径审计与基部重定义建议](../../人工路径审计与地上部基部重定义建议_20260722.md)
- [地上部有效域与基部过渡区验证](../../地上部表型有效域与基部过渡区_v1验证报告_20260722.md)
- [路线B首轮训练与val对照](../../地上部有效域路线B首轮训练与val对照报告_20260722.md)
- [路线B人工复核与下一轮建议](../../路线B人工路径复核与下一轮改进建议_20260723.md)
- 有效域实现：`../../phenotype_roi_basal_anchor.py`
- 排除清单：`../../phenotype_input_exclusions_v1.csv`
- Route B自动教师：`../../pseudo_labels_g1prime_v4_phenotype_roi/`
- Route B训练摘要：`../../training_outputs/core_dinov2_v4_phenotype_roi/`
- Route B点：`../../evaluation_outputs/core_dinov2_v4_phenotype_roi_val/`
- Route B图：`../../evaluation_outputs/point_conditioned_graph_v2_phenotype_roi_val/`
- A全局decoder：`../../evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_val/`
- B局部decoder：`../../evaluation_outputs/point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val/`

## 当前身份

路线B是当前primary/minimum-sufficient Student候选的来源，但尚未通过人工表型裁决证明优于Teacher-direct。
