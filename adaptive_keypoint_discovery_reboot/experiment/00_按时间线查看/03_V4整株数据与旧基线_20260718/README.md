# 03 V4整株数据与旧基线（2026-07-18）

## 当时回答的问题

- 从全部候选来源中构建质量更稳定、分组隔离的300张数据；
- 保留整株后，模型和路径会不会被颖果与根须干扰。

## 当前正式数据

- `../../data_stage_clean_v4_fullplant_candidate/`：300张，220 train / 40 val / 40 locked test；
- [V4来源清单](../../v4_fullplant_source_manifest.csv)
- [V4数据构建与锁定说明](../../V4数据集构建与锁定说明_20260717.md)
- 构建脚本：`../../build_stage_clean_v4_fullplant.py`
- 审计脚本：`../../audit_stage_clean_v4_fullplant_candidate.py`

## 旧整株基线

- 自动教师：`../../pseudo_labels_g1prime_v4_fullplant/`
- 训练摘要：`../../training_outputs/core_dinov2_v4_fullplant/`
- 评估：`../../evaluation_outputs/core_dinov2_v4_fullplant_val/`
- [整株首轮训练与架构审计](../../V4整株首轮训练与架构审计_20260718.md)

## 当前身份

V4数据仍是当前唯一正式数据。整株直接建模的方法已被替代，因为人工审计发现颖果和根须会污染基部与路径；这部分保留为路线B的必要对照证据。
