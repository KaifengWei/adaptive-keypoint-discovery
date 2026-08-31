# 02 V3自动教师与DINOv2首轮训练（2026-07-17）

## 当时回答的问题

G1′跨增强一致产生的数量可变自动目标，能否被DINOv2动态热图Student学习。

## 权威产物

- 数据：`../../data_stage_clean_v3/`
- 自动教师：`../../pseudo_labels_g1prime_v3/`
- 三张代码冒烟：`../../pseudo_labels_g1prime_v3_smoke3/`
- [V3阶段验证结论](../../G1prime_V3阶段验证结论.md)
- [远程核心训练首轮结果](../../远程核心训练首轮结果_20260717.md)
- [核心模型恢复训练与topology v2评估](../../核心模型恢复训练与topology_v2评估_20260717.md)
- 旧路径输出：`../../outputs_g1_prime_v3/`
- V3评价：`../../evaluation_outputs/core_dinov2/`、`../../evaluation_outputs/core_dinov2_recovery_v2/`

## 当前身份

证明自动教师可以被学习，但V3不是当前正式数据入口。它不能代替V4人工表型评价，也不能作为最终论文准确率。
