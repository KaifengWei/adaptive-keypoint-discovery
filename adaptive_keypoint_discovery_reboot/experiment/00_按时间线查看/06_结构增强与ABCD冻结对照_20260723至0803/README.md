# 06 结构增强与A/B/C/D冻结对照（2026-07-23至08-03）

## 当时回答的问题

新增路径来自增强Teacher，还是来自局部叶宽尺度Decoder；更复杂的C/D是否值得保留。

## A/B/C/D

| 方案 | Teacher | Decoder | 身份 |
|---|---|---|---|
| A | Route B | 全局尺度 | 对照 |
| B | Route B | 局部叶宽尺度 | 当前主要Student候选 |
| C | 结构覆盖增强 | 全局尺度 | 对照 |
| D | 结构覆盖增强 | 局部叶宽尺度 | 冻结表型对照，不再开发 |

## 权威产物

- [结构覆盖增强首轮训练与二维拆分](../../结构覆盖增强方案C首轮训练与二维拆分结果_20260723.md)
- [方案C独立人工复核](../../方案C独立人工路径复核结果_20260725.md)
- [当前路线证据审计与冻结计划](../../当前研究路线证据审计与下一阶段冻结计划_20260802.md)
- [A/B/C/D人工配对复核与因果分析](../../A_B_C_D人工配对复核与因果分析_20260803.md)
- 结构增强教师：`../../pseudo_labels_g1prime_v4_structure_coverage/`
- 结构增强Student点：`../../evaluation_outputs/core_dinov2_v4_structure_coverage_val/`
- C全局decoder：`../../evaluation_outputs/point_conditioned_organ_paths_v3_structure_coverage_global_decoder_val/`
- D局部decoder：`../../evaluation_outputs/point_conditioned_organ_paths_v3_structure_coverage_val/`
- 人工复核系统：`../../manual_review_system/`
- 因果分析表：`../../factorized_method_review_summary_20260803.md`及同名JSON文件

## 当前结论

局部decoder有稳定覆盖收益但会引入假枝/错连；增强Teacher没有证明净收益。停止修补C/D，保留B作为最小充分Student候选，D只进行一次冻结表型对照。
