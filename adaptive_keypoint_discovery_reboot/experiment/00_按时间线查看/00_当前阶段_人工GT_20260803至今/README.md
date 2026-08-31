# 00 当前阶段：phenotype-first模型盲人工GT

## 本阶段目的

用独立人工描迹裁决冻结方法是否真正支持表型计算。人工GT只用于评价，不参与训练、阈值选择或逐图修改模型。

## 主要入口

- [已批准的人工参考协议](../../phenotype_pilot_protocol/Phenotype-first人工参考协议_v2_已批准待实施_20260803.md)
- [HTML平台设计说明](../../phenotype_pilot_protocol/模型盲人工GT_HTML平台设计说明_20260803.md)
- [平台实施与验收报告](../../phenotype_pilot_protocol/模型盲人工GT平台实施与验收报告_20260803.md)
- [人工轨迹与跨轮次匹配锁定规范](../../phenotype_pilot_protocol/人工跨轮次匹配与轨迹计算锁定规范_20260803.md)
- [MDC95与人工可靠性规则](../../phenotype_pilot_protocol/MDC95与人工可靠性计算规则_预锁定_20260803.md)
- 本地测量包：`../../phenotype_pilot_protocol/runtime/packages/`
- 本地原始导出：`../../phenotype_pilot_protocol/runtime/measurements/`

## 当前状态

- Rater 1第一轮：16个session、35条trace已提交；
- 结构、清单和240点曲线检查通过；
- 两条遮挡/插值字段组合待通过revision明确；
- 第一轮尚未正式冻结；
- Rater 1第二轮需间隔3–7天；
- 人工GT冻结前禁止运行Teacher-direct/B/D表型比较。

## 绝对不要做

- 不把 `runtime/admin/` 或人工原始导出上传GitHub；
- 不查看方法结果后修改人工GT；
- 不覆盖原始第一轮导出；
- 不修改Teacher、Student、Decoder、阈值或V4 test边界。
