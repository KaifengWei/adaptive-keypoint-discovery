# 00 当前阶段：phenotype-first模型盲人工GT

## 本阶段目的

用独立人工描迹裁决冻结方法是否真正支持表型计算。人工GT只用于评价，不参与训练、阈值选择或逐图修改模型。

## 主要入口

- [当前两阶段裁决与交接](../../phenotype_pilot_protocol/两阶段人工裁决实施与交接_20260906.md)
- [裁决工具、填写说明与管理员命令](../../phenotype_pilot_protocol/adjudication_v2/README.md)
- [当前双人共识填写入口](../../phenotype_pilot_protocol/runtime/adjudication/semantic_first_20260906/joint_consensus_20260915_v2/index.html)（GitHub不含runtime图像，需发送整个文件夹）
- [双人共识证据与交接说明](../../phenotype_pilot_protocol/第二阶段几何核验与双人共识导航_20260913.md)
- [已批准的人工参考协议](../../phenotype_pilot_protocol/Phenotype-first人工参考协议_v2_已批准待实施_20260803.md)
- [HTML平台设计说明](../../phenotype_pilot_protocol/模型盲人工GT_HTML平台设计说明_20260803.md)
- [平台实施与验收报告](../../phenotype_pilot_protocol/模型盲人工GT平台实施与验收报告_20260803.md)
- [人工轨迹与跨轮次匹配锁定规范](../../phenotype_pilot_protocol/人工跨轮次匹配与轨迹计算锁定规范_20260803.md)
- [MDC95与人工可靠性规则](../../phenotype_pilot_protocol/MDC95与人工可靠性计算规则_预锁定_20260803.md)
- [Rater 1第一轮修订与冻结审计](../../phenotype_pilot_protocol/Rater1第一轮修订与冻结审计_20260904.md)
- [Rater 1第二轮提交与冻结审计](../../phenotype_pilot_protocol/Rater1第二轮提交与冻结审计_20260904.md)
- 本地测量包：`../../phenotype_pilot_protocol/runtime/packages/`
- 本地原始导出：`../../phenotype_pilot_protocol/runtime/measurements/`

## 当前状态

- Rater 1第一轮：16个session、35条trace已完成revision修订并冻结；
- Rater 1第二轮：16个session、36条trace已通过结构、字段、清单和240点曲线检查并冻结；
- 两轮锁定样本集合一致，blind ID与image alias重合0；15/16株trace数相同，仅1株第二轮多1条，保留为重复测量差异；
- 两轮原始导出和外置冻结清单均保留在Git忽略的runtime目录；
- Rater 2第一轮16张30条已完成并冻结，三轮匿名匹配及裁决前可靠性已计算；
- 用户13张A/B/C选择归档为Rater 1独立意见；Rater 2两阶段30个结构的语义与几何意见已经校验归档；
- 当前正式门槛是两位测量者共同完成13张逐图、逐叶共识记录：7张优先讨论、6张质量确认；无完整候选则重描，无法一致则交第三位，之后才显式冻结最终GT；
- 人工GT冻结前禁止运行Teacher-direct/B/D表型比较。

## 绝对不要做

- 不把 `runtime/admin/` 或人工原始导出上传GitHub；
- 不查看方法结果后修改人工GT；
- 不覆盖原始第一轮导出；
- 不修改Teacher、Student、Decoder、阈值或V4 test边界。
