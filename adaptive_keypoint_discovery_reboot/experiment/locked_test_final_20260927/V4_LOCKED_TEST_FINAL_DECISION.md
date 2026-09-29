# V4 locked-test final decision

日期：2026-09-29。决定依据只有原预注册、单人 GT 行政修订、冻结 40 株 GT、冻结 evaluator，以及首次且唯一一次 Teacher-direct/B/D × 40 株预测包。此处**解释 test，不在 test 上重新选择或开发方法**。

1. **Teacher-direct + Pipeline V1 保留为当前更强的表型结构恢复参考。** 它的逐株可测路径召回为 84.17%，Student-B 为 66.67%；预注册 primary 配对差 B−Teacher=−17.50 个百分点，95% CI [−27.09,−8.75]，8-source-frame 敏感性方向一致。因此不能再称 Student-B 与 Teacher 的 phenotype 覆盖“接近”或“等价”，更不能称 B 全面超过 Teacher。
2. **Student-B epoch53 仍是冻结的 learned surrogate / knowledge-distillation 结果，而非被 test 否决后可重训的候选。** B 只学习原冻结 Teacher pseudo-target，checkpoint 只由原 182/34 internal validation loss 在 epoch53 选出。test 表明它以漏检为代价减少 possible-extra，并在本机同条件单次前向下取得中位约 18.72× 端到端速度优势。这是明确的效率/部署取舍，不是表型等价性证明。对于必须完整识别叶片结构的场景，当前 B 不宜作为无保留替代。
3. **Student-D 继续仅作为预定 secondary comparator。** D 的召回高于 B、仍低于 Teacher；其 test 表现不得触发主方法重选、checkpoint 变更、阈值/decoder/graph 调整或新训练。Pipeline V2 的阴性开发裁决和 V1 回退保持不变。
4. **整株表型自动化 claim 必须收窄。** 当前完整植株严格成功数 Teacher/B/D 为 4/3/5 株（各 40 株）；基部失败为 29/28/25 株。虽然部分路径的长度和角度能够计算，不能据其条件子集误差掩盖漏叶、零路径和基部位置问题。论文应分开陈述“无人工关键点标签的自适应点学习与知识迁移”“路径/表型条件精度”“整体结构恢复覆盖率”，不得宣称已达到可靠全株测量。
5. **研究边界永久保持。** final GT、Teacher、Student-B/D 权重、Pipeline V1、evaluator 和本次 saved predictions 均不修改。不得基于 test 失败重标 GT、重训、调阈值、设计 test 个例规则、开发 V3 或追加第二轮 V4 test。若未来研究第二代方法，须另外预注册并使用新的独立评价数据，不复用本次 locked test 作开发集。

关键统计、置信区间、source-frame 敏感性、效率和例图编号见 `V4_LOCKED_TEST_FINAL_RESULTS.md`；执行完整性、哈希及私有原始记录入口见 `V4_LOCKED_TEST_FINAL_RUN_AUDIT.md`。

始终分清：`Optimization target = frozen Teacher pseudo-targets`；`Checkpoint-selection target = original internal-validation objective`；`Scientific target = independent single-rater model-blind phenotype GT`。本 test 只裁决第三项，不反过来修改前两项。
