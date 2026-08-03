# Phenotype-first 人工参考协议（待审核）

- 版本：proposal-v1，2026-08-03
- 数据范围：V4 val 16 张候选
- 方法范围：Teacher-direct、Student-B、冻结的 Student-D
- 当前状态：只完成协议和选样设计，等待用户审核
- V4 test 读取数：0
- 本轮训练、模型修改、阈值调整、五随机种子：均为 0

## 1. 研究目的与冻结边界

本 pilot 只回答两个问题：

1. Teacher-direct 与 Student-B 相比，Student 学习是否给最终表型带来可测量的价值；
2. Student-B 与冻结 Student-D 相比，结构覆盖增强 Teacher 的额外复杂度是否带来实际表型收益。

B 是当前 primary / minimum-sufficient Student candidate。C、D 不再开发；D 仅作为一次性 frozen ablation comparator。人工参考只用于评价，绝不用于训练、Teacher目标、关键点坐标监督或逐图调参。

第一阶段不要求 A 全量进入人工表型比较。A 的既有结果可作为 conservative reference；如后续确有必要，可在同一人工参考上追加自动评价，不需要重新标注。

## 2. 16 张样本的选取

### 2.1 Core 12 的无泄漏选择

Core 在读取任何 A/B/C/D 比较字段之前锁定，只使用：

- 采集时期、原始扫描帧和横/竖放置方向；
- 由标准化原图及数据预处理阶段 shoot mask 得到的面积、外接框长宽比、填充率、骨架密度、端点/分叉复杂度和平均厚度；
- 不使用关键点、Teacher、Student、路径、表型输出或人工方案优劣。

按7个 val 原始扫描帧分配 `3/2/2/2/1/1/1` 个名额；每个扫描帧内先选形态中位样本，再用最远点抽样补足形态差异。自动端点和分叉数只用于形态分层，不作为植物学叶片真值。

锁定后的覆盖为：4个采集时期分别 `4/2/3/3` 张；7个扫描帧全部覆盖；横向10张、纵向2张。锁定后才读取可见叶数作审计，分布为1叶1张、2叶9张、3叶2张，该字段没有参与选择。

| Core编号 | 采集帧 | 时期 | 方向 | 锁定后可见叶审计 | 形态覆盖作用 |
|---|---|---|---|---:|---|
| v4_val_0004 | src_0347 | 0515 | 横向 | 2 | 中等长度、双叶、轻度弯曲 |
| v4_val_0005 | src_0347 | 0515 | 横向 | 2 | 较短、叶片重叠较明显 |
| v4_val_0006 | src_0347 | 0515 | 横向 | 2 | 细长、近直线、短侧叶 |
| v4_val_0009 | src_0546 | 0516 | 横向 | 1 | 单叶、叶片较宽 |
| v4_val_0011 | src_0546 | 0516 | 横向 | 2 | 主路径弯曲、侧叶下垂 |
| v4_val_0016 | src_0615 | 9.14-5.12 | 纵向 | 2 | 直立纵向、两叶分化 |
| v4_val_0020 | src_0615 | 9.14-5.12 | 横向 | 3 | 多叶、分化角明显 |
| v4_val_0027 | src_0565 | 0510 | 横向 | 2 | 长主叶、短侧叶 |
| v4_val_0028 | src_0565 | 0510 | 纵向 | 2 | 细长纵向、叶片接近重合 |
| v4_val_0030 | src_0349 | 0515 | 横向 | 2 | 长主路径、短小侧叶 |
| v4_val_0035 | src_0614 | 9.14-5.12 | 横向 | 3 | 多叶、弯曲和重叠复杂 |
| v4_val_0037 | src_0560 | 0510 | 横向 | 2 | 整体弯曲、小叶较弱 |

### 2.2 Diagnostic 4

Diagnostic 在 Core 锁定后才读取冻结复核结果，且不得与 Core 重复。

| Diagnostic编号 | 选择原因 | 主要诊断问题 |
|---|---|---|
| v4_val_0002 | D>B；增强Teacher恢复两片漏叶并改善基部 | D的正向收益是否真的降低表型误差 |
| v4_val_0023 | B>D；D丢失两片真叶并使基部恶化 | Teacher增强造成的漏叶和基部代价 |
| v4_val_0018 | B出现假枝/错连而D消除该错误 | 错连是否显著污染长度和角度 |
| v4_val_0034 | 局部Decoder产生假枝/错连，且基部均有争议 | 假枝、错连和基部错误的组合影响 |

Core 12 的结果单独形成主要 pilot 统计。Diagnostic 4 只逐例解释失败机制，不与 Core 合并为“16张总体平均值”。

## 3. 人工 phenotype 的操作性定义

### 3.1 一片可见叶

满足以下条件即记录为一片可见叶：

1. 具有可区分的叶尖；
2. 绿色叶片中心走向可从叶尖沿植株追踪至共同地上部路径，或可明确判断其在某处分化；
3. 不把颖果、根、根须、扫描阴影或断裂杂物计作叶；
4. 不设最短叶长阈值。极短新生叶只要具有独立叶尖且可追踪，也计入；
5. 若能确认叶片存在但连续中心线不足以可靠测量，记为 `visible_unmeasurable`，仍进入叶片存在/漏叶统计，但不进入连续长度、角度和曲率误差。

遮挡间隙不超过 shoot 外接框对角线的5%，且两端位置和方向能形成唯一平滑延续时，允许插值并记录 `interpolation_used=yes`；否则标为不可测，不主观补线。

### 3.2 路径起点与终点

本研究测量的是 **shoot-base-to-leaf-tip organ path length**，不是严格植物学意义上从叶舌/叶枕开始的叶片长度。

- 共同起点：绿色地上部从颖果/根复合区域向叶侧离开的中心线点，即已经人工通过的 `shoot-side basal transition`；
- 每条路径终点：相应可见叶片叶尖的中心；不取毛边、根尖或扫描噪声；
- 多叶植株的各条路径允许共享起点后的主轴段，直到各自分化。

这一口径与当前 Decoder 的“共同地上部基点到各终端”输出一致，可避免把算法路径长度错误称为纯叶片长度。

### 3.3 长度

人工沿可见组织中心绘制折线，平台对折线进行弧长参数化和平滑重采样后计算中心线路径长度。

- 主单位：毫米；
- 审计单位：输出图像像素；
- 敏感性单位：`length_px / shoot_bbox_diagonal_px`；
- 不使用裁剪后像素值直接跨图比较物理长度。

7个 val 源 TIFF 均已核对为600 dpi。由于标准化输出曾按最长边1600像素缩放，每图换算为：

`mm_per_output_pixel = 25.4 / (600 × resize_scale)`。

当前16张的换算范围为 `0.0467–0.1230 mm/px`。正式论文前仍建议用一个已知尺寸标尺对扫描仪元数据进行一次实物校验；完成前应写作“基于扫描元数据换算的毫米”。

### 3.4 叶片分化角

先从人工参考路径中指定最长、连续且可作为主体走向的路径为 `reference_main_path`。对每条非主路径：

1. 分化点定义为该叶路径与主体路径最后共享的中心线位置；
2. 参考轴取主体路径在分化点近端长度 `s` 内的局部切线；
3. 叶片轴取分化点远端长度 `s` 内的局部切线；
4. `s = 0.05 × shoot_bbox_diagonal`，若可用段不足则缩短并记录实际区间；
5. 分化角取两条局部切线的无符号夹角，范围 `0–180°`。

主路径本身的分化角记为 `not_applicable`，不得用0°混入角度误差平均值。

### 3.5 曲率

曲率为探索指标。人工折线使用与所有方法相同的平滑和弧长重采样，计算：

`mean absolute curvature = total absolute turning radians / physical path length mm`。

单位为 `mm⁻¹`。同时保留总转角。曲率只有在重复人工测量达到可靠性门槛后才进入辅助结论，不得单独决定方法去留。

### 3.6 多叶身份与预测匹配

- 人工共识完成后，以共同基点为中心，按叶尖在图像坐标中的顺时针极角排序，赋予 `GT01、GT02……`；身份不依赖任何模型；
- 每个方法的预测路径与GT做一对一匹配；匹配只用归一化叶尖距离、基点相容性和中心线空间重合，不使用长度、角度或曲率值，防止“按结果挑最接近的路径”；
- 同一预测不能匹配多片GT叶，同一GT叶不能匹配多条预测；
- 无合格预测记为 `missed_gt`；剩余预测记为 `extra_prediction`；无法唯一匹配时记为 `ambiguous`并进入盲法裁决。

## 4. 人工测量流程

1. **协议培训**：用不属于16张 pilot 的2张 val 开发图练习；练习结果不进入统计。
2. **盲化**：页面只显示无标注标准化原图；不显示Teacher-direct、B、D或旧复核结果。样本使用随机 blind ID，顺序每轮重新打乱。
3. **植株级判断**：先记录可见叶数、可测叶数、共同地上部基点是否可判定。
4. **叶片描迹**：逐叶从共同基点描到叶尖；记录遮挡、插值、可测性和置信度。
5. **首轮检查**：只检查字段完整性和曲线是否越出真实植株，不查看任何模型结果。
6. **重复测量**：Rater 1 间隔3–7天完成第二轮，不能查看第一轮轨迹。
7. **独立复核**：Rater 2 至少完成6张Core和2张Diagnostic；条件允许时完成全部16张。
8. **一致性裁决**：长度差超过5%、角度差超过5°，或叶片存在/身份不一致时，隐藏方法输出后共同复核；保留原始两次记录，另写一条 adjudicated reference，不覆盖原值。
9. **冻结GT**：人工一致性通过并生成哈希后，才运行Teacher-direct、B、D的自动表型比较。

推荐在现有离线HTML复核体系上增加“原图描迹—导出JSON/CSV”页面；在用户批准本协议前不实现该页面。

## 5. CSV schema

采用三个相互分离的平面表，避免把人工原始记录和方法评价混在一起。

### 5.1 `phenotype_pilot_samples.csv`

每行一株，保存冻结分组、采集来源、选择依据及像素—毫米换算。核心字段：

`dataset_id, pilot_group, selection_basis, source_frame_id, acquisition_period, candidate_orientation, source_dpi, resize_scale, mm_per_output_px, image_path`

### 5.2 `manual_phenotype_reference.csv`

每行是“一位测量者、一个轮次、一株中的一片叶”。核心字段：

`protocol_version, dataset_id, pilot_group, blind_id, rater_id, measurement_round, measurement_date, gt_leaf_id, leaf_exists, visibility_status, trace_status, basal_transition_status, start_x_px, start_y_px, tip_x_px, tip_y_px, trace_file, length_px, source_dpi, resize_scale, mm_per_output_px, length_mm, shoot_bbox_diag_px, length_bbox_norm, reference_main_path, divergence_point_x_px, divergence_point_y_px, local_interval_px, divergence_angle_deg, total_turning_angle_deg, mean_abs_curvature_mm_inv, occlusion_grade, interpolation_used, confidence, evaluation_only_no_training, notes`

允许值和缺失原因必须显式编码，例如：

- `visibility_status = measurable | visible_unmeasurable | not_a_leaf`；
- `trace_status = complete | incomplete | not_applicable`；
- `occlusion_grade = none | minor_interpolated | major_unmeasurable`；
- `confidence = high | medium | low`；
- `evaluation_only_no_training`固定为1。

### 5.3 `phenotype_method_evaluation.csv`

每行是一条GT叶与一种方法的匹配结果，或一条未匹配预测：

`dataset_id, pilot_group, method_id, gt_leaf_id, pred_path_id, match_status, match_cost_tip_norm, match_cost_curve_norm, gt_length_mm, pred_length_mm, length_abs_error_mm, length_ape, gt_divergence_angle_deg, pred_divergence_angle_deg, angle_abs_error_deg, gt_curvature_mm_inv, pred_curvature_mm_inv, curvature_abs_error_mm_inv, gt_leaf_count, pred_path_count, matched_count, missed_count, extra_count, evaluation_flag, notes`

`method_id`只允许 `teacher_direct | student_B | student_D`。原始人工表不得包含method字段，避免泄漏。

## 6. 误差计算与统计

### 6.1 Leaf-level

只对 `matched` 且GT可测的叶计算连续误差：

- 长度绝对误差：`|L_pred - L_gt|`，单位mm；
- 长度相对误差：`|L_pred - L_gt| / L_gt`；
- 角度绝对误差：`|θ_pred - θ_gt|`，单位度；
- 曲率绝对误差：`|κ_pred - κ_gt|`，只作探索；
- 同时报告GT叶召回率、预测路径精确率、missed和extra数量，不允许只在成功匹配叶上宣称模型更准。

Leaf-level置信区间必须以“植株”为重采样簇，不能把同株多叶当作相互独立样本。

### 6.2 Plant-level paired error

每株每方法先汇总，再在同一株上进行配对比较：

- `leaf_recall = matched_gt / measurable_gt`；
- `path_precision = matched_prediction / all_predictions`；
- `total_length_APE = |ΣL_pred(all paths) - ΣL_gt| / ΣL_gt`；
- `matched_leaf_length_MdAPE`：该株已匹配叶长度相对误差中位数；
- `matched_angle_MAE`：该株可评价分化角的平均绝对误差；
- `leaf_count_AE = |pred_path_count - gt_leaf_count|`。

总长度误差能让漏叶和假枝进入结果，但可能发生正负抵消，所以必须与召回率、精确率和叶片级误差同时报告，不形成单一“万能总分”。

### 6.3 Core 12 的方法比较

- 统计单位为plant，`n=12`；
- 报告每种方法的中位数、四分位数和逐株配对差；
- 方法差异用精确Wilcoxon符号秩或配对置换检验作辅助；
- 95%置信区间采用按植株重采样的cluster bootstrap；
- pilot不以单个p值宣布论文结论，主要判断效应方向、大小及是否超过人工测量误差底线。

“明显收益”定义为：配对误差改善超过人工重复测量得到的 `MDC95`，并且至少达到长度相对误差5个百分点或角度3°中的相应最低实际差异，同时另一个主指标和叶片召回/精确率没有超过MDC95的系统性恶化。

### 6.4 Diagnostic 4

逐图列出Teacher-direct、B、D的匹配、漏叶、假枝、长度和角度差。只作因果解释和失败审计，不与Core合并计算平均数、p值或“16张总体准确率”。

## 7. 决策规则

- B在Core的主要表型误差上优于D且超过测量误差底线：淘汰增强Teacher，主线冻结B；
- D明显优于B：只讨论冻结D的收益是否足以抵偿额外Teacher复杂度，不立即恢复C/D开发；
- Teacher-direct不劣于B：重新审视Student是否有必要；
- B明显优于Teacher-direct：进入Student学习价值验证；
- 三者均不理想：停止Teacher规则修补，下一轮研究point representation或relation modeling；
- 曲率不得作为唯一生死指标。

## 8. 预计人工工作量

当前16张共有34片复核者记录的可见叶，仅用于工作量估计，正式叶数以盲法人工参考为准。

| 工作 | 最低配置 | 预计时间 |
|---|---:|---:|
| 协议培训和2张练习图 | 1次 | 20–30分钟 |
| Rater 1第一轮 | 16株、约34条路径 | 70–100分钟 |
| Rater 1第二轮 | 同上 | 70–100分钟 |
| Rater 2最低复核 | 8株、约16–18条路径 | 35–55分钟 |
| 分歧裁决与完整性检查 | 预计10–20%记录 | 30–45分钟 |
| 最低总人工量 | 约84–86条描迹 | 约3.5–5小时 |

若Rater 2完成全部16张，则约102条描迹，总人工量约4.5–6小时。GT冻结后，三种方法的计算、匹配和统计应由脚本自动完成，不再增加人工描迹工作量。

## 9. 当前等待用户确认的内容

1. 是否接受上述12张Core和4张Diagnostic；
2. 是否接受“共同地上部基点到叶尖”的路径长度，而不是严格叶片叶身长度；
3. 是否采用600 dpi元数据换算毫米，并在论文前补一次实物标尺验证；
4. Rater 2采用最低8张还是推荐的全部16张；
5. 协议通过后，是否实现新的模型盲原图描迹HTML平台。

在以上五项确认前，本协议保持 `proposal_pending_user_approval`，不开始人工GT、不运行Teacher-direct/B/D表型比较。
