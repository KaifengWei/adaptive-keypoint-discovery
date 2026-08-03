# Phenotype-first 人工参考协议 v2（已批准，待实现描迹平台）

- 协议版本：`phenotype-first-pilot-v2`
- 修订日期：2026-08-03
- 数据范围：V4 val 的锁定 Core 12 + Diagnostic 4
- 方法范围：Teacher-direct、Student-B、冻结的 Student-D
- 当前状态：协议、schema 与 HTML 设计已冻结；正式人工描迹尚未开始
- 禁止事项：修改 Teacher/Student/Decoder、新训练、调阈值、方案 E、V4 test、五随机种子、查看方法输出后修改人工 GT

## 1. 研究目的与不可改变的边界

本 pilot 只回答两个问题：Student-B 相对 Teacher-direct 是否带来可测量的表型价值；冻结 Student-D 相对 Student-B 的额外复杂度是否带来足以保留的收益。人工描迹只用于评价，不进入自动教师、学生训练、关键点坐标监督或逐图调参。

Core 12 形成主要 pilot 统计；Diagnostic 4 只解释已冻结的 B/D 分歧和失败机制，不与 Core 合并为“16 张总体平均值”。曲率继续保持 exploratory，不作为任何方法的单独淘汰标准。

## 2. 锁定样本

本次不重新依据模型表现选样。以下集合永久沿用 proposal-v1 的锁定结果。

| 分组 | dataset_id |
|---|---|
| Core 12 | `v4_val_0004, v4_val_0005, v4_val_0006, v4_val_0009, v4_val_0011, v4_val_0016, v4_val_0020, v4_val_0027, v4_val_0028, v4_val_0030, v4_val_0035, v4_val_0037` |
| Diagnostic 4 | `v4_val_0002, v4_val_0018, v4_val_0023, v4_val_0034` |

既有 `visible_leaf_count_postlock` 仅保留在管理员选样审计表中，用于解释既往选样覆盖和估算工作量。它不得进入测量者页面、图片文件名、浏览器脚本、下载包或人工原始记录。

## 3. 人工表型的操作性定义

### 3.1 可见叶与动态建轨

测量者在无任何预期叶数提示的原图上先独立判断可见叶。每发现一片叶，使用“添加叶片”创建一条空 trace；系统不得预先生成 GT 行，也不得预填 `leaf_exists=yes`。

一片可见叶应有可区分叶尖，并能从叶尖沿绿色组织中心走向追踪至共同地上部路径或明确的分化处。颖果、根、根须、扫描阴影和断裂杂物不计作叶。若叶片存在但无法可靠连续描迹，记录为 `visible_unmeasurable`，进入存在/漏叶统计，但不进入连续长度、角度或曲率误差。

遮挡间隙不超过 shoot 外接框对角线 5%，且两端位置与方向形成唯一平滑延续时，允许插值并记录 `interpolation_used=yes`；否则标为不可测，不主观补线。

### 3.2 正式长度名称与路径端点

本研究的正式长度名称为：

> **base-to-tip structural path length / 基部—叶尖结构路径长度**

它从共同的 `shoot-side basal transition` 沿植株中心结构走到相应可见叶尖。多叶植株的不同路径可以共享起点后的地上部主路径段，直至各自分化。因此，论文、图表、代码输出和统计字段中禁止把它直接简称为 `leaf length / 叶长`。

未来可另行增加 `divergence-to-tip branch length`，但本 pilot 不增加该人工描迹负担。

### 3.3 三套长度尺度

每条可测路径必须同时保留：

1. `structural_path_length_px`：输出图像像素弧长；
2. `structural_path_length_bbox_norm = structural_path_length_px / shoot_bbox_diagonal_px`；
3. `structural_path_length_mm_metadata`：依据扫描元数据换算的毫米值。

7 个源 TIFF 的元数据均为 600 dpi，标准化图像经历缩放，因此：

`mm_per_output_px = 25.4 / (600 × resize_scale)`。

在已知尺寸实物标尺校验完成前，任何毫米数值、表头和图注都必须附注：`derived from scanner metadata / 基于扫描元数据换算`。正式论文前必须完成一次已知尺寸标尺校验，并记录实测尺寸、像素尺寸、误差和校准日期；校验不得追溯性改变像素原始记录。

### 3.4 reference_main_path 的确定性规则

`reference_main_path` 不由测量者主观勾选，而在一次提交内由软件对全部可测 trace 统一确定：

1. 优先选择 `structural_path_length_px` 最大的路径；
2. 若候选路径与最大值的相对差不超过 1%，选择基点到叶尖欧氏弦长较大的路径；
3. 若弦长差不超过 `max(1 px, 0.001 × shoot_bbox_diagonal_px)`，选择以共同基点为原点、图像正 x 轴为 0°、顺时针增加时极角较小的叶尖；
4. 若仍完全相同，按提交前生成的随机 `trace_uuid` 字典序升序取第一条。

该规则固定后不得由 Rater 1/2 手工覆盖。主路径本身的分化角为 `not_applicable`，不得以 0°进入角度误差。

### 3.5 角度与曲率

对每条非主路径，分化点为其与主路径最后共享的中心线位置；两侧局部切线区间默认取 `0.05 × shoot_bbox_diagonal_px`，不足时缩短并记录实际区间。分化角为两条切线的无符号夹角，范围 0–180°。

曲率仅作探索性指标。人工与方法路径使用相同平滑和弧长重采样规则；报告总转角，并可计算平均绝对曲率。曲率不进入单独淘汰门槛，未完成实物标尺校验前其 `mm^-1` 结果同样注明 `derived from scanner metadata`。

### 3.6 提交后 GT 身份

提交前页面中只存在随机 `trace_uuid`，不显示 `GT01/GT02...`。提交并锁定该株全部 trace 后，才以共同基点为中心按叶尖顺时针极角排序并赋予 `GT01、GT02...`；完全重合时以 `trace_uuid` 破同值。GT 身份不依赖任何模型输出或旧可见叶数。

## 4. 真正的盲法与数据边界

### 4.1 opaque blind ID

正式包构建时为“图像 × 测量者 × 轮次”分别生成不可推断的随机 blind ID：使用密码学安全随机源，从去歧义字符集生成至少 12 位标识，并做全包唯一性检查。不同测量者和 Rater 1 两轮不得复用同一 blind ID，图片文件也重命名为 blind ID。

`blind_id → dataset_id` 映射只写入管理员映射表。管理员映射表不得进入测量者下载包、HTML 源码、浏览器本地存储、人工原始 CSV/JSON 或协作仓库的测量分发目录。测量页面及其原始导出均不得包含 `dataset_id`、`pilot_group`、旧叶数或任何方法标识。

### 4.2 解盲时点

各测量轮次提交后先冻结原始 CSV/JSON、图片清单和 SHA-256。为计算重复性和制作方法盲裁决页面，管理员脚本可以在后台把同一原图的不同 blind session 配成匿名组，但不得向测量者、裁决者或中间导出暴露 `dataset_id`；这一步只叫“匿名技术配对”，不叫正式解盲。Rater 1 第二轮与 Rater 2 全部结束、分歧裁决完成并冻结最终人工 GT 后，才允许把 `dataset_id` 写入解盲GT表，随后运行 Teacher-direct、Student-B、Student-D 的自动匹配与统计。任何方法图、路径、指标或旧人工路径结论都不得用于修改已经冻结的人工 GT。

## 5. 人工测量安排

1. 使用不属于 16 张 pilot 的两张开发图培训；练习不进入统计。
2. Rater 1 完成全部 16 张第一轮。
3. 间隔 3–7 天，Rater 1 使用全新 blind ID 和重新随机的顺序完成全部 16 张第二轮，且不能查看第一轮轨迹。
4. Rater 2 推荐完成全部 16 张；现实工作量受限时，最低也必须完成全部 Core 12。原“6 Core + 2 Diagnostic”方案废止。
5. 首轮检查仅检查字段完整性、轨迹是否明显越出植株和文件可读性，不查看方法结果。
6. 长度差超过 5%、角度差超过 5°、叶片存在或身份不一致时，在保持方法盲的条件下进行裁决。原始记录不覆盖，另建 `adjudicated` 记录。

旧叶数 34 只用于粗略工时估计，不能作为测量页面的进度目标。预计 Rater 1 两轮约 68 条动态描迹；Rater 2 完成 Core 12 约增加 25 条，推荐全 16 约增加 34 条。实际条数以盲法动态创建结果为准。

## 6. 测量误差底线与 MDC95（测量前锁定）

长度和角度必须分别计算，不得合并成一个分数。只使用成功匹配、两次均可测的同一人工 trace 对；缺失值不插补，按 endpoint 成对排除并报告有效对数。置信区间以植株为重采样簇。

设同一对象的两次测量差为 `d_i = x_i2 - x_i1`：

- `bias = mean(d_i)`；
- `SD_diff = sample SD(d_i)`；
- `SEM = SD_diff / sqrt(2)`；
- `MDC95 = 1.96 × sqrt(2) × SEM = 1.96 × SD_diff`；
- Bland–Altman 一致性限为 `bias ± 1.96 × SD_diff`。

### 6.1 intra-rater repeatability

使用 Rater 1 第二轮减第一轮。分别计算：

- 基部—叶尖结构路径长度的像素、bbox-normalized、metadata-derived mm 和对称相对差 `100 × (x2-x1)/mean(x1,x2)`；
- 分化角（度）。

同时报告双向混合、绝对一致、单次测量的 `ICC(A,1)` 及 95% CI，作为重复性描述；MDC95 的决策计算仍固定采用差值 SD 公式。

### 6.2 inter-rater reliability

使用 Rater 2 第一轮减 Rater 1 第一轮。主分析至少覆盖全部 Core 12；若 Rater 2 完成 16 张，则另报告 Diagnostic 描述结果，但 Core 统计保持独立。长度与角度采用与 intra-rater 相同的 SEM、MDC95、Bland–Altman 和成对缺失规则。

同时报告双向随机、绝对一致、单次测量的 `ICC(A,1)` 及 95% CI。用于后续方法差异判定的人工误差底线，按 endpoint 取 `max(MDC95_intra, MDC95_inter)`。

`SD_pooled × sqrt(1-ICC)` 仅允许作为预先声明的敏感性分析，不能替代上述主 MDC95 公式，也不得测完后择优选择。曲率只报告探索性重复性描述，不设 MDC95 方法淘汰门槛。

## 7. 数据表分层

1. **测量者原始层**：只含 opaque blind ID、测量者、轮次、动态 trace、像素坐标、像素与 bbox-normalized 结果以及状态字段；无 dataset_id、分组、方法字段和旧叶数。
2. **管理员映射层**：只含 blind ID 与 dataset_id、测量者、轮次、随机顺序及包标识；独立保存，不发给测量者。
3. **解盲人工 GT 层**：GT 冻结后才合并 dataset_id、分组、扫描元数据换算和裁决记录。
4. **方法评价层**：GT 冻结并解盲后生成 Teacher-direct/B/D 的匹配和误差；不得反写人工原始层。

具体字段、允许值和泄漏检查见同目录工作簿与《模型盲人工GT_HTML平台设计说明_20260803.md》。

## 8. 方法比较与决策纪律

Core 12 以植株为统计单位，报告中位数、四分位数、逐株配对差、精确 Wilcoxon 或配对置换辅助结果，以及按植株 cluster bootstrap 的 95% CI。必须同时报告 GT 叶召回率、预测路径精确率、missed 和 extra，不能只在成功匹配路径上报告误差。

方法差异只有在超过对应 endpoint 的人工误差底线 `max(MDC95_intra, MDC95_inter)` 时，才可被解释为超出人工重复测量噪声。Diagnostic 4 逐例报告，不参与 Core 平均值和显著性检验。

曲率始终是 exploratory；不得因单独曲率表现淘汰 Teacher-direct、B 或 D。

## 9. 当前实施门槛

本协议 v2 已吸收用户九项修订并锁定。下一步只能先展示本次协议、schema 与 HTML 设计修改摘要。收到用户继续指令后，方可实现模型盲原图描迹 HTML；实现完成并通过静态泄漏检查和离线浏览器验收后，才开始正式人工描迹。
