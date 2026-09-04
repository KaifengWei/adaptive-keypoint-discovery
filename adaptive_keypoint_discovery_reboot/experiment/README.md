# 自适应关键点发现实验目录导航

更新时间：2026-09-04

本文件按“先做了什么、为什么保留、现在该看哪里”解释 `experiment/`。目录中并不是所有文件都代表现行方案；它同时保存了早期失败证据、V3可行性证明、V4方法开发、冻结对照和当前人工GT评价。

日常浏览请直接进入 [`00_按时间线查看/`](00_按时间线查看/README.md)。原根目录保留为执行层，避免物理搬动代码和结果后破坏脚本相对路径、Git历史及冻结复现性。

## 1. 当前只需要记住的一条工作流

```text
原始扫描图像
  -> V4整株标准化与固定划分（220 train / 40 val / 40 locked test）
  -> 地上部表型有效域 phenotype_roi_v1
  -> G1′自动教师生成数量可变的结构候选
  -> 冻结DINOv2骨干 + 可训练动态热图头学习教师目标
  -> 学习点条件图
  -> 局部叶宽尺度器官路径解码
  -> 基部—叶尖结构路径及表型
  -> 模型盲人工GT裁决
```

当前方法候选是 **B = Route B Teacher + local decoder**。增强教师方案C没有证明净收益；D只保留为冻结对照。Teacher-direct、Student-B和冻结Student-D的表型比较尚未运行，必须等人工GT全部完成、可靠性计算和裁决冻结后再运行。

V4 `test` 仍然锁定，禁止用于调参或当前人工GT pilot。

## 2. 现在真正正在做什么

当前唯一活动阶段是 `phenotype_pilot_protocol/`：

1. Rater 1第一轮16张、35条可测轨迹已完成revision修订并于2026-09-04冻结；
2. 14株保持revision 1，只有匿名顺序第8、14张进入revision 2；两处遮挡/插值字段已经明确，全包矛盾数为0；
3. 与2026-08-31原始导出相比，基点、控制点、240点重采样曲线、长度、角度和主路径均未改变，原始文件继续保留；
4. Rater 1第二轮已在4天间隔后打开，使用与第一轮零重合的新blind ID，测量期间不得查看第一轮轨迹；
5. Rater 2推荐完成16张，最低必须完成全部Core 12；
6. 两轮与第二测量者完成后，先计算intra/inter-rater可靠性、SEM和MDC95，再进行方法盲裁决；
7. 人工GT与裁决冻结后才允许正式解盲并比较Teacher-direct、Student-B和冻结Student-D。

## 3. 时间线：这些东西是怎么一步步产生的

| 时间 | 阶段 | 主要目录/文件 | 现在的身份 |
|---|---|---|---|
| 7月16日 | G0/G1可行性、图片质量排查、四类候选比较 | `outputs_g1*`、`data_quality_control`、`data_clean_core20`、四类方法HTML/摘要 | 历史证据；当前流程不读取 |
| 7月17日 | V3 98张数据、G1′自动教师、首轮DINOv2热图训练 | `data_stage_clean_v3`、`pseudo_labels_g1prime_v3`、`core_dinov2*` | 证明“自动教师能被学习”；不是当前正式数据 |
| 7月17日 | 旧拓扑桥接和topology v2 | `g1_prime_phenotype_bridge.py`、`reevaluate_topology_from_points.py`、`topology_reuse_*` | 暴露“点没有真正决定拓扑”；历史对照 |
| 7月18日 | 构建V4整株数据300张 | `data_stage_clean_v4_fullplant_candidate`、V4构建/审计脚本、source manifest | 当前唯一正式数据入口 |
| 7月18–21日 | V4整株直接训练、点条件图v1 | `core_dinov2_v4_fullplant*`、`point_conditioned_graph_v1_val` | 发现颖果/根须干扰；保留为旧基线 |
| 7月22日 | 地上部有效域与shoot侧基部，形成路线B | `phenotype_roi_basal_anchor*`、`core_dinov2_v4_phenotype_roi*`、graph/path v2 | 当前最小充分Student候选的来源 |
| 7月23日 | 结构覆盖增强教师与局部decoder二维拆分 | structure-coverage伪标签/训练/graph/path v3 | 形成A/B/C/D冻结对照 |
| 7月25日–8月3日 | 人工复核A/B/C/D并做因果分析 | `manual_review_system`、factorized结果、A/B/C/D报告 | 已完成；结论是保留B、停止修补C/D |
| 8月3日至今 | phenotype-first模型盲人工GT | `phenotype_pilot_protocol` | 当前活动阶段 |

## 4. 顶层目录逐一说明

### A. 当前关键资产：不能删除

| 目录 | 是什么 | 当前用途 |
|---|---|---|
| `data_stage_clean_v4_fullplant_candidate/` | 300张V4整株白底图、原图复核图、shoot/seed-base-root/full-plant掩膜和固定划分 | 当前唯一正式数据集；test继续锁定 |
| `phenotype_pilot_protocol/` | GT协议、几何规则、MDC95规则、网页、盲包生成器、验证和本地runtime | 当前正在执行的人工GT主目录 |
| `third_party/` | DINOv2上游源码及预训练权重 | 训练和推理依赖；Git忽略但本机运行需要 |
| `configs/` | 训练配置 | 保留V4 full-plant、phenotype-ROI和structure-coverage复现配置；消融配置目前未执行 |
| `training_outputs/` | 训练摘要、曲线、配置以及本地checkpoint | B/D训练证据；正式checkpoint主要保存在远程，不要整体删除 |
| `evaluation_outputs/` | 各代模型点、图、路径、叠加图、CSV和summary | 当前及历史对照的主要结果库，论文证据来源 |

### B. 当前冻结方法链：现在不修改，GT冻结后还要使用

| 文件/目录 | 作用 |
|---|---|
| `g1_prime_structural_support.py` | 当前G1′结构候选/自动教师核心 |
| `generate_g1prime_pseudolabels.py` | 生成跨增强一致的教师目标 |
| `adaptive_point_model.py` | DINOv2特征后的动态热图模型定义 |
| `train_adaptive_point_detector.py` | 学生模型训练入口 |
| `evaluate_adaptive_point_detector.py` | 学习点稳定性和前景支持评估 |
| `phenotype_roi_basal_anchor.py` | 排除颖果/根须并定义shoot侧基部资格 |
| `point_conditioned_graph.py` | 只以学习点作为图节点，骨架只提供节点间测地边 |
| `point_conditioned_organ_paths.py` | 从点条件图解码地上部器官路径 |
| `evaluate_*`、`test_*` | 相应评估入口和单元测试；用于复现与防止改坏冻结逻辑 |
| `run_remote_v4_phenotype_roi_train_val.sh` | 路线B远程复现入口 |
| `run_remote_v4_structure_coverage_train_val.sh` | C/D冻结对照远程复现入口 |

### C. `evaluation_outputs/` 中A/B/C/D到底对应什么

| 代号 | Teacher/Student来源 | Decoder | 结果目录 | 身份 |
|---|---|---|---|---|
| A | Route B Teacher训练出的phenotype-ROI Student点 | global decoder | `point_conditioned_organ_paths_v2_phenotype_roi_val/` | 冻结拆分对照 |
| B | 与A相同 | local decoder | `point_conditioned_organ_paths_v2_phenotype_roi_local_decoder_val/` | 当前primary/minimum-sufficient Student候选 |
| C | structure-coverage增强教师训练出的Student点 | global decoder | `point_conditioned_organ_paths_v3_structure_coverage_global_decoder_val/` | 冻结拆分对照 |
| D | 与C相同 | local decoder | `point_conditioned_organ_paths_v3_structure_coverage_val/` | 仅保留一次冻结表型对照，不再开发 |

相关上游目录：

- `core_dinov2_v4_phenotype_roi_val/`：A/B共用的Student点；
- `point_conditioned_graph_v2_phenotype_roi_val/`：A/B共用的点条件图；
- `core_dinov2_v4_structure_coverage_val/`：C/D共用的Student点；
- `point_conditioned_graph_v3_structure_coverage_val/`：C/D共用的点条件图。

`Teacher-direct`不是A。它指G1′自动教师点直接进入与B完全相同的冻结graph/local decoder/几何算子。正式Teacher-direct表型输出尚未生成，必须等GT冻结后运行。

### D. 数据与伪标签版本

| 目录 | 含义 | 当前状态 |
|---|---|---|
| `data_clean_core20/` | 早期干净背景20张及smoke6/locked14拆分 | 历史小样本，不再训练 |
| `data_quality_control/` | 背景质量假设的接触表 | 历史审计 |
| `data_source_reaudit/` | 2,239张来源图片盘点和候选筛选 | 数据来源证据，不是训练输入 |
| `data_stage_source_audit/` | V3时期分组的接触表 | 历史审计 |
| `data_stage_clean_v3/` | 98张V3正式集 | 早期学习可行性复现；当前不用 |
| `pseudo_labels_g1prime_v3_smoke3/` | V3三张代码冒烟教师目标 | 冒烟产物 |
| `pseudo_labels_g1prime_v3/` | V3 87张正式自动教师目标 | 历史可行性证据 |
| `pseudo_labels_g1prime_v4_fullplant/` | V4整株基线教师目标 | 旧基线，证明根须/颖果问题 |
| `pseudo_labels_g1prime_v4_phenotype_roi/` | 路线B的地上部有效域教师目标 | 当前B来源，保留 |
| `pseudo_labels_g1prime_v4_structure_coverage/` | 增强教师目标 | C/D冻结对照，保留但不继续开发 |

### E. 早期输出目录：当前代码不会读取，但不是毫无价值

| 目录 | 说明 | 建议 |
|---|---|---|
| `outputs_g1/` | 最初DINOv2响应、局部峰、聚类等G1试验 | 只作研究路线证据，可归档 |
| `outputs_g1_prime/` | 最初6张G1′结构候选 | 只作早期效果图，可归档 |
| `outputs_g1_prime_v3/` | V3 11张路径和表型桥接 | 只作旧链路失败证据，可归档 |
| `outputs_clean_smoke6_baselines/` | 干净背景下三种非G1′基线 | 只作四方法对比证据，可归档 |
| `outputs_clean_smoke6_g1prime/` | 干净背景G1′结果 | 只作早期选择G1′的依据，可归档 |
| `manual_review_system/` | 有效域、路径和A/B/C/D人工复核平台及分析脚本 | 已完成的人工决策证据；当前GT不用它，但论文方法选择需要 |
| `training_logs/` | V3首轮及恢复训练日志 | 历史复现证据，体积很小，保留 |

“可归档”不等于“没有价值”。这些目录不参与当前命令，但保存了为什么放弃早期方案、为什么采用地上部有效域、为什么最终选B的证据链。

## 5. 顶层脚本按执行逻辑分组

### 5.1 早期数据与质量试验（历史）

`g0_build_dataset.py`、`analyze_pilot_data_quality.py`、`build_quality_control_subsets.py`、`audit_dataset_images.py`、`build_clean_core20.py`、`run_clean_smoke6_baselines.py`、`run_clean_smoke6_g1prime.py`。

它们只复现7月16日的质量假设和小样本方法筛选，不应作为当前正式实验入口。

### 5.2 V3可行性证明（历史）

`audit_stage_source.py`、`build_stage_clean_dataset.py`、`finalize_stage_clean_v3.py`、`g1_dinov2_feasibility.py`、`g1_prime_phenotype_bridge.py`、`reevaluate_topology_from_points.py`、`run_remote_core.sh`。

它们证明自动教师目标能被DINOv2热图模型学习，并暴露旧拓扑独立于学习点的问题。

### 5.3 V4当前数据入口

`build_stage_clean_v4_fullplant.py`、`audit_stage_clean_v4_fullplant_candidate.py`、`v4_fullplant_source_manifest.csv`。

这三项定义当前300张V4数据及220/40/40划分。`make_v4_val_review_assets.py`和`audit_v4_fullplant_val_regions.py`只用于当时的视觉复核。

### 5.4 当前冻结方法与复现实验

`g1_prime_structural_support.py`、`generate_g1prime_pseudolabels.py`、`adaptive_point_model.py`、`train_adaptive_point_detector.py`、`evaluate_adaptive_point_detector.py`、`phenotype_roi_basal_anchor.py`、`point_conditioned_graph.py`、`point_conditioned_organ_paths.py`及对应`evaluate_*`和`test_*`。

当前人工GT阶段禁止修改它们。GT冻结后只运行冻结输出，不边测边改。

### 5.5 远程运行与环境

`setup_remote_env.sh`、`run_remote_v4_fullplant_train_val.sh`、`run_remote_v4_phenotype_roi_train_val.sh`、`run_remote_v4_structure_coverage_train_val.sh`、`远程3090训练执行说明.md`。

full-plant脚本是旧基线复现；phenotype-ROI与structure-coverage分别对应B和C/D的复现。

## 6. 哪些内容现在完全不应当拿来做结论

- `四类方法关键技术与效果对比.html`和早期摘要：教学/方案筛选材料，不是正式论文最终比较；
- V3的11张val/test结果：只证明链路可运行，不能代替V4最终评价；
- `graph_success=1.0`：只表示程序返回图，不表示叶片语义正确；
- `manual_phenotype_reference_pending.csv`：旧自动路径人工模板，不是当前模型盲GT；
- 80轮训练的单一最佳loss：不能单独证明最终模型充分收敛；
- A/B/C/D的路径数：只能解释decoder和teacher的因果作用，不能代替人工表型误差；
- 已冻结的Rater 1第一轮GT：单独一轮仍不能计算MDC95，也不能单独裁决方法。

## 7. 真正可以安全清理的东西

当前能够确定“删除不会损失科学证据”的只有：

1. `experiment/__pycache__/`；
2. `experiment/manual_review_system/__pycache__/`；
3. `training_outputs/cpu_smoke3/`中的本地一轮CPU冒烟checkpoint `best.pt`，约85 MB；它的训练摘要已明确标记为`cpu_code_smoke3_not_scientific_training`，当前模型不读取它；
4. `training_outputs/cpu_smoke3_eval2/`的CPU冒烟评估图和表，约0.3 MB；
5. 重复打包文件，例如已有展开目录时的早期`overlays.zip`。

其余“历史目录”大多只有几MB到十几MB，且记录方法为什么失败或为什么被替代。建议先逻辑归档，不直接删除。未经用户明确确认，不移动V4数据、教师目标、B/D训练摘要、evaluation outputs、第三方DINOv2权重、人工原始GT或任何远程checkpoint。

## 8. 以后最快的查找方式

- 看当前项目事实：上级 `PROJECT_STATE.md`；
- 看整个实验目录：本文件；
- 看当前人工GT：`phenotype_pilot_protocol/`；
- 看正式数据：`data_stage_clean_v4_fullplant_candidate/`；
- 看B/D及历史方法结果：`evaluation_outputs/`；
- 看A/B/C/D结论：`A_B_C_D人工配对复核与因果分析_20260803.md`；
- 看冻结方法边界：`当前研究路线证据审计与下一阶段冻结计划_20260802.md`；
- 看远程执行：`远程3090训练执行说明.md`。

## 9. 下一步顺序

```text
R1第一轮revision修订与冻结（已完成）
  -> 在不查看第一轮记录的条件下完成R1第二轮（当前）
  -> Rater 2完成16张（最低Core 12）
  -> 匿名跨session匹配
  -> intra/inter-rater可靠性、SEM、MDC95
  -> 方法盲分歧裁决并冻结GT
  -> 管理解盲
  -> 运行冻结Teacher-direct / Student-B / Student-D表型比较
  -> 再决定正式消融、多随机种子和何时解除V4 test锁定
```
