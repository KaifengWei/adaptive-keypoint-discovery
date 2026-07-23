# 自适应关键点发现：当前可执行入口

本目录只回答一个问题：不给出人工关键点定义与关键点坐标标签时，能否从单株秧苗图像中发现数量可变、位置稳定、能支撑表型计算的点。

## 当前主线

1. `data_stage_clean_v4_fullplant_candidate`：当前唯一 V4 图像集，300 张整株白底图及三类器官掩膜；test 已通过视觉复核，但仍禁止模型读取。
2. `v4_fullplant_source_manifest.csv`：当前 V4 的样本身份、来源与冻结划分，不再依赖 shoot-only 数据目录。
3. `build_stage_clean_v4_fullplant.py` 与 `audit_stage_clean_v4_fullplant_candidate.py`：整株重建和数据审计入口。
4. `g1_prime_structural_support.py`：G1′自动教师候选器；V4 后续只对 train 生成教师目标。
5. `adaptive_point_model.py` 与 `train_adaptive_point_detector.py`：冻结 DINOv2，训练动态热图头；不设置固定槽位数。
6. `g1_prime_phenotype_bridge.py` 与 `reevaluate_topology_from_points.py`：现有路径、样条和表型桥接；代码审计确认路径主要由整幅骨架独立生成，学习点尚未真正决定拓扑。
7. `phenotype_roi_basal_anchor.py`：完整整株留档与地上部表型输入分离；`phenotype_roi_v1`排除颖果/根须干扰，`basal_transition`只约束已有学习节点的基部资格。
8. `point_conditioned_graph.py` 与 `point_conditioned_organ_paths.py`：学习点决定图节点与器官路径；路线B下基部只能从shoot侧过渡区附近的已有学习节点中选择。

## 目录怎么找

- 当前数据：只看 `data_stage_clean_v4_fullplant_candidate`。
- 当前事实：先看上级 `PROJECT_STATE.md`。
- 当前数据说明：看 `V4数据集构建与锁定说明_20260717.md`。
- 既有 GPU 训练产物：看 `training_outputs` 和 `evaluation_outputs`。
- 人工复核协作、判定标准与离线资料包构建：看 `manual_review_system`。
- V4 首轮结果与下一步路线：看 `V4整株首轮训练与架构审计_20260718.md`。
- V4 val 全量复核图和待填写表：看 `evaluation_outputs/core_dinov2_v4_fullplant_val`。
- V3 只用于复现已记录的首轮结果，不再作为下一轮数据入口。

旧 shoot-only、V1/V2、V3 临时候选、V4 pre-gate、smoke 和 full-plant failed 目录均已清理；失败结论只保留在 Markdown 和 Git 历史中。

## 已确认的边界

- G1′点候选可以进入自动伪标签与可学习模型阶段。
- 当前路径程序可返回连通图，但学习点尚未决定路径拓扑，不得把 `graph_success=1.0` 写成关键点链路已通过。
- 当前没有物理尺度，长度只能输出像素或包围框归一化值。
- V4 test 仅完成视觉复核并保存，尚未授权模型评价。

## 先读

- [项目核心概念与论文术语说明](../项目核心概念与论文术语说明.md)
- `..\AGENTS.md`
- `..\PROJECT_STATE.md`
- `V4数据集构建与锁定说明_20260717.md`
- `远程3090训练执行说明.md`

## 当前数据检查

```powershell
python .\audit_stage_clean_v4_fullplant_candidate.py
python .\build_stage_clean_v4_fullplant.py --refresh-contact-sheets-only
```

V4 full-plant基线与点条件图v1已经完成。用户已确认路线B并通过40/40张val有效域视觉门槛；`run_remote_v4_phenotype_roi_train_val.sh`已在RTX3090完成216张train教师、80轮训练和40张val-only评估。新路线人工路径审核已完成：基部和错连显著改善，但严格联合通过仅22/40，漏叶仍为8张。当前建议与证据见`路线B人工路径复核与下一轮改进建议_20260723.md`；在用户确认自动教师和短枝解码改动前，不进入人工表型误差、消融、五随机种子或test。任何模型命令都不得包含 V4 test。
用户已于2026-07-23确认下一轮实施“结构覆盖增强自动教师 + 局部尺度短枝解码”。在新教师、训练和val双人复核完成前，仍不进入人工表型误差、消融、五随机种子或test。任何模型命令都不得包含V4 test。
