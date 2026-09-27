# 表型驱动的自适应关键点发现

本仓库保存当前秧苗自适应关键点发现小论文的可复现实验工程。唯一研究问题是：在没有人工关键点定义和坐标标签的情况下，算法能否根据单株秧苗的结构与形态，自主确定用于表型计算的关键点位置、数量及稳定性。

## 主入口

- [项目核心概念与论文术语说明](adaptive_keypoint_discovery_reboot/项目核心概念与论文术语说明.md)：大白话解释V1—V4、G0/G1/G1′、DINOv2、当前模型和论文正式术语；适合沟通汇报与快速查询。
- `AGENTS.md`：仓库根级自动约束，确保新账号或新任务先恢复研究边界。
- `adaptive_keypoint_discovery_reboot/AGENTS.md`：研究边界与不可偏离的口径。
- `adaptive_keypoint_discovery_reboot/ACCOUNT_HANDOFF.md`：跨账号、跨设备和新任务的强制接续顺序。
- `adaptive_keypoint_discovery_reboot/PROJECT_STATE.md`：当前数据、代码、结果和下一步状态。
- `adaptive_keypoint_discovery_reboot/experiment/`：数据清单、候选生成、路径重建、表型输出及训练代码。
- `adaptive_keypoint_discovery_reboot/远程算力检查与Codex连续工作说明.md`：远程 GPU 执行说明。

## Git 数据边界

仓库保留代码、配置、论文说明、审计表、关键可视化结果，以及体量较小且已复核的 `data_stage_clean_v3`。原始大数据集、旧版处理数据、第三方 DINOv2 源码、预训练权重、训练 checkpoint、缓存和隔离文件不进入 Git；它们可以按项目清单在远程服务器重新获取或单独同步。

## 当前进度

截至2026-09-27，V4固定为220 train / 40 val / 40 locked test。16株val pilot人工GT和冻结方法比较已完成；Student-B第53轮因CONVERGED正式冻结，方法门槛为B：快速learned surrogate，未证明表型优于Teacher-direct。唯一Pipeline V2开发裁决为明显恶化，永久回退并冻结V1；val方法开发和新训练均关闭，Teacher-direct为强参照，Student-D仅secondary对照。

当前V4 locked-test协议已正式行政修订为**single-rater model-blind full-40 GT**：一人完成全部40株存在性和所有可测几何，不进行test双人共识或重估inter-rater可靠性。两张纯合成图的平台验证通过后，唯一正式测量包已准备完成（40/40资产和泄漏检查PASS）。用户标注仍待完成，`test_model_reads=0`、`INFERENCE_GATE=CLOSED`；只有raw验证→GT冻结/hash→evaluator冻结→显式授权后才可第一次模型推理。完整登记见[单人GT行政修订](adaptive_keypoint_discovery_reboot/experiment/locked_test_final_20260927/V4_LOCKED_TEST_GT_SINGLE_RATER_AMENDMENT.md)、[测量包审计与本机入口](adaptive_keypoint_discovery_reboot/experiment/locked_test_final_20260927/V4_TEST_GT_SINGLE_RATER_PACKAGE_AUDIT.md)和[PROJECT_STATE](adaptive_keypoint_discovery_reboot/PROJECT_STATE.md)。私有测量包、映射和raw记录不进入Git。

实验文件请从[`experiment/00_按时间线查看/`](adaptive_keypoint_discovery_reboot/experiment/00_按时间线查看/README.md)进入；原`experiment/`根目录保留为稳定执行层，避免物理移动破坏脚本路径和冻结复现性。
