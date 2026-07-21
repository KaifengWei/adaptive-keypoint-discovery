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

本研究当前已建立300张V4整株数据集；220张train已生成自动教师目标并完成核心模型首轮训练，40张val已完成评估。代码审计确认现有路径主要由整幅骨架独立生成，学习点尚未真正决定拓扑；关键点条件结构图、人工路径复核、表型参考误差、消融和多随机种子复跑仍未完成。V4 test继续保持模型锁定。
