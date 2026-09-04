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

V4已固定为300张整株数据：220张train、40张val和40张locked test。点条件图已替代旧的完整骨架独立拓扑，地上部有效域用于排除颖果和根须干扰。冻结A/B/C/D的40张人工配对复核已经完成：局部叶宽尺度Decoder能够稳定恢复更多真叶路径，但伴随假枝和错连代价；结构覆盖增强Teacher没有证明净收益。因此停止修补C/D，B=`Route B Teacher + local decoder`作为当前primary/minimum-sufficient Student候选，D只保留一次冻结表型对照。B尚未被证明优于更简单的Teacher-direct。

phenotype-first protocol v2、`phenotype-geometry-v1`、跨session匹配、MDC95规则和模型盲人工GT网页均已测量前锁定。Rater 1两轮均已完成并冻结：第一轮16个session、35条轨迹，第二轮16个session、36条轨迹；两轮为同一锁定样本集合，匿名编号零重合。15/16株两轮叶片轨迹数相同，1株第二轮多1条，该差异作为重复测量观测保留，不参照前一轮修正。下一步由另一位测量者完成Rater 2，再进行匿名匹配、可靠性计算和方法盲裁决。人工GT及裁决冻结前不运行Teacher-direct、Student-B或Student-D表型比较，不修改B/D模型，不启动正式消融、五随机种子或V4 test。

实验文件请从[`experiment/00_按时间线查看/`](adaptive_keypoint_discovery_reboot/experiment/00_按时间线查看/README.md)进入；原`experiment/`根目录保留为稳定执行层，避免物理移动破坏脚本路径和冻结复现性。
