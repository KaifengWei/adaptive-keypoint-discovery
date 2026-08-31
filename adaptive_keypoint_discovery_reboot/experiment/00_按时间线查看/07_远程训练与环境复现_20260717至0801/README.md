# 07 远程训练与环境复现

## 主要入口

- [远程3090训练执行说明](../../远程3090训练执行说明.md)
- [迁移到新RTX3090服务器记录](../../迁移到新RTX3090服务器_执行记录_20260801.md)
- 环境脚本：`../../setup_remote_env.sh`
- V3旧训练：`../../run_remote_core.sh`
- V4整株旧基线：`../../run_remote_v4_fullplant_train_val.sh`
- 路线B：`../../run_remote_v4_phenotype_roi_train_val.sh`
- C/D结构增强对照：`../../run_remote_v4_structure_coverage_train_val.sh`
- 当前训练入口：`../../train_adaptive_point_detector.py`
- 训练配置：`../../configs/`
- 训练日志：`../../training_logs/`
- 训练摘要：`../../training_outputs/`

## 当前边界

默认训练机是`cv`。`kf3090`已封存。当前人工GT阶段不启动新训练；V4 test锁定不因设备变化而解除。
