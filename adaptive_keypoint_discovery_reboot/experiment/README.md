# 自适应关键点发现：当前可执行入口

本目录只回答一个问题：不给出人工关键点定义与关键点坐标标签时，能否从单株秧苗图像中发现数量可变、位置稳定、能支撑表型计算的点。

## 当前主线

1. `data_stage_clean_v3`：98 张地上部纯净图；30 张用户时期元数据图用于独立划分，68 张时期未确认图只作辅助自监督训练。
2. `g1_prime_structural_support.py`：G1′候选器，融合前景、骨架端点/分叉/曲率候选和冻结 DINOv2 证据。
3. `g1_prime_phenotype_bridge.py`：把候选点连接、排序为茎叶路径，生成自适应样条与像素/归一化表型。
4. `generate_g1prime_pseudolabels.py`：跨翻转、旋转和亮度变换筛选稳定点，生成数量可变的自动伪标签。
5. `adaptive_point_model.py` 与 `train_adaptive_point_detector.py`：冻结 DINOv2，训练动态热图头；不设置固定槽位数。
6. `evaluate_adaptive_point_detector.py`：在锁定 val/test 上评估重复性、前景命中、路径和表型输出。

## 已确认的边界

- V1/V2 是失败或候选审计版本，不用于正式训练。
- G1′点候选可以进入自动伪标签与可学习模型阶段。
- 路径阶段仍存在少量基部歧义和漏叶，不得宣称表型精度已验证。
- 当前没有物理尺度，长度只能输出像素或包围框归一化值。
- CPU 单步训练只证明代码可运行，不是实验结果。

## 先读

- `..\AGENTS.md`
- `..\PROJECT_STATE.md`
- `G1prime_V3阶段验证结论.md`
- `远程3090训练执行说明.md`

## 本机已通过的检查

```powershell
python .\g1_prime_phenotype_bridge.py --limit 0 --splits val test --device cpu --output .\outputs_g1_prime_v3\phenotype_primary11
python .\generate_g1prime_pseudolabels.py --limit 3 --device cpu --output .\pseudo_labels_g1prime_v3_smoke3
python .\train_adaptive_point_detector.py --config .\configs\train_cpu_smoke3.json --dry-run
python .\train_adaptive_point_detector.py --config .\configs\train_cpu_smoke3.json --max-steps 1
```

正式训练不要使用 `train_cpu_smoke3.json`，它只用于入口检查。
