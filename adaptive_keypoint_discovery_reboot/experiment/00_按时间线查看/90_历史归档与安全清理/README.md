# 90 历史归档与安全清理

## 可直接清理

- `../../__pycache__/`
- `../../manual_review_system/__pycache__/`
- `../../training_outputs/cpu_smoke3/best.pt`：约85 MB，仅一轮CPU代码冒烟，不是科学训练；
- `../../training_outputs/cpu_smoke3_eval2/`：CPU冒烟评估；
- 已有展开目录时的重复旧ZIP。

## 当前不执行，但应保留为证据

- `outputs_g1*`和`outputs_clean_smoke6*`：早期候选与数据质量结论；
- `data_stage_clean_v3/`、`pseudo_labels_g1prime_v3/`：V3可学习性证明；
- V4 full-plant训练和评价：根须/颖果干扰基线；
- A/B/C/D四组评价输出及人工复核：方法选择与消融证据；
- 所有阶段Markdown报告：失败结论和路线变更证据。

## 禁止删除

- V4正式数据与split清单；
- Route B和structure-coverage教师、训练摘要、评价输出；
- DINOv2源码与预训练权重；
- 远程正式checkpoint；
- 人工GT原始导出、管理员映射和冻结哈希；
- `PROJECT_STATE.md`、协议和冻结规则。

物理归档或删除前必须另做Git状态、引用和体积审计，不能只按文件名判断。
