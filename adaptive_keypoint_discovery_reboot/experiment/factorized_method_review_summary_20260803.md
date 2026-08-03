# A/B/C/D 人工配对复核统计

- 图像：40；
- 完整性：`pass`；
- 用途：只作路径语义评价，不作为关键点训练标签。

## 每图最佳方案

| 选择 | 数量 |
|---|---:|
| A | 1 |
| D | 1 |
| none | 14 |
| tie | 24 |

## 四组配对比较

- `a_vs_b_decoder`：A_better=4，B_better=7，same=29
- `a_vs_c_teacher`：A_better=5，C_better=5，same=30
- `c_vs_d_decoder`：C_better=2，D_better=6，same=32
- `b_vs_d_teacher`：B_better=4，D_better=5，same=30，uncertain=1
- `d_vs_a`：improved=11，same=22，worse=7

## 各方案错误审计

- A：漏叶下界=23，假枝下界=0，错连图=6，基部错误图=5。
- B：漏叶下界=16，假枝下界=4，错连图=10，基部错误图=5。
- C：漏叶下界=22，假枝下界=0，错连图=3，基部错误图=7。
- D：漏叶下界=16，假枝下界=3，错连图=5，基部错误图=7。
