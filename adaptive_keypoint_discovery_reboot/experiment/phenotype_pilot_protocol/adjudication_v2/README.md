# 两阶段人工分歧裁决（2026-09-06）

**最新状态（2026-09-24）：最终GT已按用户确认的可见独立结构路径定义冻结。** 下文2026-09-15页面说明保留为历史过程，不再要求填写任何新复核表。唯一现行接续见`../最终GT冻结与后续比较入口_20260924.md`；私有解盲GT位于`../runtime/final_gt/phenotype_pilot_20260924_v1/`，生成脚本为本目录`freeze_final_gt.py`。原第7张第3条仅有结构存在性GT，几何曲线缺失，不得补造。

这是人工参考的裁决工具，输入仅为13张既定分歧样本的标准化图像和三轮冻结人工记录。软件不调用任何模型，不生成最终GT，不改动描迹平滑、匹配或表型计算算法。

## 当前正式入口（2026-09-15简化后）

第二位测量者两阶段13张真实人工JSON/CSV均已校验归档。经表型等效复审，6张完全一致、4张差异小于锁定的长度/分化角人工误差底线，真正需要人工确认的只有原复核第5、6、7张。正式填写入口改为：

`../runtime/adjudication/semantic_first_20260906/minimal_gt_resolution_20260915_v1/index.html`

必须分发整个 `minimal_gt_resolution_20260915_v1` 文件夹，不能只发HTML。页面不再展示A/B/C，每张只提出一个实际问题：第5张确认3片或4片，第6张确认小结构是真叶还是毛边干扰，第7张确认争议短结构是否为独立叶片。完成3张后下载 `minimal_gt_resolution.json` 和同名CSV，一起保存到包内 `results/`。

此前的13张复杂填写包 `joint_consensus_20260915_v2` 已停止作为正式入口，只保留审计痕迹。自动等效依据、唯一代表路径规则和完整数值见 `../最终GT最小复核与自动等效审计_20260915.md`。新导出不会修改候选曲线，也不会自动运行模型或冻结GT。

用户已完成的13张裁决完整保存在 `../runtime/adjudication/opinions/rater1_20260905/`，身份为“第一位测量者的独立裁决意见”，不作废，不要求用户重新画三轮。

## 两个步骤

1. 第一包只看标准化输入图，逐叶或逐个疑似结构添加定位圈，判断身份、完整可测性、原因、信心；无需重画中心线。不预填数量，不在文件中放置任何候选轨迹。提交全部13张后，分别下载 `semantic_decisions.json`、`semantic_decisions.csv`，保存到本包 `results/`。
2. 管理员校验并归档第一步原始导出后，才生成第二包。第二包展示已记录结构和随机重标的A/B/C曲线，按结构选择具体路径、重描或保留争议。第一步的判断只读，新发现写在第二步备注中。

第2步支持逐叶提出混合候选建议，但混合不同原始记录的曲线会带来共同基点不一致的可能，因此输出仍须共识检查；不能直接拼接为最终GT。

## 管理员命令

在本目录执行，Python需安装Pillow。既有分发包禁止覆盖；需要新版时使用独立输出目录并保留上一版清单。

```powershell
python build.py archive-opinion
python build.py semantic
python validate.py --package ../runtime/adjudication/semantic_first_20260906/semantic

# 收到真实人工JSON和同名CSV后执行；现在不能用测试文件代替真实结果。
python build.py geometry --semantic-export ../runtime/adjudication/semantic_first_20260906/semantic/results/semantic_decisions.json
python validate.py --package ../runtime/adjudication/semantic_first_20260906/geometry --export ../runtime/adjudication/semantic_first_20260906/geometry/results/geometry_decisions.json
python build_consensus_review.py
python build_joint_consensus.py build

# 当前推荐：生成只有3张问题图的最小复核包
python build_minimal_gt_resolution.py build

# 收到最小复核JSON和同名CSV后执行：
python build_minimal_gt_resolution.py validate --package ../runtime/adjudication/semantic_first_20260906/minimal_gt_resolution_20260915_v1 --export ../runtime/adjudication/semantic_first_20260906/minimal_gt_resolution_20260915_v1/results/minimal_gt_resolution.json
```

`build.py geometry` 强制检查完整13张、图片哈希、坐标范围、分类值、CSV/JSON一致性和已提交声明，并把第一步导出按SHA-256归档到管理员目录。它不能验证人在脑中是否认出了旧图，所以这仍是第二位测量者的独立意见，不称为独立第三方盲裁决。

## 共识和后续边界

收回第二位测量者的意见后，按相同原图和叶片位置整理差异，只把仍有分歧的项目交两人讨论。第一位测量者两轮记录只算同一个人的重复测量，不能形成“两票对一票”。意见一致还须通过图像可测性检查；意见不一致或候选都不完整时保留待定、安排一次必要重描，或请第三位有相关经验的人员裁决。

裁剪或缺损造成的真实叶片可记为存在但不可测。无法确定身份时保持不确定，不默认为不存在。如何进入计数统计须在最终GT冻结前逐项记入共识决定，不自动排除样本或改变锁定Core/Diagnostic集合。多条候选几何等效的情况下可记录等效；涉及叶片数量、身份或完整性分歧时不可按字母决定。

三轮原始记录及其裁决前可靠性结果保留。因后续发现图像截断，既有MDC95描述的是当时三轮记录的重复差异，不能证明输入图像完整或人工真值正确。若补充质量敏感性分析，须单独命名并说明原因，不能覆盖测量前锁定主计算结果。

## schema

每个JSON包记录版本、匿名包ID、阶段和导出时间；每张记录包含匿名图像ID、图像哈希、提交时间、修订历史、检查完成声明和动态结构列表。CSV每图一行，保留分类计数及完整 `record_json`，必须与JSON逐项一致。

| 字段 | 内容 |
|---|---|
| `leaf_id`, `point` | 随机结构ID及输入图像像素位置；定位圈不作为新路径测量或训练坐标 |
| `status` | `measurable / visible_unmeasurable / uncertain / non_leaf` |
| `reason` | 无、叶尖裁剪、基部裁剪、遮挡、损伤、身份不确定、干扰或其他 |
| `confidence`, `notes` | 第一阶段独立判断及理由；不可测／不确定／非叶片必须填写理由 |
| `candidates_seen` | 第一包固定false，第二包true；这是流程记录，不是心理盲法证明 |
| `history` | 每次修订前的已提交记录；原始导出另行归档 |
| `geometry_choice` | 第二阶段指定匿名候选与路径ID，或重描／保留争议／不适用 |
| `geometry_notes`, `geometry_confidence` | 几何理由、发现的新问题与信心 |
| `semantic_export_sha256` | 第二包绑定的第一阶段原始JSON；避免静默切换语义版本 |

## 验证

`python -m unittest discover -s . -p "test*.py" -v` 检查两阶段门禁、旧共识约束和最小复核导出。`node dry_run_minimal_gt_resolution.cjs` 在系统临时目录中模拟3张填写、一次修订、备份恢复及JSON/CSV导出，不写入真实 `results/`。浏览器演练需要通过 `NODE_PATH` 指向安装的Playwright；可用 `EDGE_PATH`、`PYTHON_EXE` 指定程序。
