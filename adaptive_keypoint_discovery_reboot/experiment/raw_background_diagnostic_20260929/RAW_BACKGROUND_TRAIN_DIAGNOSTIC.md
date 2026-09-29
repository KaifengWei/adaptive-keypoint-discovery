# V4 train 原背景—白底冻结方法配对诊断

日期：2026-09-29。状态：**探索性输入敏感性诊断，不是第二轮V4 test，也不是新的方法选择。**预先锁定的规则见[PROTOCOL.md](PROTOCOL.md)。

## 执行与身份

仅使用V4 train新增扫描单株中20个不同`source_frame_id`，每帧按固定哈希规则选一株；20株、3种冻结方法、clean/raw两条件，共120个method–plant–condition实例。`cv-public`连接的同一台`neaucs2-OMEN` RTX3090完成全部实例，执行异常0；V4 val/test图像、人工GT和test evaluator均未进入本次运行。历史正式V4 test累计120个实例的关闭状态未改变，本次`test_model_reads=0`。

原背景条件从原始扫描帧按V4记录的同株裁剪框重建，保持原RGB，缩放到原构建尺寸并加相同白色配准边框；**不是**800像素的`raw_review`缩略图。两条件尺寸、letterbox、冻结器官掩膜、从clean图推导的phenotype ROI/基部过渡区、权重、点生成器、Pipeline V1 graph/decoder及阈值全部相同，仅替换ROI内RGB。因此这是“原背景像素的局部输入敏感性试验”，**不是无掩膜/无ROI的完整原图端到端系统**。整幅518画布中差异像素占比中位0.60%，固定ROI内中位19.20%。

三方法运行时Git提交为`7788a09`，配对runner源码SHA-256=`962c1f8187830ad1d3f85cc2d2e907e0296d4c28ef99c165a0ce31069a9d2fb8`，三份远程运行账本中的源码SHA均与本机一致。

本机与远程`train.csv`字节SHA因跨系统文本格式不同，分别为`e5ce10f3...f6457e3e5`与`fc457c3d...f8af52b2fc5`；运行前把本机原字节副本传入私有包并验SHA，证明两端CSV的220行、顺序和全部字段解析结果完全一致。20张白底图和20张原背景配准图的SHA、尺寸及40个模型画布均通过预检。冻结权重/源码由原正式运行的哈希校验入口再次核对；GPU、CUDA/cuDNN及最小卷积检查通过。未改写远程原manifest。

## 仅描述性结果

| 冻结方法 | 点数中位数 clean→raw | 点数改变株数 | 路径数中位数 clean→raw | raw路径增加/减少/不变 | 零路径株数 clean→raw | 基点存在株数 clean→raw |
|---|---:|---:|---:|---:|---:|---:|
| Teacher-direct | 7→6.5 | 18/20 | 2→2 | 3 / 4 / 13 | 1→0 | 19→20 |
| Student-B epoch53 | 3.5→4 | 6/20 | 1→1 | 1 / 2 / 17 | 3→3 | 17→17 |
| Student-D（secondary） | 4→4 | 4/20 | 1→2 | 2 / 0 / 18 | 2→2 | 18→18 |

三法分别有17/5/5株的接受节点数改变，程序级graph failure均为0。但“有路径”“路径变多”不等于真实叶片恢复；无train人工GT，不能报告路径准确率、表型收益或方法赢家。基点在双方均存在的株上，中位位移均为0 px；Student-B的`v4_train_new_0089`出现约65.7源像素的基点移动，不能被中位数掩盖。

## 可复核的视觉案例

- `v4_train_new_0008`：Student-D由1条变2条，原背景叠线在视觉上覆盖上方分支；这只是待人工判断的可能恢复，不构成D优于B/Teacher的证据。
- `v4_train_new_0009`：Teacher-direct由2条变1条，原背景叠线中短侧枝不再单独成路径；提示直接去掉白底并非普遍改善。
- **`v4_train_new_0089`**：原裁剪可见上方植株与底部扫描器边沿；白底版本的主要深色结构是底部扫描器边沿，而真实植株主体没有完整保留。原V4 train元数据已把该图标记为`white_with_large_residual`，人工质量状态为`pending`；它是明确需要另行处理的数据质量案例。两条件在既有clean-derived ROI下仍沿底部边沿生成路径，不能以本次raw结果声称该病例已经得到修复。

这三个案例按固定输入包及路径变化逐例展示，不是人工GT，也不据此改模型。其余17株和所有失败/零路径均保存在完整并排页与逐株CSV，可继续查看；不选择仅有利的图进入统计。

## 结论与边界

原背景确实会改变冻结方法的候选点与部分路径，但方向混合，**没有证据支持把现有白底输入整体替换为原背景输入**。更有价值的发现是至少一个train标准化样本存在严重扫描边沿残留与植株主体缺失；这应进入数据质量审计，而非事后修改已经完成的V4 locked-test、GT或Student-B。若未来要主张“原背景处理更好”，必须先预注册一套真正不依赖clean-derived掩膜/ROI的完整原图管线，在新的独立样本和模型盲人工参考上验证；当前结果只能生成假设。

## 可复现产物（私有，未入Git）

- 输入包：`experiment/phenotype_pilot_protocol/runtime/raw_background_train_20260929/input_01/`；`input_manifest.json` SHA-256=`a270ce04284602277aa370ee9c9ad01cd6b9ed2d4476a13fad312a83547706ff`。
- 远程冻结预测的本机字节副本：同级`run_01/`；Teacher/B/D结果SHA-256依次为`8eadcaba54c92d39d36b2fd313ee41945fe5b229e4127d659cdc5b96ffbd1363`、`5c75317d9d912b99e158bcb4d0400c9be15c648d0a418e9e0ca046855eb54f3e`、`47423eeca32d1bfd9fe22152cad54e9cd7153dafae1651e13e2dfa144f93e41a`。
- 完整逐株表、汇总和160幅并排图：同级`analysis_01/`，入口`index.html`；`summary.json` SHA-256=`3e1f4079db07df05f28500a80b3807d8f1487b2e0c1acfa4cc2c07f85da4ebb4`，gallery manifest SHA-256=`83be5af993b524d09aaee0e972dd51dc9ebfc8047d55585c145aec535007c01d`。20/20株×8面板加载、导航、缩放及浏览器console检查通过。

大图、原扫描路径、原始预测和管理员级私有文件留在Git忽略runtime；Git仅保存规则、重建/推理/分析脚本与本报告。
