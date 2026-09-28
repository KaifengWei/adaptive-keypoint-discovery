# V4 locked-test 单人模型盲 GT 提交与冻结审计

日期：2026-09-28。适用原`V4_LOCKED_TEST_PREREGISTRATION.md`及`V4_LOCKED_TEST_GT_SINGLE_RATER_AMENDMENT.md`；仅执行人工记录行政核验与已锁定几何派生，不修改模型、方法、GT定义、匹配或test split。

## 1. 原始提交与来源

测量者完成40/40株正式提交。四份原始导出已按字节复制到Git忽略的`experiment/phenotype_pilot_protocol/runtime/v4_test_single_rater_20260927/results/`，没有覆盖原下载文件：

| 文件 | SHA-256 |
|---|---|
| `raw_annotations.json` | `b3a201f6ca38073b08dabab74d672f3136339177b577a8571cc7238f992c2073` |
| `sessions.csv` | `0656f3cd5a20a6f2c7e7147e19449b4b167681ba9c963e2323cdd5ba9a6d63bb` |
| `traces.csv` | `9b1d0432e6c92bfc814f69a18149675dce5bfe529a9092ad9918219c6fe34b1d` |
| `SHA256_LEDGER.json` | `27e98ff9567f7ec538f12b87c7f29c29ada5be9b3fae40c9f9cec8d3c49d3c91` |

既有`package_tools.validate_exports`复核原始JSON envelope/SHA、40株身份、每张完整提交及revision chain、状态/控制点/基点/叶尖坐标、CSV逐条对应和导出清单哈希，结果PASS：40株、40个正式snapshot、无技术修订；91条结构，其中`measurable=88`、`visible_unmeasurable=3`、`uncertain=0`、`non_target_structure=0`。此为**技术完整性核验**，不是新增第二测量者生物学复核。

## 2. 唯一最终 GT 冻结

版本化适配器`single_rater_gt/freeze_test_gt.py`只读取上述冻结人工导出、私有blind映射、锁定test CSV元数据与原shoot mask，并调用既有`phenotype_gt_geometry.py`。原shoot mask仅用于预注册规定的地上部bbox diagonal `D`；没有读取test RGB像素、预测结果或运行任何Teacher/Student。源crop与标准化尺寸一致性、mask二值/像素计数和40/40身份同时核验。600 dpi毫米只作`derived from scanner metadata`，不能当作实物标尺校准。

测量原始控制点保持原样归档；可测路径由冻结PCHIP弧长240点算子派生，完整派生points的SHA-256作最终trace UUID，原测量trace UUID另外保留。每株按既定长度/弦长/顺时针角/UUID tie-break只选一个`reference_main_path`；非主路径按冻结分化角算法计算。三条`visible_unmeasurable`仅保留存在性，无长度和角度；3个分化角无法解析，保留null，不填0或人工补线。

私有冻结目录：`experiment/phenotype_pilot_protocol/runtime/v4_test_single_rater_20260927/final_gt/single_rater_20260928_v1/`。该目录由`.gitignore`排除，**不得上传GitHub或公开给测量者**。

| 冻结产物 | SHA-256 |
|---|---|
| `final_gt.json` | `8d3f33854075c841020bd063efdefe0550aaaa7b00c746639687ff35e00222ad` |
| `final_gt_structures.csv` | `a981ba4613c3dd5aa740f4c66ec258ba59e8d7d7d010e1fd290380133f3b0ba8` |
| `freeze_manifest.json` | `9ce4dd58dbf9c7be9c466c5cbfae77bbeff2993a0154bc6121c6a281f3592aea` |
| 冻结脚本 | `f7be71e9a635b26c902edd40bce4e64fd927d8d81f7e3b5a1015c4a9383149f7` |

最终GT为40株、91条结构、88条完整几何、3条存在性但无几何；40株各有一个确定性主路径。45个非主路径分化角可解析，3个不可解析。导出后重算文件SHA与`freeze_manifest.json`逐项相同；全部240点几何为有限数且在标准化图像范围内。冻结目录不允许覆盖；任何真实技术损坏须保留原版本和完整修订链，并在模型访问前另行审计。

## 3. 验证与下一门槛

原`phenotype_gt_geometry.py`5项测试、测量包12项schema测试、GT冻结适配器纯合成测试与Python语法检查通过。真实40株仅进行已提交raw/metadata/原shoot mask的预注册几何派生，没有浏览或判断test图像内容，也没有按结果修改人工记录。

当前`final GT = FROZEN`，`test evaluator = PENDING`，`test_model_reads = 0`（指本项目已登记执行记录），`INFERENCE_GATE = CLOSED`。下一步仅能冻结test-specific evaluator及其合成/历史输出验证，登记其源码、依赖和GT哈希；然后还需用户**显式授权**，才可首次运行40株×冻结方法的一次性test推理。此GT冻结本身不是推理授权。GT是`single-rater model-blind phenotype reference`，不得写成双人共识GT；开发阶段3.72%长度与17.08°角度MDC95仅为既有噪声参照，不新估test可靠性或用作正式等价界值。
