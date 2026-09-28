# V4 locked-test evaluator freeze audit

冻结日期：2026-09-29。状态：`final GT = FROZEN`；`test evaluator = FROZEN`；`test_model_reads = 0`；`INFERENCE_GATE = CLOSED_AWAITING_EXPLICIT_USER_AUTHORIZATION`。

本记录只冻结离线评估器和未来保存预测的接口。**没有运行 Teacher-direct、Student-B、Student-D 对任何真实 V4 test 图像的推理；没有用真实 test GT/预测做 evaluator smoke。** 已接受的40株单人模型盲GT保持原字节，91个结构（88个可测、3个仅存在性）、3个未解析分化角保留null。任何将来模型结果都不得引发GT生物学修改。

## 冻结入口与来源

- 代码：`V4_LOCKED_TEST_EVALUATOR.py`；SHA-256 `07e6ad7b06dfe58019c85e1956e0e60de84333c6b203109074a2c8231d43e167`；独立代码提交 `e4e21bd2dbdfed501e96b335ca12d2b88579a690`。
- 合成/历史保存结果验收：`test_V4_LOCKED_TEST_EVALUATOR.py`；SHA-256 `6cfe794cfca50004775fcdf620ccac0ae79c8218fd6546752f359f58c9f565af`。本文与`PROJECT_STATE.md`另以文档提交保存；此处代码提交用于避免在审计文件内自引用提交哈希。
- 冻结人工GT `final_gt.json` SHA-256 `8d3f33854075c841020bd063efdefe0550aaaa7b00c746639687ff35e00222ad`，位于Git忽略的私有runtime；仅校验其文件哈希，没有将GT内容提交Git或用于本轮真实test计算。
- 原预注册 `V4_LOCKED_TEST_PREREGISTRATION.md` SHA-256 `31734d9dad03dc38d93a59cea8b55ea4eb22ac67854d3e506d813b34eef3b3d9`；单人GT行政修订仍生效，其余评价规则不变。
- `phenotype_gt_geometry.py` SHA-256 `c10079ab308e0e6c4a9e3c1ea9ceceb9b2d112c181f7cfa410b6308f63eef800`：同一文件提供PCHIP240、确定性主路径、分化角和Hungarian tip/curve/polar matcher，既是geometry hash也是matcher hash。
- 原pilot evaluator `evaluate_frozen_phenotype_pilot.py` SHA-256 `b3285a35feab6748422f4341abe4427630143268c26e3b9c944474faa26aedd8`：仅复用其518画布inverse-letterbox；没有改写换算。
- Pipeline V1四源码SHA：`point_conditioned_graph.py`=`3a733ee2fb4213938f92a0f91c969bffe2b4c8e23b82356f07fd03ebba9577ca`；`point_conditioned_organ_paths.py`=`f960485eb6265122955ccc3ffa8a944c2fa66127f8c9c5b58683b46d3c3ab357`；`phenotype_roi_basal_anchor.py`=`d4dfedc3b3fa25c7a086d276d3e4f9ab9728ca7c9a28b2130cb7f678041db696`；`g1_prime_phenotype_bridge.py`=`027a364330888b403823c98ff77deb5e00db8792ad35e0831a45e97138292117`。源码哈希清单的排序紧凑JSON组合SHA-256=`7e34775b96042436f41dda9794c85068c56af9c199cce26d58d4c2441d480925`。
- 40株公开测量资产manifest SHA-256 `404b89b87b6ec3ea211069762de98718ad7ea47b3f55c54b6b06b45b74c87c45`；只用于核对图像身份与尺寸，不作为第二套生物学GT。
- 冻结权重来源SHA：Teacher DINO=`f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`，Student-B epoch53=`bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2`，Student-D=`b904eed30832d1a2c6cc20aca97e3d0140cc4444235c6a3b4b4b606bab17ce4a`。评估器只校验保存预测包中的来源声明；正式推理阶段仍须独立核对实际加载权重的字节SHA。
- 本机冻结验收运行时：Windows 10 build 26100、Python 3.11.0、NumPy 1.26.4、SciPy 1.16.3。评估器不导入Torch、模型、图像读取器或训练代码。

## 显式分离的输入schema

正式入口只在后续**显式首次test推理授权且保存预测封存后**接受两个独立文件：上述准确SHA的`final_gt.json`和`v4-locked-test-saved-predictions-v1`保存预测JSON。预测包顶层必须恰有`schema_version="v4-locked-test-saved-predictions-v1"`、`split="test"`、`pipeline_v1_sha256`、`method_weights_sha256`、`methods`；methods恰为`Teacher-direct`、`Student-B`、`Student-D`，每法40条，dataset_id全集与GT一致，无遗漏/重复。

每条植株预测记录必须提供`dataset_id,image_sha256,width,height,source_frame_id,paths,base_xy_model_canvas,basal_node_status,graph_status,decoder_status,point_count,accepted_node_count,rejected_node_count,graph_node_count,underground_association_count,provenance`。每条path必须提供`path_id,full_base_to_tip_path,provenance`；路径点在冻结518画布坐标系。空路径也必须有植株记录和明确失败状态。输入schema不含人工GT/自动ROI/替代matching阈值；模型预测生产者不可从评估器反向修改冻结方法。

正式入口校验GT、预注册/几何/matcher/inverse-letterbox/Pipeline V1源码hash、公开资产manifest hash、40株/8帧及三方法图像SHA/尺寸与权重来源标签。输出只允许写入新建的Git忽略私有runtime目录，存在目录拒绝覆盖；写入前先计算全部统计与输出字节，另出逐文件SHA ledger。

## 已冻结计算与完整输出

调用既有`phenotype-geometry-v1`的trace指标、确定性main-path、分化角和Hungarian一对一匹配；预测路径先通过原inverse-letterbox回标准化图。保留原tip≤0.12D、cost≤0.15、dummy0.15、0.55/0.30/0.15 tip/curve/polar和稳定tie epsilon。单人GT只通过冻结记录及原几何规则进入评价。

未来无条件生成40×3的`plant_level.json`、所有匹配/漏配/额外/仅存在性结构的`path_level.json`、`provenance.json`、三对配对的`paired_plant_level.json`、`statistics.json`、预注册40株例图排名`preregistered_figure_rank.json`及`evaluation_ledger.json`。零预测、零路径、基点/graph失败、缺角、可见但不可测和NA均原样保留；没有匹配的条件误差为null而非0。

逐株按原定义计算G/V/E/Q/P/M/U、measurable recall、existence recall及missed识别区间、precision及possible-extra识别区间、完整植株lower/upper、基点失败、path-count、matched length/angle和点/节点/路径数量。未匹配预测不自动命名为假枝。角度单列main/branch身份冲突、GT或预测角度未解析、可评价N；MDC95 3.72%/17.08°只作开发阶段人工噪声参考，不是等价性门槛。毫米误差始终注明`derived from scanner metadata`。

Primary=`Student-B − Teacher-direct`；Secondary=`Student-D − Teacher-direct`与`Student-D − Student-B`。条件长度/角度差仅在同一GT trace同时被双方匹配且角度可解析的交集上形成，每株先平均；单独报告共享植株/路径N。三方法每个draw共享40株有放回索引：10,000次PCG64 seed `20260927`；8个source-frame整组抽样的敏感性分析10,000次PCG64 seed `20260928`。95% percentile CI使用linear quantile，全NA draw排除并报告有效次数。宏平均及median/IQR为主，path-pooled只描述。预注册例图完整40株按原score tuple和SHA tie排序，固定rank1/20/40，不能换图。

## 验收证据及停止边界

`python -m py_compile`通过；`verify_sources()`全部冻结源码哈希通过；9项测试全部通过（0失败）：perfect match、wrong/missing base、no prediction/zero path/all-NA、partial recall与一对一Hungarian、extra prediction、`visible_unmeasurable`区间、unresolved angle、同一GT身份的Teacher/B配对、40×3完整输出和固定seed byte-identical 10,000次bootstrap、缺方法/缺株拒绝、非方图inverse letterbox/资产尺寸拒绝，以及**仅历史已保存val结果**的几何长度口径一致性。历史测试没有重跑val模型、没有打开test图像。`git diff --cached --check`通过。

本轮到此停止。**`test_model_reads=0`，`INFERENCE_GATE=CLOSED_AWAITING_EXPLICIT_USER_AUTHORIZATION`。** 用户审核本冻结审计后，须另行授权第一次且唯一一次正式40×3 test inference；GT freeze和evaluator freeze本身不是授权。
