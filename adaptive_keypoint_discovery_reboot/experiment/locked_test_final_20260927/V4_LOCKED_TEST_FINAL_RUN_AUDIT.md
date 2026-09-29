# V4 locked-test final run audit

日期：2026-09-29。该记录描述**首次且唯一一次**已获用户明确授权的 V4 test 正式推理与离线评价。`test_model_reads` 已由 0 变为 120 个 method–plant 正式实例；`INFERENCE_GATE=PERMANENTLY_CLOSED_AFTER_AUTHORIZED_RUN`。不得把旧 checker 的预注册阶段常量 `CLOSED_GT_AND_TEST_EVALUATOR_PENDING` 当作本轮实际状态。

## 执行前冻结核验

- 用户先授权将 `cv-public` 的旧仓库安全快进到冻结版本；经祖先关系及路径冲突核验后，远程从 `a8a9419` 快进至 `0d8c7d5`，三个未跟踪的历史 val 目录原样保留。新的、仅编排冻结调用的 GT-blind 一次性预测生产脚本和纯合成测试提交为 `d9c58adc8a4285dd92429604b375bc8ea1c6ac91`；远程通过 `git merge --ff-only` 与此完全一致，冻结方法源码未修改。脚本 SHA-256：`493f9c250636d070c17513a22a8003610246e0185691623fd9cb9c94c017dc9c`。
- 远程主机 `neaucs2-OMEN`，通过 `cv-public` 的既定 ED25519 指纹连接；唯一使用 RTX 3090、`/media/neaucs2/evs/envs/adaptive_kp/bin/python`，启动 PyTorch 时清除继承的 `LD_LIBRARY_PATH`。运行前 synthetic CUDA 和脚本合成检验均通过，无其他 compute-app 占用。方法各在独立进程中按 Teacher-direct → Student-B → Student-D 执行。
- 冻结 DINOv2 权重实际字节 SHA-256：`f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`；Student-B epoch53：`bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2`；Student-D：`b904eed30832d1a2c6cc20aca97e3d0140cc4444235c6a3b4b4b606bab17ce4a`。运行前再次实际计算并与冻结账本核对，模型加载仍使用这些固定文件。
- Pipeline V1 四源码分别为 `point_conditioned_graph.py`=`3a733ee2fb4213938f92a0f91c969bffe2b4c8e23b82356f07fd03ebba9577ca`、`point_conditioned_organ_paths.py`=`f960485eb6265122955ccc3ffa8a944c2fa66127f8c9c5b58683b46d3c3ab357`、`phenotype_roi_basal_anchor.py`=`d4dfedc3b3fa25c7a086d276d3e4f9ab9728ca7c9a28b2130cb7f678041db696`、`g1_prime_phenotype_bridge.py`=`027a364330888b403823c98ff77deb5e00db8792ad35e0831a45e97138292117`；组合 SHA=`7e34775b96042436f41dda9794c85068c56af9c199cce26d58d4c2441d480925`。
- 冻结几何/Hungarian 源码 SHA=`c10079ab308e0e6c4a9e3c1ea9ceceb9b2d112c181f7cfa410b6308f63eef800`；inverse-letterbox 来源=`b3285a35feab6748422f4341abe4427630143268c26e3b9c944474faa26aedd8`；评估器 SHA=`07e6ad7b06dfe58019c85e1956e0e60de84333c6b203109074a2c8231d43e167`；原预注册 SHA=`31734d9dad03dc38d93a59cea8b55ea4eb22ac67854d3e506d813b34eef3b3d9`；final GT SHA=`8d3f33854075c841020bd063efdefe0550aaaa7b00c746639687ff35e00222ad`。
- 冻结 V4 test 清单为 40 株、8 个 source frames，canonical identity SHA=`2acfb37e6850ac208baca2cb8d56cd71ba9b9a12ae9c64f62f2a1a1fce56dd0c`；40/40 张 PNG 的实际字节 SHA 和宽高分别与清单及模型盲测量资产清单一致。模型前没有 test smoke/预览或按图挑选。远程工作树的受跟踪文件无变动；三个无关未跟踪 val 目录没有参与本次运行。

## GT-blind 预测及封存

- 原 `run_frozen_diagnostics.py` 的 Teacher/B 点生成和完全相同的冻结 graph/decoder 调用用于正式预测；D 按其原冻结 checkpoint/config 调用相同 Pipeline V1。纯合成图上，新编排入口的 graph/decoder 输出与原冻结调用字节一致。没有更改 Teacher、B/D 权重、threshold/NMS、ROI、association、graph、decoder、phenotype geometry、人工 GT 或 split。
- 每方法先用同一张既有 train 图 `v4_legacy_0001` warm-up 两次，然后每株只执行一次 point generation、一次 graph+decoder、一次 phenotype geometry；CUDA 同步后逐阶段计时，batch=1。event ledger 恰为 6 个非test warm-up、120 个 `instance_started`、120 个 `instance_completed`、0 个执行异常；每法恰 40/40 条，零路径和失败状态保留。没有重跑任何 test method–plant 实例。
- 预测生产脚本没有打开或导入 `final_gt.json`。完整 `v4-locked-test-saved-predictions-v1.json` 先封存，SHA-256=`305974bdfdbe9734e9b53f2200f0e6003a3cada01489e1061ac34cfb71267375`；运行计时记录 SHA=`22cc273db7c04d18fc0296c4742817e75cbfcac9faa7c6fa309f2bdda3c12d82`；run ledger SHA=`030290d0a04e367f3823bd4baae5b3223d0e102cf1e115685aa5ce690af8bf6f`；event ledger SHA=`818c8966cb135c14c566676a79ba3b5563674b5c72019dd65bd505934bcfafab`。本机传输后的预测包 SHA 与远程一致。
- 私有预测和计时位置：`experiment/phenotype_pilot_protocol/runtime/v4_locked_test_final_20260929/run_01/`，Git 忽略，不提交逐图预测、图像或人工 GT。完整 `prediction_freeze_manifest.json` 与上述账本一并保留。

## 封存后评价

- 只有预测包封存且传输 SHA 确认后，才在本机调用 SHA 已冻结的 `V4_LOCKED_TEST_EVALUATOR.py` 读取预测包和 final GT。评估器校验 40×3 全量 schema、GT/图像/尺寸、权重来源与冻结源码哈希，生成 `plant_level.json`、`path_level.json`、`provenance.json`、`paired_plant_level.json`、`statistics.json`、`preregistered_figure_rank.json`、`evaluation_ledger.json`，实际为 120 条植株级记录。
- 私有评价位置：`experiment/phenotype_pilot_protocol/runtime/v4_locked_test_final_20260929/evaluation_01/`；`evaluation_ledger.json` SHA-256=`76145183e356a96cd100edde7b3d50b44b1d86e80d32ebc9b57f920e3735ee19`，其中记录所有六个结果 JSON 的逐文件 SHA；`statistics.json` SHA=`e838d5716694536a0a50b44f0a3d798083cc5469c039cc88372baa28578a678a`。10,000 次 plant bootstrap 和独立 10,000 次 source-frame sensitivity 均按预注册 seed 生成。
- 真实test之后未修改/重训任何冻结方法，未重标 GT，未调阈值，未执行第二轮推理或额外test benchmark。自第一次真实test模型读取起，所有 test 失败只可作为本次结果和后续独立研究的讨论，不可回填当前方法。

结果与解释分别见 `V4_LOCKED_TEST_FINAL_RESULTS.md` 和 `V4_LOCKED_TEST_FINAL_DECISION.md`。该研究的三个目标继续分开：optimization target 为冻结 Teacher pseudo-targets；checkpoint-selection target 为原 internal-validation loss；scientific target 为独立的单人模型盲 phenotype GT。
