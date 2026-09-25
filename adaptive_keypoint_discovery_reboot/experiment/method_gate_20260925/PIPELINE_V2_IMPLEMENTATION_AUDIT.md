# Pipeline V2 implementation audit（2026-09-25）

## 冻结顺序与合规性

唯一算法依据是 `PIPELINE_V2_PREREGISTRATION.md`，接受提交 `6824bd9`，文件 SHA-256 `8dfb79d5fc7ec52ee2520bd97413becb1fdcab4d82829c0fddd074fa5772b2eb`。这次只在新文件 `point_structure_association_v2.py` 改变已有 learned point 到冻结骨架像素的**接受判定**。最近像素 `cKDTree.query(k=1)`、原始置信度同像素去重、点作为唯一节点、V1 测地 MST、基部选择、局部解码及表型几何均调用未改动的旧实现；Teacher-direct 与冻结 Student-B 使用同一个函数和常数。没有训练或运行模型推理，没有改变保存点、GT、ROI、阈值/NMS 或数据划分。

执行顺序：

1. 完成静态编译、7项纯合成测试（公式边界、连续ROI方格距离、单调性、重复投影、空骨架、地下部独占排斥、加权拒绝），全部通过。没有对 `0002/0031/0005/0030` 或任何真实 val 图运行 V2 预试。
2. **只调用原 V1** 重建 40 val × 2 点源；输入点数、接受节点数、解码路径数均与冻结旧结果逐项 `80/80` 一致。最终检查记录 `runtime/pipeline_v2_20260925/V1_PREFLIGHT_FINAL.json`，SHA-256 `7f02d69addaa3371f5c91d9f6d2816bf1370d9eaa2455a7ea512484d98930847`。
3. 先提交版本化实现，最终实现 Git SHA `68402911d123a190ab2f1d2b1e01102bfe499f9c`；预注册提交仍是其祖先。随后在**任何 V2 真实 val 运行之前**生成 `runtime/pipeline_v2_20260925/V2_IMPLEMENTATION_FREEZE.json`，SHA-256 `6640db39fad47128d1d03924cedfb01d2fb0cb1738db52f388b81e289ca613ed`。该记录固定 Git SHA、四个实现代码 SHA-256、Teacher/Student 点、val manifest、40张图及其三种掩膜、冻结旧代码与旧结果、checkpoint、GT 仅哈希和预注册哈希；`v2_real_val_calls=0`、`gt_read=0`、`test_read=0`。
4. 冻结后只正式调用一次 V2：40 val × Teacher-direct/Student-B = 80 个图—方法实例，输入425个保存学习点；在读取任何 GT 内容前封存完整记录。封存总表 `runtime/pipeline_v2_20260925/FORMAL_VAL_40x2/SEALED_SUMMARY.json`，SHA-256 `1541f7882f7136aed45d9374bd068f4db13eb4010bbb97da1bd49121291169d8`，记录 `status=SEALED_40x2`、`image_records=80`、`point_records=425`、`path_records=97`、`gt_read=0`、`test_read=0`。无第二次 V2 正式运行。
5. 仅封存后读取冻结的16株人工GT，并按原几何与匹配函数一次性生成 `runtime/pipeline_v2_20260925/GT_COMPARISON_AND_GATE.json`，SHA-256 `4e61c27daca1e7b6315208d43a06511af9a6e16d389944b3fcf5e2cf5fba7a59`。未修改实现、预注册或 GT。

## 版本与输入锚点

| 项目 | SHA-256 |
|---|---|
| V2 关联实现 | `4b9fc7f075f9d4f22303f265c4489aec18d8d9021c5e15621ce6905c9767c9c5` |
| V2 合成测试 | `3cf67be26505dbf11078cbce805cd429883ed2e1fb7579cc3cc153b5a835f106` |
| 单次运行与冻结入口 | `86f435adb69c45725934030f1c18af8f82786e73c029920a8f003c76e12ba4bd` |
| 一次性GT裁决入口 | `0e51126e0f0eea600355901f601a9146ca3d9efb932eb9b72234714c3482a220` |
| Teacher / Student 保存点 | `1ed017a6d6ba6f360bf216c44b01ffedaef9c2a14f18d46d75b1f640763b1814` / `9a11bbcc7283333fca34c0e82a3931edbfa96cef8347ec952f450daecb1aa756` |
| V4 val manifest | `6931e8b5156edbb8cf0f7274e5da618c894ed8e53075fb92aecfc8e146252d42` |
| Student-B epoch53 checkpoint | `bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2` |
| 最终人工GT（正式运行前只核哈希） | `1b3018c83b2695951dfdf2adff10020f9c3e6e739babfe12ad235a88ae7b53b3` |

单次运行输出分别保存逐点 `A_i,d_i/D,δ_i,r_i,σ_i`、ROI/地下部判定、接受/拒绝和图节点/表型路径参与；逐图基部与路径数；完整图节点/边；路径几何；decoder 末端剪除决策。五个 JSONL 的 SHA-256 全在 `SEALED_SUMMARY.json` 中；这些含解盲信息的明细位于 Git 忽略目录，不上传 GitHub。公开代码和本审计可复现机制，私有文件用已记录哈希核对。V4 test **未读取、未预注册**。
