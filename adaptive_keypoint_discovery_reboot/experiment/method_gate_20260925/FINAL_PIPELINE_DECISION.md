# Final Pipeline Decision（2026-09-25）

**RESULT 3 = V2 明显恶化 → 回退并冻结 Pipeline V1。**这是唯一正式 V2 运行后，按 `PIPELINE_V2_PREREGISTRATION.md` 第6节“安全门槛先于收益门槛”得出的强制裁决，不是事后调整参数的建议。

V2在40株val中虽使Teacher/Student原超距点分别有5/3个实际入图并参与路径，但没有解除两株不同原始帧的原基部失败或零路径，也没有新增正确的人工GT匹配。相反，Teacher/Student零路径图从1/2张增至8/11张；Core和Diagnostic四组可测GT匹配均下降（Teacher 20→12、9→6；Student 20→15、7→5），并新增3个预定义的GT基部偏离事件。按预注册，任何这些安全失败都足以否决V2；条件性预测精度上升不补偿漏配。完整数字和不确定区间见 `PIPELINE_V2_V1_V2_COMPARISON.md`。

从此：

- 冻结的正式接口为原 `0.025D` 硬投影 **Pipeline V1**，对应未改的 `experiment/point_conditioned_graph.py`、冻结ROI/基部、图、`local_learned_support` decoder和 `phenotype-geometry-v1`。V2代码、一次运行输出和失败证据保留为**阴性消融/方法开发记录**，不得作为最终模型接口或继续调参的起点。
- `FINAL_METHOD_GATE=B` 的既有定位不被此结果偷换：epoch-53 Student-B仍是冻结的快速 learned surrogate，Teacher-direct仍是强参照。这里的裁决只针对点—结构关联接口，不重新训练或重选Teacher/Student。
- 预注册规定的唯一V2开发机会已经使用完毕。不得扫描新的投影/ROI阈值、修改公式、对个例补丁、重跑正式V2、开发V3、修改GT或重训Student。
- V4 test **未读取、未预注册**。本节点停止val方法开发。唯一下一阶段是**另行**预注册V4 locked-test最终评价；本文件本身不授权也不构成test协议。

复现锚点：预注册提交 `6824bd9`；V2冻结实现提交 `68402911d123a190ab2f1d2b1e01102bfe499f9c`；实现冻结记录SHA-256 `6640db39fad47128d1d03924cedfb01d2fb0cb1738db52f388b81e289ca613ed`；80实例正式V2封存表SHA-256 `1541f7882f7136aed45d9374bd068f4db13eb4010bbb97da1bd49121291169d8`；16株GT裁决JSON SHA-256 `4e61c27daca1e7b6315208d43a06511af9a6e16d389944b3fcf5e2cf5fba7a59`。含解盲数据的逐株明细只在Git忽略的 `experiment/phenotype_pilot_protocol/runtime/pipeline_v2_20260925/`，不上传GitHub。
