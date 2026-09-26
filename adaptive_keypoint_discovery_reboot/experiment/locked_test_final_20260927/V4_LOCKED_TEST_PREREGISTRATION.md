# V4 locked-test final evaluation preregistration

版本：1.0；日期：2026-09-27。状态：**仅预注册和静态检查；尚未建立 test GT，尚未冻结 test evaluator，尚未执行任何 test 模型推理。**

本协议遵循用户本轮批准的 pasted request。方法冻结基线 Git commit 为 `0ac28543e5de521759520641f62eb073cf5508f9`。本文件最终 commit 和文件 SHA-256 在提交后另行报告，不在文件内自引用。它不是训练计划，也不授权模型打开 test。

## 1. 研究问题与永久关闭的开发边界

1. 冻结的自适应关键点 → point-conditioned graph → local decoder → phenotype pipeline 能否在未参与模型/方法开发的 V4 locked test 上恢复真实地上部结构并支持表型计算？
2. Student-B 在推理时不调用 Teacher，能否保留 Teacher-direct 的结构发现和表型能力，并具有已验证的效率价值？“被学习”不要求超过 Teacher；“接近”是有置信区间的描述，不是正式等价性结论。
3. Student-D 仅检查 enhanced Teacher 的开发趋势是否重现，不能根据 test 升格为新主方法。

Primary methods：Teacher-direct + Pipeline V1；Student-B epoch-53 + Pipeline V1。Secondary：冻结 Student-D + Pipeline V1。

永久冻结：Teacher、B/D checkpoint、architecture、loss、heatmap、threshold、NMS、ROI、basal transition、association、graph、local decoder、phenotype geometry、数据 split。Pipeline V2 为 RESULT 3 阴性消融，永久回退 V1；不开发 V3。V4 val 方法开发永久关闭。用户尚未纳入的额外秧苗图像本阶段不读取、不整理、不筛选、不训练、不合并。

始终区分：

- Optimization target = frozen Teacher pseudo-targets。
- Checkpoint-selection target = internal-validation loss。
- Scientific target = independent phenotype GT。

Student-B seed=20260718，182/34 internal split，80 epochs / 3680 logged minibatches，best epoch=53，best internal-val loss=0.28449777762095135，审计 CONVERGED；不继续训练、不覆盖原 checkpoint。

## 2. 最终方法、代码和清单冻结账本

### 2.1 冻结推理行为

- DINOv2 ViT-S/14-Reg，官方权重及 third-party source commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；第三方本机 worktree 无修改。
- Teacher-direct 按 Route B 原实现：518 letterbox phenotype ROI；原 nine-view transforms；结构候选预算30；`evidence_mode=full`、`structure_coverage=False`；consensus `min_presence=0.75`、`max_localization_error=0.025`、consistency filter 开启。没有给 Teacher-direct 换成 D 的增强教师。
- B/D 按各自原配置：518输入、decoder_dim192、output_stride4；probability threshold0.35、max-pool NMS kernel5、safety cap64、fixed_k=0。保留原 ROI 点过滤和解码顺序。
- 共享 Pipeline V1：原 skeleton/support、`max(4px, 0.012D)` spur pruning；nearest skeleton projection `distance<=0.025D`、同像素去重、原测地图与区域注释；`phenotype_roi_v1`、`basal_transition`、`local_learned_support` decoder。不能换回旧 global-bbox decoder。
- shared graph + decoder 必须调用同一冻结源码。Student 推理不能执行 Teacher 候选/consensus；共享经典结构支持不等于调用 Teacher。
- source SHA 是冻结行为的最终依据；上述文字不能用于重新实现另一个“近似相同”的方法。B/D seed、阈值等不得根据 test 更改。

### 2.2 跨设备账本与状态

2026-09-27，经 `cv-public`（ED25519 fingerprint `SHA256:5w2zGYQHJU4wuuzgLRFjuaHjL9ZbyXNia4EIavPVLes`）验证成功连接 `neaucs2-OMEN`。实际 GPU 为 NVIDIA GeForce RTX 3090，24576 MiB，driver560.35.05；环境 `/media/neaucs2/evs/envs/adaptive_kp/bin/python`，Python3.12.13，PyTorch2.9.1+cu128，CUDA12.8，cuDNN91002；CUDA synthetic-convolution smoke 通过。检查没有运行训练、Teacher 或 Student。启动 PyTorch 时沿用 `env -u LD_LIBRARY_PATH`。

远程 repo `/home/neaucs2/kp/adaptive-keypoint-discovery` 当前 HEAD=`45a44dabea169872c63e62b387da6102552b7a36`，落后本机且有未跟踪实验目录；本轮不强制 pull/reset，不覆盖这些文件。正式推理前必须受控同步并复核源码/hash。远程尚缺本机 pilot evaluator；它本身也不是可直接执行的 test evaluator。

B/D checkpoint 和本节已核对的关键模型/接口源码在远程符合下方 SHA。D checkpoint 当前仅远程有，本机不复制、不 torch.load；其远程 SHA 已实际计算，不是凭名称推断。原 evaluator 的 SHA 指冻结**几何/匹配参考实现**；test-specific manifest adapter 和统计 wrapper 尚未实现、尚未冻结，不能把它冒充已经可运行的 test evaluator。

清单 raw SHA 在 Windows/远程不同，因此同时登记每份 raw hash 和跨平台 canonical identity hash。canonical 只覆盖排序后的 dataset_id、split、source_frame_id、split_group、crop_box_full、source_crop_box_fullplant、output_sha256、relative_path、normalization_version，路径分隔符转 `/`；它证明锁定身份/裁剪/图像预期字节哈希一致，不宣称全部其他 CSV 字段逐字相同。

| split | N | source frames | canonical identity SHA-256 |
|---|---:|---:|---|
| train | 220 | 20 | `5527fcf43e4a32aa32a5d872c7c2bc9a103c3a9c6e4907939238ae36106b368c` |
| val | 40 | 7 | `045e6c598a793da5dc45d22d53890a5e6bcb2d85d169d4cb3720bdeccebec817` |
| test | 40 | 8 | `2acfb37e6850ac208baca2cb8d56cd71ba9b9a12ae9c64f62f2a1a1fce56dd0c` |

三组 dataset_id、split_group、非空 source_frame_id 的跨 split 交集均为0；test来自4个采集目录、8个扫描帧。上述为 CSV 元数据审计，**没有打开 test 图像像素，没有运行模型**。将来 GT 打包前应对正式图像字节 SHA 与 manifest 核验；不重新 QC 选样或删除难样本。

机器可读冻结账本如下。`remote_only` 仅允许预注册检查缺少本机 D；正式运行主机必须实际具备并验证 D。所有数字与 hash 不允许自动修复。

<!-- LOCKED_LEDGER_BEGIN -->
```json
{
  "baseline_git": "0ac28543e5de521759520641f62eb073cf5508f9",
  "dinov2_source_git": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
  "test_model_reads": 0,
  "bootstrap_seed": 20260927,
  "bootstrap_replicates": 10000,
  "artifacts": {
    "training_outputs/core_dinov2_v4_phenotype_roi/best.pt": "bb2fb948f60d5f3159893fee27493618caa416728f4e1d8d395099df98d19aa2",
    "training_outputs/core_dinov2_v4_structure_coverage/best.pt": "b904eed30832d1a2c6cc20aca97e3d0140cc4444235c6a3b4b4b606bab17ce4a",
    "third_party/checkpoints/dinov2_vits14_reg4_pretrain.pth": "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb",
    "g1_dinov2_feasibility.py": "c12b509281a76131c42bb61ea461d0344ccc218a2e1b32032e3a59c7bdfd22af",
    "g1_prime_structural_support.py": "02dc72b1a6adbdcf545a11a30afa37d09a535a24d16d33fe4e54b5acea365ddd",
    "generate_g1prime_pseudolabels.py": "2de6ff19c9c68fdc7eccf06a4e85603e4461af56f468da38af5b71a9163b48bf",
    "adaptive_point_model.py": "6ab9d154fbad3471b08b4cc20e77cd3c035ea4a3359636cd25220b590f2fb942",
    "evaluate_adaptive_point_detector.py": "6a4d08c8e952945cde0dd25947085203478c9396141569e30e2f453edacda073",
    "phenotype_roi_basal_anchor.py": "d4dfedc3b3fa25c7a086d276d3e4f9ab9728ca7c9a28b2130cb7f678041db696",
    "g1_prime_phenotype_bridge.py": "027a364330888b403823c98ff77deb5e00db8792ad35e0831a45e97138292117",
    "evaluate_point_conditioned_graph_v1.py": "f2a6779e82aef9b058431f2507c6c133bd367742cc55abc2ea3be4358e24d720",
    "point_conditioned_graph.py": "3a733ee2fb4213938f92a0f91c969bffe2b4c8e23b82356f07fd03ebba9577ca",
    "point_conditioned_organ_paths.py": "f960485eb6265122955ccc3ffa8a944c2fa66127f8c9c5b58683b46d3c3ab357",
    "phenotype_pilot_protocol/phenotype_gt_geometry.py": "c10079ab308e0e6c4a9e3c1ea9ceceb9b2d112c181f7cfa410b6308f63eef800",
    "phenotype_pilot_protocol/evaluate_frozen_phenotype_pilot.py": "b3285a35feab6748422f4341abe4427630143268c26e3b9c944474faa26aedd8",
    "method_gate_20260925/run_frozen_diagnostics.py": "1ab3fa19dafe4781e7e25a1fa12c6d02954621501cbc678a5c3080788e6a322f",
    "configs/train_core_dinov2_v4_phenotype_roi.json": "038437fe32495fca44ac9b639126c8cdc8c26462494fd8fd97ea001d7c3f7717",
    "configs/train_core_dinov2_v4_structure_coverage.json": "c0bca8f6834f54531e70914bb785cd0d4414bed79ab580200d6c885219ee7545",
    "training_outputs/core_dinov2_v4_phenotype_roi/resolved_config.json": "87929e7b5e318631d2b9dfa4ab87d981b30491144dbd0fb07fce59786b9a89b1",
    "training_outputs/core_dinov2_v4_structure_coverage/resolved_config.json": "0566abe00e981e399e6bb0ec7509d33a7f7ecd79dcffa54b14c6eba7332b5bfc"
  },
  "remote_only": ["training_outputs/core_dinov2_v4_structure_coverage/best.pt"],
  "splits": {
    "train": {"n": 220, "frames": 20, "canonical_sha256": "5527fcf43e4a32aa32a5d872c7c2bc9a103c3a9c6e4907939238ae36106b368c", "local_raw_sha256": "e5ce10f3abccae796ba527e8b0f182f95e4c2dccdf3e12a1ef8b627f6457e3e5", "remote_raw_sha256": "fc457c3ddf04527e9fc972c486801158dd6083b504714dd89543df8af52b2fc5"},
    "val": {"n": 40, "frames": 7, "canonical_sha256": "045e6c598a793da5dc45d22d53890a5e6bcb2d85d169d4cb3720bdeccebec817", "local_raw_sha256": "6931e8b5156edbb8cf0f7274e5da618c894ed8e53075fb92aecfc8e146252d42", "remote_raw_sha256": "2b312c055687132e02d203fbf292082cb2014bbd440918e8521454d788917e9a"},
    "test": {"n": 40, "frames": 8, "canonical_sha256": "2acfb37e6850ac208baca2cb8d56cd71ba9b9a12ae9c64f62f2a1a1fce56dd0c", "local_raw_sha256": "161f342037f4c982fb5f532bc60b18aadaad6656584ee667c7e0ca828818c5f6", "remote_raw_sha256": "52867815e555a9471d607014ed72fbd6c7c4dd3761e8a5a523fffe7844dfbef3"}
  },
  "test_gt": "PENDING",
  "test_evaluator": "PENDING"
}
```
<!-- LOCKED_LEDGER_END -->

## 3. 不可跳过的时间顺序和访问门槛

`final method freeze → preregistration freeze → model-blind human test GT → GT freeze + hash → evaluator freeze → 第一次允许模型读取 test → Teacher/B/D once-only inference → final statistics`。

当前 `test_model_reads=0`：指现有登记的 V4 执行记录及本轮操作，旧 V3 test 审计不混进来；此前纯像素数据构建/整株人工 QC 不冒充模型访问。静态脚本不能证明任意外部程序从未读取过图像，不对登记之外的活动作虚假保证。

所有 GT 原始记录、admin mapping、冻结 GT、evaluator 源码、依赖版本、执行环境及对应 SHA 必须在推理前登记。检查缺失/哈希不符即阻断，不能“先跑后补”。本轮不建立正式测量包、不生成预测、不开放 evaluator 执行权限。

正式一次性推理前另需用户授权。先建立一次 run ledger，登记40×3个 method–plant 实例；输出追加且不覆盖，保存失败/空路径。不得先做真实 test 冒烟或输出预览、再正式运行。不能因难例失败而重跑或删例；纯执行故障先停止并记录，不能私下开启第二轮 test。

## 4. Model-blind test GT：定义、规模和交付

### 4.1 规模与人工负担

推荐固定 **全部40株存在性 GT，全部可测结构完成几何；Rater1和Rater2各独立一轮覆盖40株（总80个 plant-sessions）**。两人从同一规则出发，但独立操作，不能看对方轨迹或任何模型结果。它比只标模型容易恢复的叶片更能防止选择偏差，也无需再做此前多轮描迹/候选排行。

**不新增强制重复轮次。**沿用开发阶段人工重复性作为 practical reference，不把本 test 两人一轮包装成 test intra-rater repeatability。只在两人的结构身份/数量、是否可测、共同基点或路径连续性存在实质分歧时做一次模型盲共识；细小等效线条不强行分胜负。仍无法确认则保留 uncertain 或 visible_unmeasurable，不凭推测补曲线。

若用户确实无法承担全几何或Rater2全40，必须在任何 test 模型访问前批准、提交单独的行政工作量修订；本协议不默认启用事后缩小子集。几何抽样唯一备用规则：对全部40个 locked dataset_id 计算 SHA256(`V4-test-geometry-20260927|`+dataset_id)，按 hash 升序取前24株，存在性仍覆盖40株。若采用该修订，Rater2必须覆盖同一24株几何，外加其余16株存在性；不得在看过模型表现后选择。当前默认是全40几何，没有抽样、没有打开图像。

### 4.2 操作性定义

目标是“可见且可独立追踪的地上部叶片结构路径”，不是农学叶龄、完整叶计数或固定人工关键点。存在性最低要求：可区分的独立远端/叶尖，且可沿绿色组织中心追踪至共同地上部路径或明确分化处。

先共同地上部基点（不以颖果/根尖作基点），再自行动态添加结构。所有路径由同一个基点走向各自远端，共享主茎/叶鞘段可以重复描绘；**共享段和中转控制点不算插值**。插值只指确实不可见但两端可唯一平滑连接的短间隙，沿用原 `<=0.05D` 规则；其他情况不给可靠几何。不能把残损裁剪边缘自动认作完整叶尖。

状态固定：

| 状态 | 含义 | 统计资格 |
|---|---|---|
| measurable | 存在且中心路径、共同基点及远端可以可靠确定 | 存在性及长度；角度仅可解析时 |
| visible_unmeasurable | 结构存在明确，但不能可靠完成中心路径 | 存在性，几何缺失；不伪造线/角度 |
| uncertain | 不能确定是否为独立目标结构 | 不强定真阳性/阴性，进入不确定区间 |
| non_target_structure | 根/颖果/伪影等明确非目标组织 | 不进入目标存在数；可记录位置和原因 |

长度正式名称始终为 **base-to-tip structural path length / 基部—叶尖结构路径长度**。不能写 botanical leaf length；共享主路径正是该定义的一部分。保留 px、bbox-normalized；毫米仅在沿用现有缩放/600dpi metadata 转换可靠可追溯时提供，注明 `derived from scanner metadata`。无独立已知尺寸标尺校验时不宣称毫米物理准确性。

### 4.3 页面、原始 schema 与盲法

公开包仅有原标准化整株 RGB（不加 ROI/自动掩膜/点线）及测量工具，不展示来源路径、dataset_id、模型名字/结果、预期叶片数、开发集合身份。opaque blind_id 用 `secrets` 生成不少于16位 `[A-Z2-7]` 随机串；不能编码 dataset_id。图名也使用 blind_id；两人可有不同显示顺序，映射只在管理员目录，不能随包发送。

`measurement_manifest.json` 公共 schema 固定为 `schema_version,samples`；每条 sample 仅 `blind_id,image_filename,image_sha256,width,height`。不传旧 visible_leaf_count，不预建 GT01/GT02，不预填 leaf_exists=yes；提交后按实际添加项目赋 trace identity。

人工原始记录仅用 blind_id：session_id、rater_id、round=1、revision、image_sha256、shared_base_xy、image_note、submitted_at；动态 traces 含 trace_uuid、visibility_status、points_px（缺几何允许空）、可独立记录的远端 tip_xy 或 null、occlusion、interpolation_used、confidence、reason/note、用于身份定位的人工辅助标记。辅助画笔不作为长度轨迹，不能用它生成不存在的路径。raw export 不含 dataset_id 或模型结果；保留全部修订，不覆写原始输入。

图片尺寸/sha是资产元数据，不是模型输出。公开 payload 用严格白名单验证；HTML/JS/CSS/JSON/CSV及文件名做泄漏扫描；未来需再做实际浏览器无输出盲法检查及2张合成图 dry run。当前只编写静态 checker 和合成 schema 测试，**不打开真实 test 图、不制作正式页面**。

### 4.4 最终唯一 GT 与冻结

两人结果先单独存档并哈希，保留原始可靠性证据。按冻结 matcher 做人工路径对应，错误匹配/identity不确定不自动合并。共同确认目标状态和存在性数量；几何均合格且身份一致的完整候选，使用既有规则：完全同线保留共同曲线；等效候选选完整曲线 medoid，不平均、不拼接。两人候选 medoid tie 用完整 points SHA256 字典序较小者，不能根据模型选择。

两人完整几何的等效参考沿用3.72%长度/17.08°角度，并且必须叶尖、基点、结构身份一致；这些值不是强迫可疑 GT 过关的理由。实质分歧只做一次原图盲共识，可一致选择一条既有完整人工曲线；无法一致时保留几何缺失/uncertain，不自动再安排第三/第四轮。规则无法覆盖的情况先停在GT阶段，不能看模型替人裁决。

GT 管理员记录dataset_id仅在冻结前映射核验/冻结后解盲，保留source provenance、图片SHA、原始session SHA、各状态、完整曲线、共同基点、D和 main-path身份。沿用原final GT做法，几何计算用完整points内容的SHA作为trace UUID，原测量trace_uuid另外保留；冻结前用同一UUID重算并核对main身份，不允许冻结后改变。冻结40株全量 JSON/CSV/hash、测量包hash、blind mapping hash后不再改 GT。

## 5. 几何与 matching：原实现不变

复用 `phenotype-geometry-v1`：去连续重复点（相邻距离<=1e-6），弧长参数化 PCHIP240点；其通用数值EPS仍为原1e-12，二者不混用。主路径用原 deterministic tie-break（长度1%近似并列 → chord差容限 max(1px,0.001D) → clockwise角 → trace UUID）。分化连续性容差max(2px,0.01D)，3点连续离开，0.05D局部向量窗；未解析角度记缺失，不设0°。

matching 不按编号对应：原 Hungarian assignment、dummy cost0.15、tip<=0.12D、cost<=0.15、原稳定epsilon；measurable完整曲线权重0.55tip+0.30curve+0.15polar；曲线距离为64点线性采样的对称平均最近距离。原函数另有0.75tip+0.25polar分支，但本次无可靠几何的存在性结构不强配一条预测曲线，只给区间，保持先导比较的缺几何政策。

模型518画布必须先按原 letterbox inverse 转回标准化原图，再用同一 geometry计算预测与GT。测量/匹配 D为原标准化图地上部 bbox diagonal（管理员由原shoot mask计算，不能用模型预测bbox）；graph投影 D 为冻结自动结构支持的模型画布bbox，二者不可混用。不得根据 test 修改这两个尺度。

test wrapper 只允许替换已锁定manifest/GT入口和统计汇总（不是改 matching），必须在GT冻结后、首次模型读取前实现并提交哈希。验证使用纯合成路径/已有历史保存结果，不重新运行val方法开发，不能提前跑test来验 evaluator。

## 6. 主指标、区间与失败可见性

对每株、每方法：G=measurable GT数；V=visible_unmeasurable数；E=G+V；Q=uncertain数；P=预测路径数；M=原matcher的完整几何匹配数；U=P−M。non_target不入E。

1. **Measurable path recall**=M/G；missed measurable=G−M。
2. **Structure existence recall**报告 conservative identification interval `[M/E,(M+min(U,V))/E]`；全有几何时退化为确定值。不可把 count-only 个体随意配成 true positive。存在性missed interval=`[E−M−min(U,V),E−M]`。
3. **Prediction precision / possible-extra**：precision interval=`[M/P,(M+min(U,V+Q))/P]`；possible-extra interval=`[max(0,U−V−Q),U]`。若Q>0，注明“相对于已确认存在结构，额外可能为未确定目标”，不宣称精确FP；区间不是统计CI。P=0时precision为NA，zero-pred plant显式报告，不能从recall分母删除。
4. path-count绝对差及有符号差；所有漏检/多检和零路径保留。G/E=0时相关比值NA且报告计数；整体 pooled及有效plant分母一起报告。
5. 匹配路径长度 symmetric relative error=`100*abs(Lp−Lg)/((Lp+Lg)/2)`；px及normalized绝对误差。必须同时展示recall、missed、extra及 matched N。每株先计算匹配路径平均误差，再报告植株宏平均、median/IQR；所有path汇总仅描述。没有匹配的植株误差NA但保留在结构失败表，不能据条件误差宣称优越。
6. paired conditional length：只在同一GT结构同时被两个方法正确匹配的交集上计算每株平均差，报告交集植株/路径数量；不能拿两套不同成功子集的误差差当公平改善。primary B−Teacher；D−Teacher及D−B secondary。
7. divergence angle绝对误差使用原 evaluator：只有GT与预测都为可解析分支且identity匹配时计算；main/branch身份冲突、unresolved及可评估N单独列出；不另换主路径让角度好看。
8. **Complete-plant structural recovery**：确定成功需E>0、Q=V=0、M=G=E=P，且基点可靠；否则不宣称完整恢复。每株lower indicator为上述确定成功；upper indicator需E>0、M=G、E<=P<=E+Q且基点可靠（缺失曲线的结构仅视为可能恢复）。总体possible-success interval为`[sum(lower)/40,sum(upper)/40]`；含V/Q者及E=0者单列indeterminate，不能只挑可测植物作总体100%。可靠基点指有输出、可与冻结人工基点核对且未触发下面的基部偏离/缺基点标记；人工基点无法确定时不认定成功。
9. **Base failure**分开记录：冻结decoder无合格basal node（含zero-path collapse）；已输出基点但距离人工共同基点>0.025D（原图测量D）。该比较仅是冻结错误审计标签，不改变decoder阈值。
10. **False branch / extra path**首先报告unmatched及possible-extra区间。可用模型盲GT中的non_target辅助标签和冻结路径provenance解释，但不能把所有unmatched自动叫“假枝”。人工不在看模型后改变GT；未能归因者标unknown。地下部关联、point count、accepted/rejected nodes、path count按所有40株记录。
11. **MDC95**：length3.72%、angle17.08°仅人工噪声practical reference；可报告matched路径落在参考范围内比例，但不是显著性门槛，不是equivalence/noninferiority margin。
12. curvature只exploratory，不能用于选赢家或淘汰方法。

## 7. 统计分析：植株单位，不能伪增 N

- Raw plant-level tables对40株×3方法全量展开，包含0输出、NA及GT状态，配套path-level/provenance表。不得删除难例。
- Primary paired contrasts=B−Teacher的植株宏平均measurable recall、存在性区间端点、possible-extra端点、complete recovery；conditional length/angle必须和recall/共同匹配覆盖一起解释。D−Teacher及D−B同法secondary，不做test后模型选择。
- 10,000 plant-cluster bootstrap；`numpy.random.Generator(PCG64(20260927))`，40plant有放回采样，三方法共享每个draw，保留一株内所有路径。95% percentile CI使用linear quantiles(0.025,0.975)。条件指标在每次draw内按锁定可评估植株计算；全NA draw记NA并报有效replicates，不把它填0。
- 点估计包括plant宏平均和median/IQR；path-pooled结果为补充并使用同一plant-cluster draws得到不确定性。上下界分别bootstrap，不把识别区间和CI混为一谈。
- test40株来自8个source frames/4个采集目录，不能保证生物/采集完全独立。预先增加**source-frame cluster sensitivity**：8个frame有放回抽样，带入该帧所有plant、共享方法draw，10,000次，独立PCG64(20260928)。报告与主plant分析是否一致，不据它重新选指标或方法；对只有4个目录的外推局限明确说明，不宣称external generalization。
- 不做事后多重“显著赢”筛选，无正式等价性/非劣效设计。CI包含0不证明equivalent；若Student明显退化，降低learnability/practical utility的论文主张，不重训。

## 8. 效率和资源：不以计时为理由多跑test

既有2026-09-25同机16株val benchmark（2warmup、10measured passes、每阶段N160）保留为已冻结效率证据，本轮不重跑val。

未来test仅**每方法每株一次**推理。在同一cv GPU、同一环境、batch1、eval/inference_mode下，沿用既有诊断inference RNG seed20260925，使用同一已有train图做两次warmup（不是test、不是额外图片），每方法独立进程，固定Teacher→B→D顺序；记录后台GPU使用，无空闲条件则在访问test前停止。每株一次保存stage timers：point-generation-only、shared graph+decoder、end-to-end phenotype（含geometry输出，不含模型加载/磁盘读入），CUDA前后同步；记录模型加载和图像I/O另表。median/IQR/N40和成对速度比，不能改成重复test benchmark。

GPU记录同定义 `max_memory_allocated`（reset峰值后包含保留模型内存）、`max_memory_reserved`；RAM沿用独立进程Linux VmRSS轮询，原5ms interval，报告process baseline/observed peak以及短峰可能漏采，不将observed RSS叫精确硬件峰值。三方法同一采样定义，不满足公平条件则memory标unavailable，不编造Student更省内存。测量结果差于原benchmark如实报告。

三阶段计时必须来自**同一次前向和下游执行**：生成点一次、复用这批点建图/解码一次、再算表型一次；end-to-end是这次完整执行的计时，不为了计时另跑一遍模型。Teacher内部原nine-view consensus属于一个锁定Teacher预测，不能把它换成single-view；Student没有该Teacher调用。

## 9. 预锁定论文例图选择

固定3株的Teacher/B同plant并排：高表现、典型（中位附近）、最差，D仅附图。

所有40株按唯一score tuple升序排名：`(min(Teacher measurable recall, B measurable recall), -sum(T/B possible-extra lower), -max(T/B per-plant conditional length mean))`。若G=0以存在性recall下界代替；仍无定义则recall=−1；任一方没有匹配长度则length排序值=+infinity（只用于选图，不是误差估计）。ties用SHA256(`V4-test-figure-20260927|`+dataset_id)升序，完整40名单和score公开。

最差=rank1；典型=rank20；高表现=rank40。三株都公开，不人工换漂亮个例，不按B单独排名；所谓高表现只是相对排序最高，不保证方法绝对成功。源图、GT、两法完整结果及失败表同图说明，不能只放成功枝条。

## 10. 停止规则和论文解释

test第一次模型读取后，冻结GT、matching、method、threshold、checkpoint、ROI、graph、decoder、geometry和split不再改变。结果再差也不复跑调参/重训、不把D选新主方法、不开发第二个interface。

正式统计按本协议全量输出，不设事后通过线。Student无须超过Teacher才构成learnability；结合结构/表型、coverage和已冻结效率证据决定主张强弱。当前不临时建立正式equivalence/noninferiority设计。如果表现不足，只能降低当前论文claim；未来第二代方法须使用另一批独立数据并另行批准。

本轮截止：提交本协议、静态scripts及PROJECT_STATE后停止。下一步是用户确认后准备模型盲test GT包；**不是训练，不是test inference**。

## 11. 静态检查入口与当前未完成门槛

同目录 `check_locked_test_static.py` 使用stdlib，只读明确列出的源码/checkpoint字节和CSV元数据；不import模型、不打开test图像、不执行训练/推理、不遍历用户额外数据。`test_locked_test_static.py`只用内存/临时目录合成fixture。

运行：`python check_locked_test_static.py`；schema扫描（将来正式公开测量包）：`python check_locked_test_static.py --public-package <measurement-package>`。脚本会显示prereg SHA和manifest身份核验状态；它的PASS只表示静态一致，不授予模型访问权限。

当前未完成：实际test测量包与浏览器盲法/dry-run、两人原始测量、GT冻结/hash、test evaluator实现/hash、远程受控同步、首次推理前用户授权。因此 **INFERENCE_GATE=CLOSED**；`test_model_reads=0`。

本轮验收：13项纯合成schema/元数据单元测试通过；本机冻结artifact、DINOv2 source commit/worktree、三组manifest静态核验PASS。D checkpoint本机缺失但远程已核验，预注册checker明确标注REMOTE_VERIFIED_NOT_LOCAL，而不是假装本机已验证。此验收不替代后续真实测量包的浏览器盲法验收。

V4 val method development permanently closed. Pipeline V1 and Student-B frozen. Awaiting model-blind V4 test GT. No test model inference has occurred.
