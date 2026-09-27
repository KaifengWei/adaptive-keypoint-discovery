# V4 single-rater model-blind full-40 GT package audit

日期：2026-09-27。状态：**正式测量包准备完毕；原始人工标注、final GT及final test evaluator仍PENDING；INFERENCE_GATE=CLOSED。**

## 1. 行政修订及方法边界

唯一有效来源修订为同目录`V4_LOCKED_TEST_GT_SINGLE_RATER_AMENDMENT.md`，commit `e1d71d4dd08828f45d6d303e750523554402b2f6`。原协议`V4_LOCKED_TEST_PREREGISTRATION.md`保留原字节，不重写冻结历史；其人工来源相关要求由amendment正式覆盖。

由两人各40株改为**一名测量者、模型盲、全40株存在性和全部可测几何**。不新增第二测量者、共识或test-specific inter-rater可靠性。开发阶段MDC95（length 3.72%、angle 17.08°）仅作既有噪声reference，不重估、不作equivalence margin。

Teacher-direct、epoch53 Student-B、secondary Student-D、Pipeline V1、阈值/NMS、ROI、basal transition、graph、decoder、association、GT定义、`phenotype-geometry-v1`、matching、40株split及统计规则均未修改。本轮没有模型运行、训练、额外秧苗读取、final GT生成或final evaluator实现。

## 2. 先合成测试，再构建正式包

测量平台源码先于真实test打包冻结在commit `9ccb73bae38926b0a2e962701ec797fcae4c5609`。六个新文件位于`single_rater_gt/`，未覆盖旧pilot页面或任何实验流水线实现。

纯合成两图采用临时生成的RGB白底折线图，不是任何test/val植物图像。验收环境为Windows、Python3.11.0、Pillow10.2.0、Python Playwright1.62.0、headless Microsoft Edge152.0.4191.53。以下结果已实际执行，不是设计承诺：

| 检查 | 结果 |
|---|---|
| JavaScript `node --check` | PASS |
| raw schema/四状态/坐标/共享基点/尖端/修订/哈希单元测试 | 12/12 PASS |
| 两图真实浏览器操作、恢复和导出检查 | 18/18 PASS |
| 初始化无预填结构、动态添加、共同base、完整trace、tip、控制点拖动、undo | PASS |
| 四状态；非measurable无曲线/长度/角度 | PASS |
| 滚轮缩放、适合窗口、隐藏人工叠加 | PASS |
| 不完整表单阻止提交、正式提交立即SHA快照并锁定 | PASS |
| technical revision保存旧版本/hash链，禁止数量/状态/身份/信心等重判 | PASS |
| 草稿备份、刷新恢复、新浏览器恢复、锁定记录刷新恢复 | PASS |
| 恢复后导出raw JSON字节完全一致 | PASS |
| 恢复不得丢弃正式版本；损坏备份拒绝且不改现有记录 | PASS |
| raw JSON / sessions CSV / traces CSV / exact-file SHA一致性 | PASS |
| browser page/console errors | 0 |

第一次合成验证后增加了session身份与原协议说明，第二次仍仅在合成图上全量重跑通过；**这些修正均先于真实test包构建**。构建器要求合成PASS报告、六文件code hash和amendment hash完全一致，才可进入正式资产阶段；正式目标路径已存在即拒绝重建，没有第二套正式包。

私有合成验收报告：`runtime/v4_test_single_rater_20260927/admin/synthetic_browser_acceptance.json`（以`phenotype_pilot_protocol/`为相对根）。SHA-256为`74b01dba23a284f60dba6f4cdc1528479961059eef6e6fdf0a9e3ce535f207d9`。

## 3. 正式唯一40株包与盲法

测量入口：`D:\kp\adaptive_keypoint_discovery_reboot\experiment\phenotype_pilot_protocol\runtime\v4_test_single_rater_20260927\public\index.html`。

公共目录仅有`index.html`、`app.js`、`measurement_manifest.json`和40张`images/<blind_id>.png`，共43文件、4,443,948字节。HTML内嵌同一blind manifest，避免离线file://读取JSON的跨域限制；没有网络资源、自动外接框或模型结果。

公共schema严格沿用原协议：root仅`schema_version,samples`；sample仅`blind_id,image_filename,image_sha256,width,height`。ID用`secrets.choice`密码学安全随机源生成20位`[A-Z2-7]`，显示顺序用`secrets.SystemRandom`随机打乱；不编码原身份。40/40 ID唯一，文件名同ID。

只从冻结test.csv列明的40条路径原字节复制标准化RGB。40/40文件存在、复制前后与冻结expected image SHA一致、RGB/header width/height合法；无resize、重编码、内容QC、计数、难度排序、掩膜、模型forward、删图或换样本。

严格文件白名单与HTML/JS/JSON文本泄漏扫描PASS：没有原dataset身份、源路径/帧、方法名称、权重、点、路径、自动ROI/mask/skeleton、预期叶数、val结果或开发集合身份。page自身只绘制测量者新建的人工标记。

正式包随后用全新临时浏览器profile逐页**只打开资产**：40/40图片natural width/height与manifest一致，40张记录均base=null、items=[]、history=[]、未提交；annotations_created=0、console/page errors=0。未查看/截图/分析植株内容；未点击真实图描迹工具。验收后public完整inventory/hash不变，`results/`为空。该资产加载是用户授权的GT包访问，**不属于模型读取，也不声称test像素从未被打开**。

私有资产浏览器报告SHA-256：`120d543bab9ba44f0e844f7f118890258c65e8ecf4176d31e3c4966bd8f959ff`。

## 4. SHA-256登记

| 对象 | SHA-256 |
|---|---|
| 原预注册（未修改） | `31734d9dad03dc38d93a59cea8b55ea4eb22ac67854d3e506d813b34eef3b3d9` |
| single-rater amendment | `4329d946dc0f4902a6030a033f853b4d3887cf35ec0435fe773f0b83cfffa529` |
| 正式public package | `2c60a58f8dca9a60d73d03e9f9360f9650be0c2d7d492f47e8e291ab1156ebc5` |
| public manifest | `404b89b87b6ec3ea211069762de98718ad7ea47b3f55c54b6b06b45b74c87c45` |
| private admin mapping | `042e7f49f88041823f9608a79b2b051d39e2413ec1316a12d5d872fd36fa197f` |
| frozen source test manifest | `161f342037f4c982fb5f532bc60b18aadaad6656584ee667c7e0ca828818c5f6` |
| private package SHA ledger文件 | `bd6e68feb2fdf2fb3cea8ffd5eb2480ded8f5f402228ddf74b0aee55e23b5786` |

Package SHA不是ZIP SHA：定义为排序的public文件`path/bytes/sha256`清单、JSON sort_keys=True且compact separators的UTF-8字节SHA-256。完整43个asset hash及六文件code hash在私有`admin/SHA256_LEDGER.json`；ledger文件自身不进入public inventory，避免循环哈希。`.gitattributes`为本平台JS/HTML指定LF，使冻结源码跨设备checkout时不因换行变更字节。

private mapping在同一runtime的`admin/mapping.json`，与public互为兄弟目录，Git ignored；**不要把admin发给测量者，移交只发整个public目录**。private raw/GT也不能误提交Git。公开文档只登记hash，不公布映射。

## 5. 测量操作和提交记录

共同地上部基点→动态添加可见独立结构→选择四状态→measurable完整中心描迹；非可测仅可记tip/原因，points为空。初始状态无默认yes或GT编号，GT01…仅正式提交后按实际项目赋予。共享段和控制点不是插值；真实遮挡资格继续沿用`<=0.05D`及唯一平滑连接，不计算新的几何规则。

Raw snapshot保存blind/image/manifest SHA、session_id、rater=S1、round=1、revision、base_xy（共同地上部base）、实际结构、tip、备注、提交时间及previous_submission_sha256。snapshot JSON字符串与SHA一起保存，避免跨语言浮点序列化改变哈希；导出时原字符串不重写。每图正式提交立即version snapshot+SHA锁定。technical revision保留全部旧版本并链式引用前hash，禁止改变结构数量/状态/身份/信心/备注；坐标修订须明确纯技术原因。恢复不能抹除正式历史。

页面只保存raw控制点，不计算PCHIP、D、main-path、角度、物理单位或matching，不以页面中显示折线代替冻结geometry。最终GT后续仅由正式冻结raw、`phenotype-geometry-v1`和原deterministic规则生成。当前没有人工结果或final GT。

完成全部40张后逐一下载并保存四个文件：`raw_annotations.json`、`sessions.csv`、`traces.csv`、`SHA256_LEDGER.json`。保存目录为：

`D:\kp\adaptive_keypoint_discovery_reboot\experiment\phenotype_pilot_protocol\runtime\v4_test_single_rater_20260927\results\`

建议过程中另下载`annotation_progress.json`用于浏览器恢复，但它不是最终raw提交的替代。浏览器localStorage不是服务端持久归档；文件正式送达后仍需管理员按原字节归档/hash核验，不能因网页已显示提交而跳过这一门槛。

## 6. 关闭门槛与下一阶段

`test_model_reads=0`；`INFERENCE_GATE=CLOSED`。本轮新代码不导入任何模型/torch，也未运行Teacher-direct/Student-B/Student-D。冻结artifact/split/DINO source字节静态核验PASS；其中D沿用原远程核验登记，本机没有D权重，不冒充本机新核验。

下一阶段严格是：**用户完成40株并提交 → raw GT validation → final GT freeze/hash → final evaluator freeze → 用户显式授权 → 首次模型test inference**。本报告、包通过与GPU可连接均不构成推理授权；完成本任务即停止。
