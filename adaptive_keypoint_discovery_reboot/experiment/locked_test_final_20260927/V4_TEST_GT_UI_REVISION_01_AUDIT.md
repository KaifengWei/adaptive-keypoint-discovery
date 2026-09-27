# V4 test GT measurement UI technical revision 01

日期：2026-09-27；用户报告页面缩放与描迹启动问题后授权修复。实现commit：`527748f296998f32f888d2767661732c5163d20d`。

## 原因与范围

初始`适合窗口`只依据viewport宽度计算缩放，因此竖长图会溢出高度；换图时还未显式重置滚动位置。描迹逻辑要求已有共同基点、已添加并选中的结构和用户明确选择`measurable`，但缺项直到图上点击才提示，且提示放在图片下方，不易看到。这不是叶片状态应预填为yes，也不授权跳过四状态判断。

本次只修测量页面显示/操作，不改任何GT资格、geometry、matching、模型、阈值、association、graph、decoder或split。修改同一正式包的`index.html/app.js`，不另造测量包。

## 修复

- 默认及`全图适合窗口`同时按原图width/height与viewport宽高计算等比缩放，完整图居中；换图/全图按钮重置zoom和scroll。仅使用manifest资产尺寸，不计算前景、bbox、mask或内容QC。
- 滚轮围绕鼠标缩放，工具旁显示全图倍率与相对原像素显示比例。放大只影响显示，原始坐标仍用原图像素逆映射。
- 操作提示移至工具栏下、图片上方。继续描迹立即检查并提示缺共同基点、缺当前结构或缺状态，不再让点击表现为无响应。
- 添加结构仍创建状态为空的项目，不预填可测或存在性。主动聚焦并高亮右侧状态下拉框；**只有用户明确选择可测**才进入描迹模式。非可测项目仍不能产生完整曲线。
- 不改变raw schema、提交锁定/SHA、技术revision规则、身份或测量轮次。不接触用户浏览器profile/localStorage，不清缓存、不迁移或重写用户GT。

## 验证：合成先行

全部修复验证先在纯合成数据执行，未拿真实test做试标：

1. 原两图12项schema测试、18项Edge操作/备份恢复/JSON与CSV精确SHA测试全量重跑PASS。
2. 增加横长`3400×400`和竖长`400×3400`合成RGB图，13项UI回归PASS：两种方向完整适配、缺项提示可见、不自动填状态、明确可测后进入描迹、raw坐标对应原图像素逆映射、缩放不改记录、换图清滚动、resize适配、导出验证和旧版记录兼容。
3. 真实鼠标事件可能有屏幕坐标量化；回归按实际pointer坐标验证原图逆映射，不把理想的亚像素鼠标目标假装成浏览器实际点击。缩放/resize前后已保存坐标和锁定hash完全不变。
4. 旧版本**合成**已提交记录先在旧页面恢复，再在同路径更新HTML/JS和刷新；localStorage内容与修订链完全一致，raw JSON导出字节一致，证明不需要清缓存或重标。
5. JS语法PASS；上述浏览器page/console errors=0。

合成PASS及全部code SHA匹配后才应用正式包UI修订。正式40株随后只做public资产/泄漏及新浏览器加载验收：40/40正常加载，模型读取0、annotations_created=0、console errors=0；未分析植物内容，未在真实图上试画。

## 资产与哈希连续性

旧HTML、JS、旧package ledger、原合成报告及原资产浏览器报告均原字节保留在Git ignored private `runtime/v4_test_single_rater_20260927/admin/ui_revision_01/previous/`。新revision record和合成验收报告在同级`ui_revision_01/`。完整修订链不是覆盖旧包后抹除记录。

| 对象 | 登记 |
|---|---|
| 上一public package SHA | `2c60a58f8dca9a60d73d03e9f9360f9650be0c2d7d492f47e8e291ab1156ebc5` |
| 当前public package SHA | `83f0879cfd744eb161ea5d45b1b1b65a0e1d0437eb20ea6fb5c32ba03ca4844b` |
| blind manifest SHA（不变） | `404b89b87b6ec3ea211069762de98718ad7ea47b3f55c54b6b06b45b74c87c45` |
| private mapping SHA（不变） | `042e7f49f88041823f9608a79b2b051d39e2413ec1316a12d5d872fd36fa197f` |
| app.js SHA | `81a8321a3d8305e69185e495526b407e335b03a675bbc8995dfc5f395f4fd6ac` |
| 图片 | 40/40原字节/hash不变 |
| ID、顺序、存储key | 全部不变 |

Package SHA继续使用原排序文件inventory定义。当前完整清单更新于原`admin/SHA256_LEDGER.json`，含previous_package_sha256；旧清单已保留。amendment与原预注册未修改。

入口仍是：`D:\kp\adaptive_keypoint_discovery_reboot\experiment\phenotype_pilot_protocol\runtime\v4_test_single_rater_20260927\public\index.html`。用户先点`备份进度JSON`，再刷新原页面即可；不要清除浏览器存储。操作顺序：**设置共同基点→添加结构→右侧主动选择状态→可测时沿中心线逐点描迹**。已有草稿/正式记录继续沿用，不因UI更新重做。

`test_model_reads=0`；`INFERENCE_GATE=CLOSED`。raw/final GT/evaluator仍待原流程完成，未推理、未训练。
