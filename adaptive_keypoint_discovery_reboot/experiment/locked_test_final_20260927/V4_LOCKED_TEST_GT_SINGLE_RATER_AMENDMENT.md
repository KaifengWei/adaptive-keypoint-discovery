# V4 locked-test GT single-rater administrative amendment

日期：2026-09-27；版本：1.0；用户已在任何模型test inference之前明确批准。

## 1. 修订对象和范围

引用同目录原 `V4_LOCKED_TEST_PREREGISTRATION.md`，原提交`11e91348ccbc647f2092568485b0bcc97f3ff125`，原文件SHA-256：`31734d9dad03dc38d93a59cea8b55ea4eb22ac67854d3e506d813b34eef3b3d9`。原文件原样保留供审计；本amendment正式覆盖其§4.1双测量者工作量、§4.3双人显示顺序及§4.4双人共识/medoid来源选择要求，以及其他段落中的双人GT来源叙述，不覆盖任何算法或评价规则。

新的唯一人工来源为 **single-rater, model-blind, full-40 V4 test GT**。这是人工工作量和GT来源行政修订，不是方法修改。研究问题、四类GT状态定义、统计指标/区间、matching、split、frozen Teacher-direct/Student-B/Student-D、Pipeline V1、所有阈值及停止规则全部不变。

## 2. 正式GT来源与测量规模

1. 冻结V4 test的40/40株全部由一名测量者独立完成，覆盖全部40株存在性和所有可测结构完整几何；不抽样、不换株。
2. 人工测量在任何模型test inference之前完成。只允许标准化原始RGB与测量者自己创建的标记；不能看到Teacher/Student/自动点/路径/ROI/mask/骨架/val结果或预期数量。
3. 保持“可见且可独立追踪的地上部叶片结构路径”定义，不以农学叶龄、完全叶数量、固定点数或固定语义编号替代。
4. `measurable`：完整共同地上部base至独立远端中心描迹；正式长度名为`base-to-tip structural path length`，不能改成普通botanical leaf length。
5. `visible_unmeasurable`：保留结构存在性，不生成完整曲线、长度或角度；可以记录可辨识的tip/定位说明，但不能以该点伪造几何。
6. `uncertain`保持不确定，不强制投票式判定；`non_target_structure`可记录位置/原因但不进入目标存在性GT。
7. 共享主路径的重复描绘、中转控制点和“继续描迹”不算插值。真实短遮挡间隙仍沿用原规则，不能改变插值资格。

## 3. 明确取消的工作及论文披露

不安排第二测量者独立复核；不进行test-specific双人共识；不计算test-specific inter-rater reliability；不新增test重复轮次或重新估计MDC95。

开发阶段既有length MDC95=3.72%、divergence angle MDC95=17.08°只作为既有人工测量噪声reference，不是本test可靠性估计，不是正式equivalence margin，也不是显著性阈值。

论文不得将本test标注写作“双人共识GT”。正确表述为`single-rater model-blind phenotype reference`；只有测量者实际具备相应专业身份时才可用`model-blind expert reference annotation`，不能凭页面角色标签自动赋予expert资格。明确披露单测量者偏差无法由本test独立估计，开发阶段可靠性证据不能冒充test-specific inter-rater可靠性。

## 4. 原始记录提交、技术修订与冻结

每株正式提交后，立即生成不可覆盖的revision snapshot和SHA-256，保留image SHA、匿名身份、raw控制点、tip、状态、时间戳和previous submission hash。全40株完成时导出raw JSON、sessions CSV、traces CSV及文件SHA账本。浏览器本地保存/下载不是管理员持久归档的替代：用户应保存这些文件；收到后按原字节版本化保存并核验完整revision chain。

正式提交前允许正常描迹、撤销和调整。正式提交后不能因观察模型结果改变结构存在性、可测性、目标身份或人工轨迹判断。仅文件损坏、导出失败、坐标非法等**纯技术错误**可创建明确标记的technical revision，保留原版本、原因、前后hash和完整链；不是“再次审核并挑更像模型的曲线”。技术修订不得被用于更改状态/数量/身份。文件无法恢复时停止，由管理员记录，不自动补造标注。

最终GT仅由冻结人工记录、冻结`phenotype-geometry-v1`及原deterministic main-path/angle规则生成。测量平台只保存raw控制点，不另设Bezier/平滑/匹配算法；240-point PCHIP、D、main-path、angle和physical-unit派生只由冻结几何实现生成，未可靠测量的结构仍无曲线/长度/角度。单测量者来源不需要双人medoid候选选择。

本轮不生成final GT、不实现final test evaluator，不授权推理。原时间链严格保留：

`single-rater raw annotation → raw GT validation → final GT freeze + hash → final evaluator freeze → explicit inference authorization → 第一次模型读取V4 test`。

## 5. 测量包生成与盲法门槛

先只用纯合成fixture验证base、四状态、动态结构、raw描迹、tip、撤销、技术revision、备份恢复、JSON/CSV/SHA、刷新恢复、JS语法、schema与browser console。合成通过后才一次性生成唯一正式40株包；不能试标真实test。

公共目录仅原标准化RGB、HTML/JS与blind manifest；blind_id使用密码学安全随机源生成不少于16位opaque字符串，图片名同ID。私有映射/账本/审计记录存Git ignored runtime/admin，不随包发送。正式包生成后仅资产/尺寸/SHA/schema/泄漏/打开验收；不自动QC、计数、难度排序、mask、模型forward或淘汰样本。结果目录初始为空，页面记录全部空白，不预填GT01/存在数。

`test_model_reads = 0`；`INFERENCE_GATE = CLOSED`。Teacher-direct、Student-B、Student-D均不运行，不训练，不读取额外秧苗数据，不修改任何冻结方法/GT定义/评价规则。
