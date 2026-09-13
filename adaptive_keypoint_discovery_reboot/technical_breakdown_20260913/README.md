# DINOv2 自适应关键点算法技术拆解

交付：`DINOv2自适应关键点算法技术拆解.docx`，20页、9幅图。桌面另有同名副本。基准说明`C:/Users/F/Desktop/2.docx`未修改。

## 内容与证据

按原说明Stage 1—3组织技术解释，追踪ROI输入、ViT-S/14-Reg、全骨干冻结、G1′九视图伪标签、830401参数卷积头、损失与动态峰值解码、点条件图、局部尺度短枝规则、历史与正式表型定义以及人工GT门槛。

- `build_evidence.py`：只读取3张V4 val，使用冻结路线B checkpoint在CPU前向；另读取1张train展示存档伪标签。无训练、无test、无新方法表型比较。
- `build_diagrams.py`：按实际源码绘制机制图并绘制存档训练曲线。
- `build_document.py`：生成Word和桌面副本。
- `evidence.json`：checkpoint哈希、张量形状对应的映射、点坐标复核及训练样本原始伪标签。
- `assets/`：正文PNG以及SVG源图。真实中间张量NPZ本地保留但不提交Git，脚本可重建。
- `source_audit.json`：主要源文件的SHA-256及工作目录元数据盘点。

本次CPU点数为3/3/5，与历史坐标在数值精度内一致；置信度存在CPU/GPU小量差异。图中路径与人工结论继续引用冻结历史结果。不能将此项展示称为新的性能实验或最终验证。

## 阅读与渲染核验

本地历史会话`rollout-2026-07-13T21-15-35-019f5b9e-66d6-70d2-8c90-483b81b080db.jsonl`中的教师/学生、冻结与人工审核讨论用于恢复语境；文件名是会话建立日期，不用作其中后续消息的事件日期。未导出完整聊天或任何runtime人工GT。

本环境未暴露workspace dependency loader，打包的render_docx.py因缺少pdf2image无法运行。采用现有Python工具完成构建，Microsoft Word COM导出PDF、PyMuPDF渲染PNG，逐页检查全部20页。最终第2—18页与已检查上一版逐像素一致；第1、19、20页单独复核。检查版只保留本地qa目录，不作为交付或Git资料。

没有改动模型、配置、划分、阈值或评价规则；没有解盲或运行Teacher-direct/B/D表型比较。研究结论仍以PROJECT_STATE.md为准。
