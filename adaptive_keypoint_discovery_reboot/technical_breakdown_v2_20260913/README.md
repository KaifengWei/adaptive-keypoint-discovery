# DINOv2 自适应关键点技术交接详解 修订版

交付文件为同目录的 `DINOv2自适应关键点技术交接详解_修订版.docx` 与同名 PDF，35 页、36 幅图。

本版按用户桌面 `2.docx` 新增的分层模块示例重绘机制图，四种早期候选方式各自解释；补入首轮学生、数据失败、ROI、短叶、共识及点条件图的真实图文。原文档未覆盖。

构建顺序：`make_evidence.py` → `make_details.py` → `make_architecture.py` → `make_readable_panels.py` → `write_document.py`。依赖现有实验代码、锁定DINOv2权重、原始图像及上一版可复现张量；不是自包含实验包。

`consensus_walkthrough.csv` 提供16个原图候选的逐点筛选。`walkthrough.json` 提供归档坐标一致性和点移除数据。`teacher_identity_walkthrough.json` 为原/增强教师单视图提案，不能当作共识标签。`figure_index.json` 和 `delivery_audit.json` 保存图件路径、来源与文件哈希。

文档使用 Word COM 导出 PDF，再以 PyMuPDF 逐页渲染检查。QA图片、参考样图和中间数组留在本地，不进入版本控制。实验模型、历史标签与评估输出保持原样；V4 test 未读取。
