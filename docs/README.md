# 文档导航

当前操作说明与历史实验快照分开维护。2026-09-18 完成第一阶段整理；代码和命令路径未变。

| 问题 | 文档 |
|---|---|
| 如何安装、训练、恢复和评测？ | [RUNNING](guides/RUNNING.md) |
| 评分代表什么，哪些数据被过滤？ | [EVALUATION](guides/EVALUATION.md) |
| 模型如何编码、读取 memory 和传播梯度？ | [METHOD](guides/METHOD.md) |
| reader 选项如何配置？ | [READER_OPTIONS](guides/READER_OPTIONS.md) |
| 应选哪份配置？ | [配置索引](../configs/README.md) |
| 接手本轮改造看什么？ | [交接文档](HANDOVER.md)（数据链、配置分层、评估/训练启动改造的动机、实测数据与遗留项） |

## 可视化快照

推荐先读 METHOD 获取当前实现，再查看 [QwenMemory 总览](visuals/qwenmemory_overview.html)（页面标注 2026-09-18，含 writer/reader/objectives；本次核对了核心流程，未逐条审计页面）。其可编辑图源为 [drawio 文件](visuals/qwenmemory_method.drawio)。

[早期 method overview](visuals/method_overview.html) 自报对应 commit `76bb625`，保留用于理解当时的 reader 与 A/B 设计，不作为最新实验状态。

历史文件保留当时数字和命令；其中旧文档路径、外部机器路径或未收录的论文报告是来源记录，不保证当前可用。当前文档路径以本导航为准。
