> 历史记录：文中 scripts/archive/ 脚本已删除，仅可从 Git 历史恢复；当前评测入口是 scripts/evaluation/test_all_suite.sh。

# 实验索引

整理日期：2026-09-18。以下状态来自已保存文档，本次没有读取远端训练进度或重算结果。“已记录”不等于本次复验；预测与待执行步骤保留为历史计划。配置明细见[配置索引](../../configs/README.md)，评分规则见[评测指南](../guides/EVALUATION.md)。

| 研究问题 | 配置 | 入口（仓库根目录） | 证据与状态 |
|---|---|---|---|
| 同形状下 reader ON/OFF 是否改善 QA？ | [ON](../../configs/4090/qwen-1.7b/reader/train_reader-on_ctx1024_m64.yaml)、[OFF](../../configs/4090/qwen-1.7b/reader/train_reader-off_ctx1024_m64.yaml) | `bash scripts/archive/run_ab.sh` | [A/B 记录](AB_H200.md)；包含不同轮次，需按上下文区分形状和训练量；[第三轮开头](ROUND3_PLAN.md)引用 `718b606` 的第二轮结果，远端终点待核实 |
| M=64→32→16 的影响？ | [M32](../../configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m32.yaml)、[M16](../../configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m16.yaml) | `CONFIG=... bash scripts/train.sh` 或 `QUEUE=m32,m16 bash scripts/archive/run_queue.sh` | [第三轮计划](ROUND3_PLAN.md)；文件存在，完成状态和最终结果待核实 |
| 增长 context 上限的影响？ | [ctx2048](../../configs/4090/qwen-1.7b/context_length/train_reader-on_ctx2048_m64.yaml) | `CONFIG=... bash scripts/train.sh` | [第三轮计划](ROUND3_PLAN.md)；非纯 memory-length 消融，完成状态待核实 |
| 8B ON 的规模效应？ | 由脚本基于 ON 生成配置 | `bash scripts/archive/run_on_8b.sh` | [第三轮计划](ROUND3_PLAN.md)；实际运行配置与结果待核实 |
| 4090 上运行全开 reader？ | [on_4090_all](../../configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m8.yaml) | `MACHINE=4090 CONFIG=configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m8.yaml bash scripts/train.sh` | YAML 有资源与训练设定说明；完成状态待核实 |
| memory 与 ICL 是否使用同一 SQuAD 口径？ | memory 从 checkpoint 恢复配置；ICL 用入口参数 | `CHECKPOINT=... bash scripts/archive/eval_squad_v1v2.sh` / `bash scripts/archive/eval_icl_squad.sh` | [评测异常](EVAL_ANOMALIES.md)；子集、过滤与多机差异是比较前提 |
| 早期 target/dtype 等修复影响？ | 见原实验记录 | 历史命令见原文 | [改进记录](../history/IMPROVEMENTS.md)、[工程记录](../history/ENGINEERING.md)，本次未重算 |

每次追加实验应记录：研究假设、代码 commit、完整配置、数据版本/过滤、seed、实际 QA 数、checkpoint/结果路径、指标和异常。旧文档内的输出路径是历史证据指针，不保证本地存在；本次未将预测值升级成最终结果。
