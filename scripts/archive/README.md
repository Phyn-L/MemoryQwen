# 归档专用实验脚本

这些脚本保留了历史 A/B、8B、队列以及专用 SQuAD 评测的命令和协议。它们不再是日常入口，也不应成为新实验变体的模板。新运行使用上级目录的 `train.sh`、`test.sh`、`icl_baseline_test.sh` 与 `configs/` YAML。

归档脚本仍保留原始运行逻辑，根目录定位已调整为适配 `scripts/archive/`。其中的默认 checkpoint、输出路径、机器数、数据状态和结果数字都属于历史快照，执行前必须逐项核对。

| 脚本 | 原用途 |
|---|---|
| `run_ab.sh` | H200 reader ON/OFF 顺序 A/B |
| `run_on_8b.sh` | 生成并启动 8B ON 配置 |
| `run_queue.sh` | 8B、压缩比和评测队列 |
| `eval_squad_v1v2.sh` | memory checkpoint 的 SQuAD v1/v2 专用评测 |
| `eval_icl_squad.sh` | ICL 与 memory 的 SQuAD 对照 |
| `test_icl_baseline.sh` | 早期 RACE 定向 ICL wrapper |
