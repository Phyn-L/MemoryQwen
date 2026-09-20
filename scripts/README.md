# Scripts 使用指南

入口按用途组织，统一评测脚本位于 evaluation/。历史 archive 已删除，可通过 Git 历史查阅。

- `train.sh` / `train.py`：训练。
- `test.py`：统一入口依赖的 checkpoint evaluator。
- `evaluation/test_all_suite.sh`：统一 checkpoint 多数据集评测入口；`evaluation/test_icl_suite.py`：独立 ICL 基线。
- `experiments/`：研究消融与实验编排脚本。
- `data/`：数据下载与准备。
- `sync_h200.sh`：以本地为准同步 H200。

示例：

```bash
bash scripts/train.sh --machine 4090 --config configs/4090/qwen-1.7b/baseline/train_baseline.yaml
bash scripts/evaluation/test_all_suite.sh --checkpoint outputs/run/last.pt --datasets squad hotpotqa
```

递归扫描 checkpoint 并指定本次评测输出目录：

```bash
bash scripts/evaluation/test_all_suite.sh --checkpoint-dir outputs/checkpoints --datasets ms_marco_v1 ms_marco_v2 --output-dir outputs/evaluation/marco
```

`--output-dir` 直接使用指定目录，包含 manifest、日志以及按数据集/checkpoint 隔离的 JSON 报告；不追加时间戳目录。不指定时使用 `outputs/all_suite/<时间戳>`。`--output-root` 可改变自动目录的父目录，与 `--output-dir` 互斥。底层 `python -m utils.launcher test` 也支持 `--output-dir`。

## Zero-shot ICL

编辑 `evaluation/test_icl_suite.sh` 顶部的 MODEL、MACHINE、DATASETS、BATCH_SIZE、OUTPUT_DIR，或通过 CLI 覆盖：

```bash
conda run --no-capture-output -n shine bash scripts/evaluation/test_icl_suite.sh --model Qwen3-8B --machine 4090 --datasets ms_marco_v1 ms_marco_v2 --output-dir outputs/icl/marco
```

模型和数据根目录由 machine 自动解析。任务串行运行，每项使用可见 GPU 做数据并行；通过 CUDA_VISIBLE_DEVICES 和 NUM_PROCESSES 控制。固定 zero-shot，终端直接显示进度，输出逐条预测、单数据集 metrics/run_config 和总 summary.json。`--dry-run` 仅打印计划。Python 编排和 worker 推理均在 test_icl_suite.py，launcher 的内部 --worker 分支用于多进程执行。
