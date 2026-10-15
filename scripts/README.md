# Scripts 使用指南

入口按用途组织，统一评测脚本位于 evaluation/。历史 archive 已删除，可通过 Git 历史查阅。

- `train.sh` / `train.py`：训练。
- `test.py`：统一入口依赖的 checkpoint evaluator。
- `evaluation/test_all_suite.sh`：统一 checkpoint 多数据集评测入口；`evaluation/test_icl_suite.py`：独立 ICL 基线。
- `experiments/`：研究消融与实验编排脚本。
- `data/build_1qa.py`：从本地 HF cache 重建原始 `1qa/meta/`（不依赖配置）。
- `data/build_cache.py`：按训练 YAML 过滤出 `1qa/cache/{train,validation}.jsonl`（训练启动时会自动补建）。
- `data/`：数据下载与准备。
- `sync_h200.sh`：以本地为准同步 H200。

示例：

```bash
bash scripts/train.sh --machine 4090 --config configs/4090/qwen-1.7b/baseline/train_baseline.yaml
bash scripts/evaluation/test_all_suite.sh --checkpoint outputs/run/last.pt --datasets squad hotpotqa
```

递归扫描 checkpoint 并指定评测输出根目录：

```bash
bash scripts/evaluation/test_all_suite.sh --ckpt-dir outputs/checkpoints --datasets ms_marco_v1 ms_marco_v2 --output-dir outputs/eval_ab_reader
```

`--output-dir` 只指定根目录（默认 `outputs/eval_ab_reader`），脚本根据 checkpoint 文件名自动创建子目录。例如 `base.pt` 的报告直接位于 `outputs/eval_ab_reader/base/squad.json`、`base/msmarcov1.json`、`base/msmarcov2.json`。checkpoint 目录只保留数据集 JSON，不创建数据集子目录、日志文件或 manifest。评测只运行 autoregressive 自由生成。全部 checkpoint 完成后，根目录的 `summary.csv` 按 checkpoint 汇总各数据集的 AR EM/F1/ROUGE-L 百分数（两位小数）。子进程输出直接显示在终端，因此 tqdm 进度条仍然可见。底层 `python -m utils.launcher test` 也支持 `--output-dir`。

## Zero-shot ICL

MemoryQwen suite 支持 `--system-prompt-mode separate|context|none`（shell 环境变量 `SYSTEM_PROMPT_MODE`，suite 默认 `separate`，独立 `scripts/test.py` 默认 `none`）。复用 ICL 的无数据集名称 system prompt，并根据 MS-MARCO source version 选择版本。

- `context`：编码 `system prompt + 原文` 为 memory slots；问题保持 checkpoint 的原有格式，仅支持 AR 评测。
- `separate`：原文单独编码为 memory slots；问题强制使用 chat template，输入 system + user question，关闭 thinking，与 ICL 的 system 文案一致。
- `none`：旧评测行为。

原文过滤和截断仍按 `max_context_tokens`；`context` 模式在此之外增加 prompt token。`separate` 在原有问题预算外增加 system/chat 开销，超长问题会报错以避免截掉 chat 边界。结果 JSON 的 `prompt_protocol` 记录模式、source version 和完整 prompt。不同模式请使用不同输出目录：

```bash
bash scripts/evaluation/test_all_suite.sh --system-prompt-mode context --output-dir outputs/eval_prompt_context
bash scripts/evaluation/test_all_suite.sh --system-prompt-mode separate --output-dir outputs/eval_prompt_separate
```

这里的“一致”指 system message 文案与 chat 渲染策略；MemoryQwen 仍通过 memory prefix 接收原文，不复用 ICL 的全文 user prompt。RACE 的原有 MemoryQwen 数据路径未提供选项和字母评分，不能仅靠此开关得到与 ICL 一致的选择题评测。

编辑 `evaluation/test_icl_suite.sh` 顶部的 MODEL、MACHINE、DATASETS、BATCH_SIZE、OUTPUT_DIR，或通过 CLI 覆盖：

```bash
conda run --no-capture-output -n shine bash scripts/evaluation/test_icl_suite.sh --model Qwen3-8B --machine 4090 --datasets ms_marco_v1 ms_marco_v2 --output-dir outputs/icl/marco
```

模型和数据根目录由 machine 自动解析。任务串行运行，每项使用可见 GPU 做数据并行；通过 CUDA_VISIBLE_DEVICES 和 NUM_PROCESSES 控制。固定 zero-shot，终端直接显示进度，输出逐条预测、单数据集 metrics/run_config 和总 summary.json。`--dry-run` 仅打印计划。Python 编排和 worker 推理均在 test_icl_suite.py，launcher 的内部 --worker 分支用于多进程执行。

## 4090 → 计算服务器同步

代码和 YAML 均在 4090 修改并提交，H200/A800 只运行计算任务：

```bash
bash scripts/sync_remote.sh h200
bash scripts/sync_remote.sh a800
bash scripts/sync_remote.sh all
```

无需 GitHub 网络或先 push。脚本读取目标 HEAD，确认是本地 HEAD 的祖先，生成增量 Git bundle，经 SSH 传输后快进更新。包含所有已提交文件的新增、修改和删除（包括 YAML），不传输未跟踪数据、模型、输出或未提交修改。相同 commit 不传输 bundle。all 会尝试两台机器，任一失败最终返回非零。

本地有未提交修改时先停止；远端有修改、历史分叉或目标提交不在本地历史时也停止，不自动 stash/reset。远端需要保留的代码/配置变更应先整理回 4090，再提交同步；机器路径通过 --machine 指定，不在计算服务器修改公共 YAML。同步前应确认没有正在运行的任务依赖待更新文件；脚本不调度或重启计算任务。

默认地址为 h200:/home/lijie/proj2/xmu/lz/MemoryQwen 和 A800:/mnt/beegfs-storage-a8002/seanlz/MemoryQwen。可用 H200_SSH_HOST、H200_REPO、A800_SSH_HOST、A800_REPO 覆盖。需要 SSH 非交互认证及远端 Git/bash/flock。旧 sync_h200.sh/sync_a800.sh 使用 GitHub 拉取；日常统一使用 sync_remote.sh。
