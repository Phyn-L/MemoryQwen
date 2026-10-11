# 三个日常运行入口

从激活项目 Python 环境开始。shell 只设置仓库运行目录、PYTHONPATH、可选的 CUDA_VISIBLE_DEVICES，并将参数交给 Python。`scripts/env.local.sh` 仍可设置本机 PATH/CUDA。实验设置写在 YAML 中，机器路径集中在 [machines.py](../../utils/machines.py)。

```bash
pip install -e '.[train]'
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 1. train.sh

```bash
bash scripts/train.sh --machine 4090 --config configs/train_baseline.yaml
bash scripts/train.sh --machine h200 --config configs/train_baseline.yaml
bash scripts/train.sh --machine h200 --config configs/train_baseline.yaml --resume outputs/<run>/last.pt
```

学习率、batch size、数据集、memory 大小、reader 开关、dtype、调度、日志和 checkpoint 路径均来自完整训练 YAML。恢复训练必须使用与原运行相符的配置。`<run>` 替换为实际路径。

训练任务组合由 YAML 控制：

```yaml
data:
  train_tasks: mix
  recon_ratio: 0.5
  recon_schedule: constant
  recon_schedule_curve: linear
  recon_ratio_start: 0.5
  recon_ratio_end: 0.5
  recon_ratio_min: 0.1
  max_input_context_length: 8192
```

`train_tasks: qa` 和 `train_tasks: recon` 分别执行单任务训练；`mix` 同时构造两类监督，并按动态 `recon_schedule` / `recon_ratio_*` 加权。QA 输入来自 JSONL 的 `conversations`，由 Qwen 原生 chat template 渲染；Recon 使用 `data.recon_prompt`，不需要向 tokenizer 添加 special token。

两阶段训练使用 `--init-from`：

```bash
bash scripts/train.sh --machine 4090 --config configs/train_recon.yaml
bash scripts/train.sh --machine 4090 --config configs/train_qa.yaml \
  --init-from outputs/<recon-run>/last.pt
```

`--init-from` 只加载模型的 trainable tensors，不恢复 optimizer、scheduler、global step、W&B run ID 或随机状态；`--resume` 才是同一运行的完整续训。

默认每张可见 GPU 一个进程；Python launcher 使用当前解释器的 torchrun，训练代码依据 YAML dtype 初始化 Accelerator。不再在 shell 固定 bf16 或实验超参数。旧 `CONFIG=...`、`MACHINE=...` 和 `NUM_PROCESSES=...` 命令仍可用；新命令推荐通过 CUDA_VISIBLE_DEVICES 选择 GPU。

## 2. checkpoint evaluator

```bash
python -m utils.launcher test --machine 4090 --ckpt outputs/<run>/last.pt --datasets hotpotqa --bs 2
python -m utils.launcher test --machine h200 --ckpt outputs/<run>/last.pt --config configs/test.yaml --bs 4
python -m utils.launcher test --machine 4090 --ckpt outputs/<run>/last.pt --datasets squad hotpotqa --split validation --bs 2
```

默认从 checkpoint 保存的配置恢复模型结构和预处理，默认 split 为 validation。`--machine` 将已知机器的模型/数据根目录替换为目标机器根目录，保留模型 snapshot revision 和子目录；不能推断的路径报错或要求显式完整配置。没有 config 的旧 checkpoint 必须提供完整训练 YAML。

测试 YAML 只含 `test:`（可选 `machine:`），设置数据集、split、batch size、QA batch size、生成长度和样本上限，不替换模型结构。CLI 优先于测试 YAML。仍支持 `--config <完整训练 YAML>` 的旧方式，此时由用户确保结构与 checkpoint 一致。

`--bs` / `--batch-size` 是每进程的 context batch；一个 context 可能有多个 QA，`--qa-batch-size` 控制内部 QA 分组。测试不继承训练时的 validation 样本上限，只有显式 `max_samples` 才限制数量。结果和完整解析配置保存到 checkpoint 所在目录的 `eval_<split>_<timestamp>.json`。多数据集同时选择时输出聚合分数；需要逐数据集分数时分别运行。

## 3. 独立 ICL evaluator

```bash
python -m utils.launcher icl --machine 4090 --model Qwen3-1.7B --config configs/icl_zeroshot.yaml --bs 2
python -m utils.launcher icl --machine h200 --model Qwen3-8B --datasets hotpotqa --config configs/icl_zeroshot.yaml --bs 4
```

shot 数、生成长度、输入长度、dtype、seed、chat template、数据 split 和输出目录均写入 ICL YAML；模型、数据集和 batch size 可由 CLI 覆盖。ICL 的 bs 是每 GPU 的 QA 数。

模型名在当前机器 MODEL_ROOT 中匹配普通模型目录或 Hugging Face `models--Qwen--<模型名>` 缓存目录；优先 `refs/main`，无 ref 时要求仅有一个 snapshot。支持本地已有的 Qwen family 名称（大小写不敏感）和完整路径，不自动下载、不任意选择多个 revision 中的一个。实际架构兼容性仍取决于安装的 transformers 版本。

结果包含逐 QA 预测、各数据集指标和 `run_config.json`。默认创建带模型与时间的输出目录；`--resume` 必须显式指定原 `--output-dir`。ICL 目前要求 CUDA。

## 数据与机器

| machine | MODEL_ROOT | DATA_ROOT |
|---|---|---|
| 4090 | `/data/lz/hf_cache/hub` | `/data/lz/contexts` |
| h200 | `/home/lijie/proj2/xmu/lz` | `/home/lijie/proj2/xmu/lz/contexts` |

训练数据采用 `<DATA_ROOT>/ift_c1qa.jsonl` 或 `<DATA_ROOT>/ift_pwc.jsonl` 的 conversation schema：每行包含 context + conversations。PWC 保留多轮 user/assistant 历史；C1QA 通常每行一个 assistant turn。独立 ICL 评估仍使用其自身配置的数据源。

两条路径均使用现有答案评分，不计算 supporting-fact/joint 指标。训练 memory 数据先按 `data.max_input_context_length` 过滤，再按 `data.max_context_tokens` 截断；ICL 的对应限制写在 YAML 的 `data.max_context_tokens`，默认 null。公平比较时需对齐这个限制与样本集合，不能只对齐 bs。

更多变体与旧名映射见[配置索引](../../configs/README.md)。原 A/B、8B、队列和 SQuAD 专用脚本暂作历史兼容工具；日常运行使用上述三个入口，不再新增专用变体 shell。旧脚本具有各自历史默认值，不等同于新预设。

当前推荐 checkpoint 编排入口为 `bash scripts/evaluation/test_all_suite.sh`。旧 shell wrappers 和 `scripts/archive/` 已删除；文中的历史配置路径需按 configs/README.md 选择当前配置。
