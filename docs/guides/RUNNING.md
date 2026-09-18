# 三个日常运行入口

从激活项目 Python 环境开始。shell 只设置仓库运行目录、PYTHONPATH、可选的 CUDA_VISIBLE_DEVICES，并将参数交给 Python。`scripts/env.local.sh` 仍可设置本机 PATH/CUDA。实验设置写在 YAML 中，机器路径集中在 [machines.py](../../utils/machines.py)。

```bash
pip install -e '.[train]'
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 1. train.sh

```bash
bash scripts/train.sh --machine 4090 --config configs/4090/qwen-1.7b/baseline/train_baseline.yaml
bash scripts/train.sh --machine h200 --config configs/4090/qwen-1.7b/memory_length/train_reader-on_ctx1024_m32.yaml
bash scripts/train.sh --machine h200 --config configs/4090/qwen-8b/reader/train_reader-on_ctx1024_m64.yaml --resume outputs/<run>/last.pt
```

学习率、batch size、数据集、memory 大小、reader 开关、dtype、调度、日志和 checkpoint 路径均来自完整训练 YAML。恢复训练必须使用与原运行相符的配置。`<run>` 替换为实际路径。

默认每张可见 GPU 一个进程；Python launcher 使用当前解释器的 torchrun，训练代码依据 YAML dtype 初始化 Accelerator。不再在 shell 固定 bf16 或实验超参数。旧 `CONFIG=...`、`MACHINE=...` 和 `NUM_PROCESSES=...` 命令仍可用；新命令推荐通过 CUDA_VISIBLE_DEVICES 选择 GPU。

## 2. test.sh

```bash
bash scripts/test.sh --machine 4090 --ckpt outputs/<run>/last.pt --datasets hotpotqa --bs 2
bash scripts/test.sh --machine h200 --ckpt outputs/<run>/last.pt --config configs/4090/evaluation/test_hotpotqa.yaml --bs 4
bash scripts/test.sh --machine 4090 --ckpt outputs/<run>/last.pt --datasets squad hotpotqa --split validation --bs 2
```

默认从 checkpoint 保存的配置恢复模型结构和预处理，默认 split 为 validation。`--machine` 将已知机器的模型/数据根目录替换为目标机器根目录，保留模型 snapshot revision 和子目录；不能推断的路径报错或要求显式完整配置。没有 config 的旧 checkpoint 必须提供完整训练 YAML。

测试 YAML 只含 `test:`（可选 `machine:`），设置数据集、split、batch size、QA batch size、生成长度和样本上限，不替换模型结构。CLI 优先于测试 YAML。仍支持 `--config <完整训练 YAML>` 的旧方式，此时由用户确保结构与 checkpoint 一致。

`--bs` / `--batch-size` 是每进程的 context batch；一个 context 可能有多个 QA，`--qa-batch-size` 控制内部 QA 分组。测试不继承训练时的 validation 样本上限，只有显式 `max_samples` 才限制数量。结果和完整解析配置保存到 checkpoint 所在目录的 `eval_<split>_<timestamp>.json`。多数据集同时选择时输出聚合分数；需要逐数据集分数时分别运行。

## 3. icl_baseline_test.sh

```bash
bash scripts/icl_baseline_test.sh --machine 4090 --model Qwen3-1.7B --config configs/4090/icl/icl_hotpotqa_4shot.yaml --bs 2
bash scripts/icl_baseline_test.sh --machine h200 --model Qwen3-8B --datasets hotpotqa --config configs/4090/icl/icl_hotpotqa_0shot.yaml --bs 4
bash scripts/icl_baseline_test.sh --machine 4090 --model Qwen3-4B-Instruct-2507 --config configs/4090/icl/icl_squad_4shot.yaml
```

`icl_test.sh` 是同一入口的简短别名。未传 `--config` 时使用 `configs/4090/icl/icl_squad_4shot.yaml`。shot 数、生成长度、输入长度、dtype、seed、chat template、数据 split 和输出目录均写入 ICL YAML；模型、数据集和 batch size 可由 CLI 覆盖。ICL 的 bs 是每 GPU 的 QA 数。

模型名在当前机器 MODEL_ROOT 中匹配普通模型目录或 Hugging Face `models--Qwen--<模型名>` 缓存目录；优先 `refs/main`，无 ref 时要求仅有一个 snapshot。支持本地已有的 Qwen family 名称（大小写不敏感）和完整路径，不自动下载、不任意选择多个 revision 中的一个。实际架构兼容性仍取决于安装的 transformers 版本。

结果包含逐 QA 预测、各数据集指标和 `run_config.json`。默认创建带模型与时间的输出目录；`--resume` 必须显式指定原 `--output-dir`。ICL 目前要求 CUDA。

## 数据与机器

| machine | MODEL_ROOT | DATA_ROOT |
|---|---|---|
| 4090 | `/data/lz/hf_cache/hub` | `/data/lz/contexts/aggregated` |
| h200 | `/home/lijie/proj2/xmu/lz` | `/home/lijie/proj2/xmu/lz/aggregated` |

所有数据采用 `<DATA_ROOT>/<dataset>/<split>.jsonl` 的 aggregated schema：每行 context + qa_pairs；HotpotQA 名称为 `hotpotqa`。当前本地 HotpotQA train/validation 有标注，test 为空，两类预设均使用 validation。ICL demonstrations 只从 train 取。

两条路径均使用现有答案评分，不计算 supporting-fact/joint 指标。memory 会按 checkpoint 的 context 长度规则过滤；ICL 的对应限制写在 YAML 的 `data.max_context_tokens`，默认 null。公平比较时需对齐这个限制与样本集合，不能只对齐 bs。

更多变体与旧名映射见[配置索引](../../configs/README.md)。原 A/B、8B、队列和 SQuAD 专用脚本暂作历史兼容工具；日常运行使用上述三个入口，不再新增专用变体 shell。旧脚本具有各自历史默认值，不等同于新预设。
