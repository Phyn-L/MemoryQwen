# 三个日常运行入口

从激活项目 Python 环境开始。shell 只设置仓库运行目录、PYTHONPATH、可选的 CUDA_VISIBLE_DEVICES，并将参数交给 Python。`scripts/env.local.sh` 仍可设置本机 PATH/CUDA。实验设置写在 YAML 中，机器路径集中在 [machines.py](../../utils/machines.py)。

```bash
pip install -e '.[train]'
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## SwanLab 私有实例

训练日志默认发送到自部署的 SwanLab，首次运行前配置私有服务地址及该实例的 API Key：

```bash
export SWANLAB_HOST="http://<私有服务主机>:<端口>"
read -rs 'SWANLAB_API_KEY?Private SwanLab API key: '
export SWANLAB_API_KEY
bash scripts/train.sh --machine 4090 --config configs/train_baseline.yaml
```

以上密钥输入命令适用于 zsh。也可提前使用 `swanlab login --host "$SWANLAB_HOST"`
交互登录私有实例。API Key 不写入 YAML、checkpoint 或版本库。
`SWANLAB_HOST` 优先于 `logging.swanlab_host`；在线模式缺少显式地址时直接报错。
H200 上训练时，地址必须能从 H200 访问，`localhost` 指向 H200 自己。

`logging.swanlab_project` 控制项目，`swanlab_workspace` 控制工作空间，
`swanlab_run_name` 控制看板中的实验名。
只由主进程上传指标，训练和验证曲线沿用原指标名，横轴为优化器步。
checkpoint 默认保存在 `outputs/{model-name}/{model-name}_{YYYYMMDD_HHMMSS}/`，
模型名沿用项目标签规则，例如 `Qwen1.7B`、`Qwen8B`。
`checkpoint.output_dir` 可设置输出根目录；运行时会追加模型名和带时间的运行目录。
`logging.swanlab_log_dir` 自动设为最终 checkpoint 目录，不再接受独立路径覆盖。

`--resume` 从 checkpoint 恢复 SwanLab run ID，并要求原实验存在；
`--swanlab-run-id <ID>` 可显式指定要续写的实验。

无服务连接时设置 `SWANLAB_MODE=offline` 保存本地日志，
或设置 `SWANLAB_MODE=disabled` 关闭实验记录；
也可在 YAML 中设置 `logging.swanlab_mode`，环境变量优先。

## 0. 数据

训练数据由 `scripts/data/` 构建，默认落在 `data.files.root`（`${DATA_ROOT:-contexts}`）下：

```bash
# 原始数据：本地 HF cache -> contexts/1qa/meta/（不依赖配置）
python scripts/data/build_1qa.py
python scripts/data/build_1qa.py --dry-run       # 只看每数据集/划分的统计，不写盘

# 过滤后的训练/验证集：过滤规则来自训练 YAML -> contexts/1qa/cache/
python scripts/data/build_cache.py --config configs/train_baseline.yaml
python scripts/data/build_cache.py --config configs/train_baseline.yaml --splits validation --dry-run
python scripts/data/build_cache.py --config configs/train_baseline.yaml --num-workers 16   # 默认 min(8, CPU)
```

手动构建时每个划分会显示一个进度条（按文件字节推进，postfix 是 `rows/contexts/qa_pairs`）；`--quiet` 或 `--no-progress` 关掉，非主进程（多卡训练自动补建时）本来就不画。

`scripts/train.py` 会在启动时自动补建缺失或过期的 cache（多卡只建一次），所以上面的第二条命令是可选的；数据根要被多个实验共用、或想先确认过滤后有多少 context 时，手动跑一次更省事。

**手动构建过 cache 之后，训练不会再重复过滤**：loader 会核对 `cache/build.json` 里的指纹（源 meta 文件 + tokenizer + 过滤规则），一致就直接读 `cache/train.jsonl`，日志里打印 `[data] train: train.jsonl is this config's filtered cache (loader-side re-filter skipped)`。指纹不一致（改了过滤参数、换了 tokenizer、或用了 `--data-file` 指向原始文件）才会退回旧路径：逐行过滤并在 `<data.root>/1qa/cache/loader/` 下另存一份（`data.preprocess_cache_dir` 可改到本地快盘；排序长度缓存 `sortish_lengths-*.json` 也放这里）。所以你之前看到的 `Filtering train.jsonl … / Reading train.jsonl …` 那两分钟（以及一份重复的 GB 级文件）只在指纹不匹配时才会出现。

1QA 划分从数据根解析为 `<root>/1qa/1qa-{train,validation,test}.jsonl`，可以用 `--data-root` 或 `data.files.<split>_conversation_file` 覆盖。旧的 `/data/lz/contexts/aggregated` 语料已弃用，不再出现在任何默认值里。

## 1. train.sh

```bash
bash scripts/train.sh --machine 4090 --config configs/train_baseline.yaml
bash scripts/train.sh --machine h200 --config configs/train_baseline.yaml
bash scripts/train.sh --machine h200 --config configs/train_baseline.yaml --resume outputs/<run>/last.pt
```

学习率、batch size、数据集、memory 大小、reader 开关、dtype、调度、日志和 checkpoint 路径来自配置：`configs/harness.yaml`（共享 harness）+ `configs/models/<模型>.yaml`（模型层）+ 实验文件（`_base_` 组合前两者，只写自己的差异）。恢复训练必须使用与原运行相符的配置。`<run>` 替换为实际路径。

`--resume` 在 sampler 层跳过已消费的 batch，这些 batch 不再访问 Dataset 或执行
collate/tokenization。多卡跳过在 Accelerate 分片之后进行，保留各 rank 的顺序和尾部补齐；
要复现原数据流，需保持相同的 batch size、梯度累积、GPU 数、seed 和数据配置。
启动时仍需读取 JSONL 数据集并恢复模型及 optimizer checkpoint。

显存不够时用梯度累积而不是缩小有效 batch：

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8   # 有效 batch = 2 × 8 × 卡数
```

`loss`/`scheduler`/`evaluation` 的间隔都按**优化器步**计（累积组算一步），checkpoint 也只在评估步保存。

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
  max_input_tokens: 40960
```

目标权重与蒸馏参数在 `loss:` 段（`loss.qa_weight` / `loss.recon_weight` / `loss.distill_*` / `loss.max_logits_rows`），`memory:` 只放结构参数；旧字段名会直接报错。`train_tasks: qa` 和 `train_tasks: recon` 分别执行单任务训练；`mix` 同时构造两类监督，并按动态 `recon_schedule` / `recon_ratio_*` 加权。QA 输入来自 JSONL 的 `conversations`，由 Qwen 原生 chat template 渲染；Recon reader 不输入 question token，直接从 memory prefix 预测 context。

两阶段训练使用 `--init-from`：

```bash
bash scripts/train.sh --machine 4090 --config configs/train_recon.yaml
bash scripts/train.sh --machine 4090 --config configs/train_qa.yaml \
  --init-from outputs/<recon-run>/last.pt
```

`--init-from` 只加载模型的 trainable tensors，不恢复 optimizer、scheduler、global step、SwanLab run ID 或随机状态；`--resume` 才是同一运行的完整续训。

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

1QA 训练数据采用 `<DATA_ROOT>/1qa/cache/train.jsonl` 与 `cache/validation.jsonl`（每个 context 一行，`qa_pairs` 里放该 context 的多个 QA），由 `<DATA_ROOT>/1qa/meta/meta_<split>.json` 按配置过滤而来；评测套件读 `<DATA_ROOT>/1qa/meta/<split>/<slug>.json`。只有 `configs/conversation_*.yaml` 仍使用 `<DATA_ROOT>/ift_c1qa.jsonl` / `ift_pwc.jsonl` 这种 conversation schema：每行包含 context + conversations。PWC 保留多轮 user/assistant 历史；C1QA 通常每行一个 assistant turn。独立 ICL 评估仍使用其自身配置的数据源。

**评估抽样确定、解码贪心**：QA 评估按 `training.seed` 对每个数据集独立随机抽取至多 `evaluation.max_eval_sample_per_dataset` 条，无合计条数上限；Recon 复用入选 QA 涉及的去重 context。解码一律 `argmax`（贪心，遇 EOS 停止），没有 temperature/top-p/beam。同一配置和 checkpoint 的评估样本固定且可复现。

两条路径均使用现有答案评分，不计算 supporting-fact/joint 指标。训练 memory 数据先按 `data.max_input_tokens` 的模型窗口检查，再按 `data.max_context_tokens` 截断；ICL 的对应限制写在 YAML 的 `data.max_context_tokens`，默认 null。公平比较时需对齐这个限制与样本集合，不能只对齐 bs。

更多变体与旧名映射见[配置索引](../../configs/README.md)。原 A/B、8B、队列和 SQuAD 专用脚本暂作历史兼容工具；日常运行使用上述三个入口，不再新增专用变体 shell。旧脚本具有各自历史默认值，不等同于新预设。

当前推荐 checkpoint 编排入口为 `bash scripts/evaluation/test_all_suite.sh`。旧 shell wrappers 和 `scripts/archive/` 已删除；文中的历史配置路径需按 configs/README.md 选择当前配置。
