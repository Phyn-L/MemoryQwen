# 配置导航

先选择机器，再选择模型规模，最后选择 ablation 功能；训练参数全部由 YAML 声明。

```text
configs/
├── 4090/
│   ├── qwen-1.7b/{baseline,reader,memory_length,context_length,legacy}/
│   ├── qwen-4b/{baseline,legacy}/
│   ├── qwen-8b/{baseline,reader,legacy}/
│   ├── evaluation/
│   └── icl/
└── h200/                     # 同样的分类
```

- `baseline`：各模型规模的基础训练配置。
- `reader`：固定 ctx1024、m64 的 reader ON/OFF 配置。
- `memory_length`：m8、m16、m32 的 memory 长度变体；m64 参照 `reader` 中的 ON 配置。
- `context_length`：ctx2048 变体；ctx1024 参照 `reader` 中的 ON 配置。
- `evaluation`：checkpoint 测试协议。
- `icl`：数据集和 shot 数变体。
- `legacy`：旧别名副本及整理前不同内容的配置快照，保留用于复现实验。

```bash
bash scripts/train.sh --machine 4090 --config configs/4090/qwen-1.7b/baseline/train_baseline.yaml
bash scripts/train.sh --machine h200 --config configs/h200/qwen-1.7b/memory_length/train_reader-on_ctx1024_m32.yaml
bash scripts/test.sh --machine h200 --ckpt outputs/example.pt --config configs/h200/evaluation/test_hotpotqa.yaml
```

顶层仅保留两个机器目录及本说明。评测协议（evaluation）与 ICL 配置跨模型共享，放在机器目录下。旧模型、evaluation 和 icl 目录已移除，历史配置路径由配置加载器映射到新位置；新实验请使用机器目录下的路径。目录名称本身不会覆盖 `--machine`，运行时仍应显式指定机器。

本地工作区为准；在本地修改对应机器 YAML 后执行 `bash scripts/sync_h200.sh`，将两套配置共同同步到 H200。
