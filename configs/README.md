# 配置导航

先选择机器，再选择实验内容；模型规模放在第三层。训练参数全部由 YAML 声明。

```text
configs/
├── 4090/
│   ├── baseline/qwen-{1.7b,4b,8b}/
│   ├── reader/qwen-{1.7b,8b}/
│   ├── memory_length/qwen-1.7b/
│   ├── context_length/qwen-1.7b/
│   ├── evaluation/
│   ├── icl/
│   └── legacy/
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
bash scripts/train.sh --machine 4090 --config configs/4090/baseline/qwen-1.7b/train_baseline.yaml
bash scripts/train.sh --machine h200 --config configs/h200/memory_length/qwen-1.7b/train_reader-on_ctx1024_m32.yaml
bash scripts/test.sh --machine h200 --ckpt outputs/example.pt --config configs/h200/evaluation/test_hotpotqa.yaml
```

原 `qwen-*`、`evaluation`、`icl` 目录只保留兼容软链接，无前缀路径指向 4090 配置；新实验请使用机器目录下的路径。目录名称本身不会覆盖 `--machine`，运行时仍应显式指定机器。

本地工作区为准；在本地修改对应机器 YAML 后执行 `bash scripts/sync_h200.sh`，将两套配置共同同步到 H200。
